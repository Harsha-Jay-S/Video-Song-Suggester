"""
Describe what's happening in a video using CLIP (openai/clip-vit-base-patch32).

Workflow:
  1. Reads JPG frames from a directory (produced by extract_frames.py).
  2. Scores each frame against three phrase lists: scenes, actions, moods.
  3. Applies a minimum confidence threshold to discard noisy predictions on
     unusual content (e.g. screen recordings that CLIP wasn't trained on).
  4. Prints the top-3 per category per frame for transparency.
  5. Builds a richer description with per-frame confidence breakdown,
     then emits it to stdout so suggest_song_attributes can work from
     detailed data rather than a single aggregated label.

Usage:
    python scripts/describe_video.py [frames_dir] [--verbose]
    python scripts/describe_video.py [frames_dir] --no-gpu  # force CPU
    python scripts/describe_video.py [frames_dir] --gpu      # force GPU
"""

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from _device import add_gpu_args, resolve_device, DeviceConfig


MODEL_ID = "openai/clip-vit-base-patch32"
MIN_CONFIDENCE = 0.30
CACHE_DIR = Path.home() / ".cache" / "video-frame-extractor"
CACHE_FILE = CACHE_DIR / "frame_cache.json"


SCENE_PHRASES: List[str] = [
    "a dark screen or logo on start-up",
    "a bright logo animation on a dark background",
    "text or icons displayed on a screen",
    "a static screen with no clear content",
    "a company logo on a solid background",
    "a city street at night",
    "a city street during the day",
    "a quiet forest",
    "a sandy beach with ocean waves",
    "snow-covered mountains",
    "a cozy indoor room",
    "a busy office",
    "a concert stage with bright lights",
    "a sports field with players",
    "a desert landscape",
    "a rainy street",
    "a sunset over water",
    "a park with trees and grass",
    "a highway with cars driving",
    "a cafe with people sitting",
    "a classroom or lecture hall",
    "a wedding celebration",
    "a birthday party",
    "an empty room",
]

ACTION_PHRASES: List[str] = [
    "a static screen with no action",
    "a logo or splash screen slowly appearing",
    "text or icons changing on screen",
    "nothing happening, a completely still screen",
    "people walking",
    "people running",
    "people dancing",
    "people talking to each other",
    "people eating a meal",
    "people playing a sport",
    "people driving a car",
    "people riding bicycles",
    "people swimming",
    "a person cooking food",
    "a person playing a musical instrument",
    "a person singing",
    "a person working on a computer",
    "a person reading a book",
    "people hugging",
    "people laughing",
    "a crowd cheering",
    "an animal moving around",
    "nothing much happening, a still scene",
    "fast action and motion",
]

MOOD_PHRASES: List[str] = [
    "mysterious or eerie",
    "serious and focused",
    "a happy and upbeat moment",
    "a calm and peaceful moment",
    "a sad and emotional moment",
    "a tense and dramatic moment",
    "a romantic moment",
    "an energetic and exciting moment",
    "a nostalgic moment",
    "a playful and fun moment",
]


@dataclass
class FrameReport:
    path: str
    timestamp: float
    scene_scores: List[Tuple[str, float]] = field(default_factory=list)
    action_scores: List[Tuple[str, float]] = field(default_factory=list)
    mood_scores: List[Tuple[str, float]] = field(default_factory=list)
    top_scene: Optional[Tuple[str, float]] = None
    top_action: Optional[Tuple[str, float]] = None
    top_mood: Optional[Tuple[str, float]] = None
    complexity_score: float = 0.0


def _get_image_hash(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_frame_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            import json as _json
            with open(CACHE_FILE) as f:
                return _json.load(f)
        except Exception:
            pass
    return {}


def save_frame_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import json as _json
        with open(CACHE_FILE, "w") as f:
            _json.dump(cache, f)
    except Exception:
        pass


def compute_complexity(scene_scores: List[Tuple[str, float]], action_scores: List[Tuple[str, float]]) -> float:
    scene_entropy = 0.0
    for _, p in scene_scores:
        if p > 0:
            scene_entropy -= p * (p ** 0.5)
    action_entropy = 0.0
    for _, p in action_scores:
        if p > 0:
            action_entropy -= p * (p ** 0.5)
    max_scene = max((s for _, s in scene_scores), default=0.0)
    max_action = max((s for _, s in action_scores), default=0.0)
    return (1.0 - max_scene) + (1.0 - max_action) + (scene_entropy + action_entropy) * 0.5


def load_model(device_cfg: DeviceConfig) -> Tuple[CLIPModel, CLIPProcessor]:
    print(f"Loading CLIP model '{MODEL_ID}' on {device_cfg.device}...", file=sys.stderr)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device_cfg.device)
    if device_cfg.use_gpu:
        model = model.half()
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    return model, processor


def score_frames_batch(
    image_paths: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int = 8,
) -> List[Tuple[List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]]:
    all_scene_scores = []
    all_action_scores = []
    all_mood_scores = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        scene_inputs = processor(text=SCENE_PHRASES, images=images, return_tensors="pt", padding=True).to(device)
        action_inputs = processor(text=ACTION_PHRASES, images=images, return_tensors="pt", padding=True).to(device)
        mood_inputs = processor(text=MOOD_PHRASES, images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            if device.type == "cuda":
                scene_inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in scene_inputs.items()}
                action_inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in action_inputs.items()}
                mood_inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in mood_inputs.items()}

            scene_out = model(**scene_inputs)
            action_out = model(**action_inputs)
            mood_out = model(**mood_inputs)

        scene_probs_list = scene_out.logits_per_image.softmax(dim=1).cpu().tolist()
        action_probs_list = action_out.logits_per_image.softmax(dim=1).cpu().tolist()
        mood_probs_list = mood_out.logits_per_image.softmax(dim=1).cpu().tolist()

        for sp, ap, mp in zip(scene_probs_list, action_probs_list, mood_probs_list):
            all_scene_scores.append(list(zip(SCENE_PHRASES, sp)))
            all_action_scores.append(list(zip(ACTION_PHRASES, ap)))
            all_mood_scores.append(list(zip(MOOD_PHRASES, mp)))

    return list(zip(all_scene_scores, all_action_scores, all_mood_scores))


def top_n(
    scored: List[Tuple[str, float]],
    n: int = 1,
    min_conf: float = 0.0,
) -> List[Tuple[str, float]]:
    above = [s for s in scored if s[1] >= min_conf]
    above.sort(key=lambda x: x[1], reverse=True)
    return above[:n]


def format_top3(
    scored: List[Tuple[str, float]],
    label: str,
    indent: str = "  ",
) -> str:
    top3 = top_n(scored, n=3)
    items = " | ".join(f"'{p}' ({s:.2f})" for p, s in top3)
    return f"{indent}{label}: {items}"


def describe(frames_dir: str, verbose: bool = False, device_cfg: Optional[DeviceConfig] = None) -> None:
    if device_cfg is None:
        device_cfg = resolve_device(None)

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_paths = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_paths:
        raise RuntimeError(f"No image frames found in {frames_dir}")

    frame_cache = load_frame_cache()
    new_hashes = {}
    for p in frame_paths:
        h = _get_image_hash(p)
        new_hashes[p] = h
        if h in frame_cache:
            continue

    paths_to_analyze = [p for p in frame_paths if new_hashes[p] not in frame_cache]
    paths_cached = [p for p in frame_paths if new_hashes[p] in frame_cache]

    if paths_cached:
        print(f"Using {len(paths_cached)} cached frames, analyzing {len(paths_to_analyze)} new frames.", file=sys.stderr)

    model, processor = load_model(device_cfg)

    frame_reports: List[dict] = []
    scene_counter: Counter = Counter()
    action_counter: Counter = Counter()
    mood_counter: Counter = Counter()

    print(f"\nAnalyzing {len(paths_to_analyze)} frames...", file=sys.stderr)

    cached_results = {p: frame_cache[new_hashes[p]] for p in paths_cached}

    all_scores = []
    if paths_to_analyze:
        scores_batch = score_frames_batch(
            paths_to_analyze, model, processor, device_cfg.device, batch_size=8
        )
        for p, scores in zip(paths_to_analyze, scores_batch):
            scene_scores, action_scores, mood_scores = scores
            img_hash = new_hashes[p]
            frame_cache[img_hash] = {
                "scene": scene_scores,
                "action": action_scores,
                "mood": mood_scores,
            }
            all_scores.append((p, scene_scores, action_scores, mood_scores))

    for p in tqdm(frame_paths, desc="Processing frames", unit="frame", disable=not sys.stderr.isatty()):
        if p in cached_results:
            cached = cached_results[p]
            scene_scores = cached["scene"]
            action_scores = cached["action"]
            mood_scores = cached["mood"]
        else:
            continue

        ts = float(Path(p).stem.split("_t")[1].rstrip("s")) if "_t" in Path(p).stem else 0.0

        top_scene = top_n(scene_scores, n=1)[0]
        top_action = top_n(action_scores, n=1)[0]
        top_mood = top_n(mood_scores, n=1)[0]
        complexity = compute_complexity(scene_scores, action_scores)

        if top_scene[1] >= MIN_CONFIDENCE:
            scene_counter[top_scene[0]] += 1
        if top_action[1] >= MIN_CONFIDENCE:
            action_counter[top_action[0]] += 1
        if top_mood[1] >= MIN_CONFIDENCE:
            mood_counter[top_mood[0]] += 1

        frame_reports.append(
            f"Frame {len(frame_reports)+1}/{len(frame_paths)}:\n"
            + format_top3(scene_scores, "scene", "  ")
            + "\n"
            + format_top3(action_scores, "action", "  ")
            + "\n"
            + format_top3(mood_scores, "mood", "  ")
            + f"\n  winner: scene='{top_scene[0]}' ({top_scene[1]:.2f})"
            + f" action='{top_action[0]}' ({top_action[1]:.2f})"
            + f" mood='{top_mood[0]}' ({top_mood[1]:.2f})"
            + f" complexity={complexity:.3f}",
        )

    save_frame_cache(frame_cache)

    if verbose or True:
        print(f"\n{'='*40}\nDetailed per-frame scores:", file=sys.stderr)
        for r in frame_reports:
            for line in r.split("\n"):
                print(line, file=sys.stderr)
            print(file=sys.stderr)

    if not scene_counter or not action_counter:
        summary = (
            "The video does not match well against known scene or action "
            "categories. This may be screen content (logo, splash, animation) "
            "that CLIP was not trained on. Try reviewing the per-frame scores above."
        )
    else:
        top_scenes = [p for p, _ in scene_counter.most_common(2)]
        top_actions = [p for p, _ in action_counter.most_common(2)]
        top_moods = [p for p, _ in mood_counter.most_common(1)]

        parts = [f"Across {len(frame_paths)} frames, the video appears to show "]
        if top_scenes:
            parts.append(" and ".join(top_scenes) + ". ")
        if top_actions:
            parts.append("It mostly features " + " and ".join(top_actions) + ". ")
        if top_moods:
            parts.append("The overall mood feels like " + top_moods[0] + ".")
        summary = "".join(parts)

    per_frame_block = "\n\n".join(frame_reports)

    print("\n=== Plain-text description ===")
    print(summary)
    print("\n=== Per-frame score breakdown ===")
    print(per_frame_block)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Describe a video from extracted frames using CLIP.",
    )
    parser.add_argument(
        "frames_dir",
        nargs="?",
        default="frames",
        help="Directory containing extracted frame images (default: ./frames).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-frame top-3 scores (always on stderr).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip loading/saving the frame cache.",
    )
    add_gpu_args(parser)
    args = parser.parse_args()

    try:
        if args.no_cache:
            CACHE_FILE.unlink(missing_ok=True)
        describe(args.frames_dir, verbose=args.verbose, device_cfg=resolve_device(args.use_gpu))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())