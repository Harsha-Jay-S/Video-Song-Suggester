"""
Describe what's happening in a video using CLIP (openai/clip-vit-base-patch32).

Workflow:
  1. Reads JPG frames from a directory (produced by extract_frames.py).
  2. Scores each frame against a curated set of candidate phrases covering
     scenes, actions, and moods (CLIP zero-shot classification).
  3. Aggregates the top phrases across all frames and emits a plain-text
     description of what is happening in the video.

Usage:
    python scripts/describe_video.py [frames_dir]

If frames_dir is not provided, ./frames is used.
"""

import argparse
import os
import sys
from collections import Counter
from typing import List, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"


SCENE_PHRASES: List[str] = [
    "a city street at night",
    "a city street during the day",
    "a quiet forest",
    "a sandy beach with ocean waves",
    "snow-covered mountains",
    "a cozy indoor room",
    "a busy office",
    "a kitchen with food being prepared",
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
    "a happy and upbeat moment",
    "a calm and peaceful moment",
    "a sad and emotional moment",
    "a tense and dramatic moment",
    "a romantic moment",
    "an energetic and exciting moment",
    "a mysterious or eerie moment",
    "a nostalgic moment",
    "a playful and fun moment",
    "a serious and focused moment",
]


def load_model() -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP model '{MODEL_ID}' on {device}... (first run downloads ~600 MB)")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    return model, processor, device


def score_frame(
    image: Image.Image,
    phrases: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> List[Tuple[str, float]]:
    inputs = processor(text=phrases, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()
    return list(zip(phrases, probs))


def top_n(scored: List[Tuple[str, float]], n: int = 1) -> List[Tuple[str, float]]:
    return sorted(scored, key=lambda x: x[1], reverse=True)[:n]


def summarize(
    scenes: Counter, actions: Counter, moods: Counter, num_frames: int
) -> str:
    def fmt(counter: Counter, top: int = 2) -> List[str]:
        return [phrase for phrase, _ in counter.most_common(top)]

    top_scenes = fmt(scenes, 2)
    top_actions = fmt(actions, 2)
    top_moods = fmt(moods, 1)

    parts = [f"Across {num_frames} sampled frames, the video appears to show "]
    if top_scenes:
        parts.append(" and ".join(top_scenes))
        parts.append(". ")
    if top_actions:
        parts.append("It mostly features " + " and ".join(top_actions) + ". ")
    if top_moods:
        parts.append("The overall mood feels like " + top_moods[0] + ".")
    return "".join(parts)


def describe(frames_dir: str) -> None:
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_paths = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_paths:
        raise RuntimeError(f"No image frames found in {frames_dir}")

    model, processor, device = load_model()

    scene_counter: Counter = Counter()
    action_counter: Counter = Counter()
    mood_counter: Counter = Counter()

    print(f"\nAnalyzing {len(frame_paths)} frames...\n")
    for idx, path in enumerate(frame_paths, start=1):
        image = Image.open(path).convert("RGB")

        top_scene = top_n(score_frame(image, SCENE_PHRASES, model, processor, device))[0]
        top_action = top_n(score_frame(image, ACTION_PHRASES, model, processor, device))[0]
        top_mood = top_n(score_frame(image, MOOD_PHRASES, model, processor, device))[0]

        scene_counter[top_scene[0]] += 1
        action_counter[top_action[0]] += 1
        mood_counter[top_mood[0]] += 1

        print(
            f"  Frame {idx:>3}/{len(frame_paths)}: "
            f"scene='{top_scene[0]}' ({top_scene[1]:.2f}) | "
            f"action='{top_action[0]}' ({top_action[1]:.2f}) | "
            f"mood='{top_mood[0]}' ({top_mood[1]:.2f})"
        )

    summary = summarize(scene_counter, action_counter, mood_counter, len(frame_paths))
    print("\n=== Plain-text description ===")
    print(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe a video from extracted frames using CLIP.")
    parser.add_argument(
        "frames_dir",
        nargs="?",
        default="frames",
        help="Directory containing extracted frame images (default: ./frames).",
    )
    args = parser.parse_args()

    try:
        describe(args.frames_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
