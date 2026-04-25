"""
Generate a 10-second background music track from a JSON object containing
'Genre', 'Tempo', and 'Mood' (as produced by suggest_song_attributes.py).

Uses Meta's MusicGen model (facebook/musicgen-small) loaded via Hugging Face
Transformers.

Usage:
    python scripts/generate_music.py --attributes-file attrs.json
    python scripts/generate_music.py --attributes '{"Genre":"Lo-fi","Tempo":"Slow","Mood":"Calm"}'
    python scripts/generate_music.py --attributes '{"Genre":"Lo-fi"}' --no-gpu  # force CPU
    python scripts/generate_music.py --attributes '{"Genre":"Lo-fi"}' --gpu      # force GPU
    python scripts/generate_music.py --attributes '{"Genre":"Lo-fi"}' --seed 42 --variations 3
    python scripts/generate_music.py --dry-run  # preview prompt without generating
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from _device import add_gpu_args, resolve_device, DeviceConfig


DEFAULT_MODEL_ID = "facebook/musicgen-small"
TOKENS_PER_SECOND = 50


INSTRUMENT_HINTS = {
    "Lo-fi": "with vinyl crackle and warm analog texture",
    "Cinematic": "with lush orchestral strings and subtle brass",
    "Electronic": "with clean synthesizers and subtle electronic textures",
    "Indie": "with acoustic guitar, light percussion, and warm vocals",
    "Jazz": "with upright bass, gentle cymbals, and piano",
    "Classical": "with solo piano and gentle dynamics",
    "Ambient": "with reverb-drenched pads and subtle textures",
}

MOOD_ADJECTIVES = {
    "Calm": ["peaceful", "serene", "relaxing"],
    "Energetic": ["driving", "upbeat", "vibrant"],
    "Melancholic": ["haunting", "bittersweet", "wistful"],
    "Uplifting": ["bright", "optimistic", "hopeful"],
    "Dramatic": ["intense", "sweeping", "cinematic"],
    "Nostalgic": ["retro", "vintage", "familiar"],
}


@dataclass
class MusicPrompt:
    full_prompt: str
    genre: str
    tempo: str
    mood: str
    instrumentation: str

    def __str__(self) -> str:
        return self.full_prompt


def build_prompt(
    attrs: Dict[str, Any],
    instrumentation_hint: Optional[str] = None,
    mood_adjectives: Optional[list] = None,
) -> MusicPrompt:
    genre = str(attrs.get("Genre", "")).strip()
    tempo = str(attrs.get("Tempo", "")).strip()
    mood = str(attrs.get("Mood", "")).strip()

    parts = []
    if genre:
        parts.append(f"{genre} background music")
    else:
        parts.append("background music")

    if mood_adjectives and mood_adjectives:
        parts.append(f"with a {mood_adjectives[0]} and {mood_adjectives[1]} atmosphere")

    if tempo:
        parts.append(f"at a {tempo.lower()} tempo")

    if instrumentation_hint:
        parts.append(instrumentation_hint)

    if mood:
        parts.append(f"conveying a {mood.lower()} feeling")

    parts.append("instrumental, suitable as a soundtrack for a short video")

    full_prompt = ", ".join(parts)

    instr = INSTRUMENT_HINTS.get(genre, "")
    if instrumentation_hint:
        instr = instrumentation_hint

    return MusicPrompt(
        full_prompt=full_prompt,
        genre=genre,
        tempo=tempo,
        mood=mood,
        instrumentation=instr,
    )


def load_attributes(args: argparse.Namespace) -> Dict[str, Any]:
    if args.attributes:
        return json.loads(args.attributes)
    if args.attributes_file:
        with open(args.attributes_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if not sys.stdin.isatty():
        return json.loads(sys.stdin.read())
    raise ValueError(
        "No attributes provided. Use --attributes, --attributes-file, or pipe JSON via stdin."
    )


def select_variant(
    attrs: Dict[str, Any],
    variation_idx: int,
) -> tuple[Optional[str], Optional[list]]:
    genre = str(attrs.get("Genre", "")).strip()
    mood = str(attrs.get("Mood", "")).strip()

    if genre in INSTRUMENT_HINTS and variation_idx == 0:
        return INSTRUMENT_HINTS[genre], None
    if genre in INSTRUMENT_HINTS and variation_idx == 1:
        return None, MOOD_ADJECTIVES.get(mood, ["warm", "ambient"])[:2]
    if genre in INSTRUMENT_HINTS and variation_idx == 2:
        return f"with lo-fi {genre.lower()} vibes and warm analog warmth", None

    return None, MOOD_ADJECTIVES.get(mood, ["warm", "gentle"])[:2]


def generate_music(
    prompt: MusicPrompt,
    duration_seconds: int,
    output_path: str,
    model_id: str,
    device_cfg: DeviceConfig,
    seed: Optional[int] = None,
    num_variations: int = 1,
) -> list[str]:
    print(f"Loading MusicGen '{model_id}' on {device_cfg.device}... (first run downloads ~1.5 GB)", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device_cfg.device)
    if device_cfg.use_gpu:
        model = model.half()
    model.eval()

    print(f"Prompt: {prompt.full_prompt}", file=sys.stderr)
    print(f"Generating {duration_seconds} seconds of audio... ({'GPU' if device_cfg.use_gpu else 'CPU'})", file=sys.stderr)

    inputs = processor(text=[prompt.full_prompt], padding=True, return_tensors="pt").to(device_cfg.device)
    if device_cfg.use_gpu:
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    max_new_tokens = duration_seconds * TOKENS_PER_SECOND

    output_paths = []
    for i in range(num_variations):
        if num_variations > 1:
            print(f"  Variation {i+1}/{num_variations}...", file=sys.stderr, flush=True)

        if seed is not None:
            gen_seed = seed + i
            torch.manual_seed(gen_seed)
            np.random.seed(gen_seed)

        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_np = audio_values[0, 0].cpu().numpy()

        peak = float(np.max(np.abs(audio_np))) or 1.0
        audio_int16 = np.int16(audio_np / peak * 32767)

        if num_variations > 1:
            suffix = f"_{i+1}"
            p = Path(output_path)
            out_path = str(p.stem + suffix + p.suffix)
        else:
            out_path = output_path

        write_wav(out_path, rate=sampling_rate, data=audio_int16)
        output_paths.append(out_path)
        print(f"  Saved {out_path} ({sampling_rate} Hz, {len(audio_int16) / sampling_rate:.2f} s).", file=sys.stderr)

    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate background music from Genre/Tempo/Mood JSON using Meta's MusicGen."
    )
    parser.add_argument("--attributes", help="Inline JSON string with Genre/Tempo/Mood.")
    parser.add_argument("--attributes-file", help="Path to a JSON file with Genre/Tempo/Mood.")
    parser.add_argument("--output", default="background_music.wav", help="Output WAV file path.")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds (default: 10).")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face MusicGen model id (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=1,
        help="Number of variations to generate (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt without generating audio.",
    )
    add_gpu_args(parser)
    args = parser.parse_args()

    try:
        attrs = load_attributes(args)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error reading attributes: {exc}", file=sys.stderr)
        return 1

    genre = str(attrs.get("Genre", "")).strip()
    instrumentation = INSTRUMENT_HINTS.get(genre, None)
    mood_adj = MOOD_ADJECTIVES.get(str(attrs.get("Mood", "")).strip(), None)
    music_prompt = build_prompt(attrs, instrumentation_hint=instrumentation, mood_adjectives=mood_adj)

    print(f"\n=== Music Generation Prompt ===", file=sys.stderr)
    print(f"Full prompt: {music_prompt.full_prompt}", file=sys.stderr)
    print(f"Genre: {music_prompt.genre}", file=sys.stderr)
    print(f"Tempo: {music_prompt.tempo}", file=sys.stderr)
    print(f"Mood: {music_prompt.mood}", file=sys.stderr)
    print(f"Instrumentation: {music_prompt.instrumentation}", file=sys.stderr)

    if args.dry_run:
        print("\n[Dry run] Skipping actual generation.")
        return 0

    device_cfg = resolve_device(args.use_gpu)

    if args.variations > 1:
        all_paths = []
        for i in range(args.variations):
            instr, mood_adj_i = select_variant(attrs, i)
            music_prompt_i = build_prompt(attrs, instrumentation_hint=instr, mood_adjectives=mood_adj_i)
            print(f"\nVariation {i+1} prompt: {music_prompt_i.full_prompt}", file=sys.stderr)
            paths = generate_music(
                music_prompt_i,
                args.duration,
                args.output,
                args.model,
                device_cfg,
                seed=args.seed,
                num_variations=1,
            )
            all_paths.extend(paths)
        print(f"\n=== Done! Generated {len(all_paths)} variations ===", file=sys.stderr)
    else:
        paths = generate_music(
            music_prompt,
            args.duration,
            args.output,
            args.model,
            device_cfg,
            seed=args.seed,
            num_variations=1,
        )
        print(f"\n=== Done! Music saved to {paths[0]} ===", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())