"""
Generate a 10-second background music track from a JSON object containing
'Genre', 'Tempo', and 'Mood' (as produced by suggest_song_attributes.py).

Uses Meta's MusicGen model (facebook/musicgen-small) loaded via Hugging Face
Transformers — the same model published by Meta's Audiocraft team, just
loaded through a more portable interface that avoids Audiocraft's heavy and
version-pinned dependencies.

Usage:
    python scripts/generate_music.py --attributes-file attrs.json
    python scripts/generate_music.py --attributes '{"Genre":"Lo-fi","Tempo":"Slow","Mood":"Calm"}'
    cat attrs.json | python scripts/generate_music.py
"""

import argparse
import json
import sys
from typing import Any, Dict

import numpy as np
import torch
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, MusicgenForConditionalGeneration


DEFAULT_MODEL_ID = "facebook/musicgen-small"

# MusicGen uses an audio frame rate of 50 Hz, so ~50 tokens == 1 second.
TOKENS_PER_SECOND = 50


def build_prompt(attrs: Dict[str, Any]) -> str:
    genre = str(attrs.get("Genre", "")).strip()
    tempo = str(attrs.get("Tempo", "")).strip()
    mood = str(attrs.get("Mood", "")).strip()

    parts = []
    if genre:
        parts.append(f"{genre} background music")
    else:
        parts.append("background music")
    if mood:
        parts.append(f"with a {mood.lower()} mood")
    if tempo:
        parts.append(f"at a {tempo.lower()} tempo")
    parts.append("instrumental, suitable as a soundtrack for a short video")
    return ", ".join(parts)


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


def generate_music(prompt: str, duration_seconds: int, output_path: str, model_id: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MusicGen '{model_id}' on {device}... (first run downloads ~1.5 GB)", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    print(f"Prompt: {prompt}", file=sys.stderr)
    print(f"Generating {duration_seconds} seconds of audio... (CPU runs are slow — this may take several minutes)", file=sys.stderr)

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    max_new_tokens = duration_seconds * TOKENS_PER_SECOND

    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_np = audio_values[0, 0].cpu().numpy()

    # Normalize to int16 PCM and write WAV.
    peak = float(np.max(np.abs(audio_np))) or 1.0
    audio_int16 = np.int16(audio_np / peak * 32767)
    write_wav(output_path, rate=sampling_rate, data=audio_int16)
    print(f"Saved {output_path} ({sampling_rate} Hz, {len(audio_int16) / sampling_rate:.2f} s).", file=sys.stderr)


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
    args = parser.parse_args()

    try:
        attrs = load_attributes(args)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error reading attributes: {exc}", file=sys.stderr)
        return 1

    prompt = build_prompt(attrs)
    generate_music(prompt, args.duration, args.output, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
