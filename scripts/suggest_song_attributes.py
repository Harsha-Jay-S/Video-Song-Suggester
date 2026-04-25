"""
Take a plain-text video description and ask a local LLM (Hugging Face
Transformers) to output a JSON object with 'Genre', 'Tempo', and 'Mood'.

By default uses Qwen/Qwen2.5-1.5B-Instruct — a small, open-source,
instruction-tuned model that runs on CPU. You can pass --model to use any
other HF instruction-tuned model (e.g. meta-llama/Llama-3.2-1B-Instruct,
which requires a Hugging Face access token because it is gated).

Usage:
    python scripts/suggest_song_attributes.py --description "A sunny beach with people laughing..."
    python scripts/suggest_song_attributes.py --description-file desc.txt
    echo "A rainy city at night..." | python scripts/suggest_song_attributes.py
    python scripts/suggest_song_attributes.py --description "..." --no-gpu  # force CPU
    python scripts/suggest_song_attributes.py --description "..." --gpu      # force GPU
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _device import add_gpu_args, resolve_device, DeviceConfig


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


SYSTEM_PROMPT = (
    "You are a music supervisor who suggests a song style for a video. "
    "Based on the video description the user provides, return ONLY a single "
    "JSON object with exactly these keys:\n"
    '  - "Genre": a music genre as a string (e.g. "Lo-fi hip hop", "Indie folk", '
    '"Electronic dance", "Cinematic orchestral").\n'
    '  - "Tempo": a tempo description as a string (e.g. "Slow", "Medium", "Fast", '
    'or a BPM range like "90-110 BPM").\n'
    '  - "Mood": one or two descriptive mood words (e.g. "Uplifting", '
    '"Melancholic", "Energetic and playful").\n'
    "Do not include any explanation, prose, or extra keys. Output JSON only."
)


LOFI_STEERING = (
    "\n\nThe previous suggestion felt too energetic for the user. "
    "Lean into a lo-fi direction: choose a chill, mellow, lo-fi-style genre, "
    "use a slower tempo, and pick a calm, relaxed mood. "
    "Still return ONLY the JSON object with the same three keys."
)


@dataclass
class SongAttributes:
    Genre: str
    Tempo: str
    Mood: str

    def to_dict(self) -> Dict[str, Any]:
        return {"Genre": self.Genre, "Tempo": self.Tempo, "Mood": self.Mood}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SongAttributes":
        return cls(
            Genre=str(d.get("Genre", "")),
            Tempo=str(d.get("Tempo", "")),
            Mood=str(d.get("Mood", "")),
        )


def load_llm(model_id: str, device_cfg: DeviceConfig):
    print(f"Loading LLM '{model_id}' on {device_cfg.device}... (first run downloads several GB)", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=device_cfg.dtype).to(device_cfg.device)
    if device_cfg.use_gpu:
        model = model.half()
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, description: str, lofi: bool = False) -> str:
    system_content = SYSTEM_PROMPT + (LOFI_STEERING if lofi else "")
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Video description:\n{description.strip()}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def suggest_attributes(
    tokenizer, model, description: str, device_cfg: DeviceConfig, lofi: bool = False
) -> SongAttributes:
    prompt = build_prompt(tokenizer, description, lofi=lofi)
    raw_output = generate(tokenizer, model, prompt, device_cfg)
    print("\n=== Raw model output ===", file=sys.stderr)
    print(raw_output, file=sys.stderr)
    return extract_json(raw_output)


def ask_lofi_followup() -> bool:
    """Ask the user the lo-fi follow-up question. Reads from /dev/tty so it
    works even when stdin is being piped (e.g. description piped in from
    describe_video.py). Returns False if no terminal is available."""
    prompt = "\nIs this too energetic? Should I try a more 'lo-fi' version? [y/N]: "
    try:
        with open("/dev/tty", "r+") as tty:
            tty.write(prompt)
            tty.flush()
            answer = tty.readline().strip().lower()
    except OSError:
        return False
    return answer in {"y", "yes"}


def generate(tokenizer, model, prompt: str, device_cfg: DeviceConfig, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device_cfg.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_json(raw: str) -> SongAttributes:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return SongAttributes.from_dict(json.loads(cleaned))
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in model output:\n{raw}")
    return SongAttributes.from_dict(json.loads(match.group(0)))


def read_description(args: argparse.Namespace) -> str:
    if args.description:
        return args.description
    if args.description_file:
        with open(args.description_file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError(
        "No description provided. Use --description, --description-file, or pipe text via stdin."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use a local LLM to turn a video description into a Genre/Tempo/Mood JSON object."
    )
    parser.add_argument("--description", help="Video description text.")
    parser.add_argument("--description-file", help="Path to a text file containing the description.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the lo-fi follow-up question (useful when piping output to another script).",
    )
    parser.add_argument(
        "--lofi",
        action="store_true",
        help="Skip straight to a lo-fi-styled suggestion (no follow-up question).",
    )
    add_gpu_args(parser)
    args = parser.parse_args()

    try:
        description = read_description(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    device_cfg = resolve_device(args.use_gpu)
    tokenizer, model = load_llm(args.model, device_cfg)

    try:
        result = suggest_attributes(tokenizer, model, description, device_cfg, lofi=args.lofi)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"\nError: failed to parse JSON from model output: {exc}", file=sys.stderr)
        return 1

    print("\n=== Suggested song attributes ===", file=sys.stderr)
    print(json.dumps(result.to_dict(), indent=2), file=sys.stderr)

    if not args.lofi and not args.no_interactive and ask_lofi_followup():
        print("\nRegenerating with a lo-fi steering hint...", file=sys.stderr)
        try:
            result = suggest_attributes(tokenizer, model, description, device_cfg, lofi=True)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"\nError: failed to parse JSON from lo-fi output: {exc}", file=sys.stderr)
            return 1
        print("\n=== Lo-fi song attributes ===", file=sys.stderr)
        print(json.dumps(result.to_dict(), indent=2), file=sys.stderr)

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())