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
"""

import argparse
import json
import re
import sys
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_llm(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LLM '{model_id}' on {device}... (first run downloads several GB)", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def build_prompt(tokenizer, description: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Video description:\n{description.strip()}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(tokenizer, model, device, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_json(raw: str) -> Dict[str, Any]:
    # Strip common markdown code fences.
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Try direct parse first.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the first {...} block.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in model output:\n{raw}")
    return json.loads(match.group(0))


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
    args = parser.parse_args()

    try:
        description = read_description(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    tokenizer, model, device = load_llm(args.model)
    prompt = build_prompt(tokenizer, description)
    raw_output = generate(tokenizer, model, device, prompt)

    print("\n=== Raw model output ===", file=sys.stderr)
    print(raw_output, file=sys.stderr)

    try:
        result = extract_json(raw_output)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"\nError: failed to parse JSON from model output: {exc}", file=sys.stderr)
        return 1

    print("\n=== Suggested song attributes ===", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
