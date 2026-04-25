# Video-Frame-Extractor

A Python pipeline that extracts frames from videos, analyzes their content using vision and language models, and generates matching background music.

## Pipeline Overview

```
video.mp4
    │
    ▼
┌──────────────┐      ┌──────────────┐
│extract_frames│ ───▶  │   frames/   │
└──────────────┘        └──────────────┘
                               │
                               ▼
                       ┌──────────────┐
                       │describe_video│
                       └──────────────┘
                               │
                               ▼
                       ┌──────────────────┐
                       │ Video Description │
                       └──────────────────┘
                               │
                               ▼
                   ┌──────────────────────┐
                   │suggest_song_attributes│
                   └──────────────────────┘
                               │
                               ▼
                       ┌──────────────────┐
                       │ Genre/Tempo/Mood  │
                       └──────────────────┘
                               │
                 ┌─────────────┴───────────┐
                 ▼                           ▼
         ┌──────────────────┐    ┌──────────────────────┐
         │  library match?   │───▶│  best track → copy   │
         └──────────────────┘    └──────────────────────┘
                 │ no close match
                 ▼
         ┌──────────────────────┐
         │    generate_music     │
         └──────────────────────┘
                               │
                               ▼
                       ┌──────────────────┐
                       │background_music.wav│
                       └──────────────────┘
```

## Scripts

### `extract_frames.py`
Extracts frames from a video using smart sampling strategies.

```bash
python scripts/extract_frames.py input.mp4 [output_dir] [--interval 2.0]
python scripts/extract_frames.py input.mp4 --scene-change  # detect scene changes
python scripts/extract_frames.py input.mp4 --dedup        # skip duplicate frames
python scripts/extract_frames.py input.mp4 --max-frames 30  # limit max frames
python scripts/extract_frames.py input.mp4 --no-gpu         # force CPU
```

Features:
- Uniform interval extraction (default)
- Scene-change detection (`--scene-change`)
- Duplicate/similar frame removal (`--dedup`, enabled by default)
- Max frame limit (`--max-frames`)
- GPU/CPU toggle (`--gpu` / `--no-gpu`)

### `describe_video.py`
Analyzes extracted frames using [CLIP](https://openai.com/clip/) (openai/clip-vit-base-patch32) to describe scenes, actions, and moods across the video.

```bash
python scripts/describe_video.py [frames_dir]
python scripts/describe_video.py [frames_dir] --no-gpu  # force CPU
python scripts/describe_video.py [frames_dir] --no-cache  # skip cache
```

Features:
- Batch processing for faster frame analysis
- Frame cache for skipping unchanged frames
- Complexity scoring per frame
- GPU/CPU toggle (`--gpu` / `--no-gpu`)

### `suggest_song_attributes.py`
Uses a local LLM to suggest music attributes (Genre, Tempo, Mood) from the video description. Supports an interactive lo-fi steering option.

```bash
python scripts/suggest_song_attributes.py --description "A sunny beach with people laughing..."
python scripts/suggest_song_attributes.py --description-file desc.txt
python scripts/suggest_song_attributes.py --description "..." --no-gpu  # force CPU
```

Flags:
- `--no-interactive` — skip the lo-fi follow-up question (useful when piping output)
- `--lofi` — skip straight to a lo-fi-styled suggestion
- `--model <id>` — override the HF model id
- `--gpu` / `--no-gpu` — GPU/CPU toggle

Default model: `Qwen/Qwen2.5-1.5B-Instruct` (runs on CPU).

**Interactive flow:** After the first suggestion, the script asks `Is this too energetic? Should I try a more 'lo-fi' version? [y/N]:` via `/dev/tty` (works even when stdin is piped). If the user responds `y`, it regenerates with lo-fi steering and prints the lo-fi version alongside the original. The final JSON is always written to stdout for piping into `generate_music.py`.

### `generate_music.py`
Generates background music using [Meta's MusicGen](https://ai.meta.com/research/musicgen/) (facebook/musicgen-small) based on Genre/Tempo/Mood attributes.

```bash
python scripts/generate_music.py --attributes '{"Genre":"Lo-fi","Tempo":"Slow","Mood":"Calm"}'
python scripts/generate_music.py --attributes-file attrs.json --output background_music.wav --duration 10
python scripts/generate_music.py --attributes '{"Genre":"Lo-fi"}' --seed 42 --variations 3
python scripts/generate_music.py --dry-run  # preview prompt without generating
```

Flags:
- `--seed <n>` — random seed for reproducible generation
- `--variations <n>` — number of music variations to generate
- `--dry-run` — preview the prompt without generating audio
- `--gpu` / `--no-gpu` — GPU/CPU toggle

Features:
- Rich prompt engineering with instrumentation hints per genre
- Mood adjectives based on the video mood
- Multiple variation generation with different prompt variants
- Reproducible generation with seed control

### `analyze_music.py`
Scans a folder of audio files and builds a JSON mood index (`library.json`). The index maps each track to its inferred Genre / Tempo / Mood so `run_me.py` can pick a match instead of generating.

```bash
python scripts/analyze_music.py /path/to/music/folder [--index-file library.json] [--recursive]
python scripts/analyze_music.py /path/to/music/folder --parallel 4  # parallel processing
python scripts/analyze_music.py /path/to/music/folder --dry-run   # preview without writing
```

Requires **librosa** (install with `uv add librosa`).

Features:
- Parallel processing with `--parallel <n>`
- Dry-run mode to preview results
- Detailed track analysis (BPM, energy, key mode)

## Dependencies

- **opencv-python-headless** - Video frame extraction
- **torch**, **transformers** - Model loading (CLIP, MusicGen, LLM)
- **pillow** - Image handling
- **scipy** - WAV file writing
- **tqdm** - Progress bars
- **librosa** - Audio feature analysis in `analyze_music.py`

Models are downloaded on first use (~600 MB for CLIP, ~1.5 GB for MusicGen, several GB for LLM).

## GPU Support

All scripts that use PyTorch models support GPU acceleration. Use `--gpu` to explicitly enable GPU or `--no-gpu` to force CPU:

```bash
python run_me.py --video input.mp4 --gpu        # enable GPU if available
python run_me.py --video input.mp4 --no-gpu   # force CPU
```

By default, scripts auto-detect CUDA availability. GPU is recommended for faster frame analysis and music generation.

## End-to-End Usage

```bash
# Build a mood index of your favourite tracks (one-time)
python scripts/analyze_music.py /path/to/music/folder --index-file library.json

# Run the full pipeline with all v2 features
uv run python run_me.py --video input.mp4 --library library.json --scene-change --max-frames 30

# Force GPU, generate 3 variations
uv run python run_me.py --video input.mp4 --gpu --variations 3

# Force generation (skip the library)
uv run python run_me.py --video input.mp4 --no-library

# Dry run to preview the pipeline
uv run python run_me.py --video input.mp4 --dry-run
```

## Version 2 Features

- **GPU toggle**: Use `--gpu` / `--no-gpu` on any script or `run_me.py`
- **Smart frame extraction**: Scene-change detection, duplicate removal, max frame limits
- **Batch processing**: CLIP frame analysis in batches for faster processing
- **Frame caching**: Skip re-analyzing unchanged frames
- **Rich music prompts**: Genre-specific instrumentation hints and mood adjectives
- **Reproducible generation**: `--seed` for deterministic output
- **Multiple variations**: `--variations` to generate and compare multiple tracks
- **Dry-run mode**: Preview pipeline steps without generating audio
- **Parallel processing**: `--parallel` in `analyze_music.py`
- **Typed dataclasses**: Structured data objects throughout the pipeline
- **Progress bars**: Visual feedback for long-running operations

This Project is Entirely done by Opencode AI. I am not responsible for anything faulty. PR's are encouraged 
