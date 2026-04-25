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
Extracts one frame every N seconds from an MP4 video using OpenCV.

```bash
python scripts/extract_frames.py input.mp4 [output_dir] [--interval 2.0]
```

### `describe_video.py`
Analyzes extracted frames using [CLIP](https://openai.com/clip/) (openai/clip-vit-base-patch32) to describe scenes, actions, and moods across the video.

```bash
python scripts/describe_video.py [frames_dir]
```

### `suggest_song_attributes.py`
Uses a local LLM to suggest music attributes (Genre, Tempo, Mood) from the video description. Supports an interactive lo-fi steering option.

```bash
python scripts/suggest_song_attributes.py --description "A sunny beach with people laughing..."
python scripts/suggest_song_attributes.py --description-file desc.txt
```

Flags:
- `--no-interactive` — skip the lo-fi follow-up question (useful when piping output)
- `--lofi` — skip straight to a lo-fi-styled suggestion
- `--model <id>` — override the HF model id

Default model: `Qwen/Qwen2.5-1.5B-Instruct` (runs on CPU).

**Interactive flow:** After the first suggestion, the script asks `Is this too energetic? Should I try a more 'lo-fi' version? [y/N]:` via `/dev/tty` (works even when stdin is piped). If the user responds `y`, it regenerates with lo-fi steering and prints the lo-fi version alongside the original. The final JSON is always written to stdout for piping into `generate_music.py`.

### `generate_music.py`
Generates background music using [Meta's MusicGen](https://ai.meta.com/research/musicgen/) (facebook/musicgen-small) based on Genre/Tempo/Mood attributes.

```bash
python scripts/generate_music.py --attributes '{"Genre":"Lo-fi","Tempo":"Slow","Mood":"Calm"}'
python scripts/generate_music.py --attributes-file attrs.json --output background_music.wav --duration 10
```

### `analyze_music.py`
Scans a folder of audio files and builds a JSON mood index (`library.json`). The index maps each track to its inferred Genre / Tempo / Mood so `run_me.py` can pick a match instead of generating.

```bash
python scripts/analyze_music.py /path/to/music/folder [--index-file library.json] [--recursive]
```

Requires **librosa** (install with `uv add librosa`).

## Dependencies

- **opencv-python-headless** - Video frame extraction
- **torch**, **transformers** - Model loading (CLIP, MusicGen, LLM)
- **pillow** - Image handling
- **scipy** - WAV file writing
- **librosa** - Audio feature analysis in `analyze_music.py`

Models are downloaded on first use (~600 MB for CLIP, ~1.5 GB for MusicGen, several GB for LLM).

## End-to-End Usage

```bash
# Build a mood index of your favourite tracks (one-time)
python scripts/analyze_music.py /path/to/music/folder --index-file library.json

# Run the full pipeline — picks the best library match before generating
uv run python run_me.py --video input.mp4 --library library.json

# Force generation (skip the library)
uv run python run_me.py --video input.mp4 --no-library
```