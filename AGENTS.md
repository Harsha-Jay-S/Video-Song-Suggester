# AGENTS.md — General Session Startup

## When I start a new project session

### Step 1: Scan before asking
List the root directory. Read the README if it exists. Look for the main entry point. Understand what the project does before answering anything.

### Step 2: Compile check
After any file change, run `python3 -m py_compile <file>` (or the language equivalent) before running anything. Fix all syntax errors first.

### Step 3: Test with real files
When given a test path, run the actual pipeline/command. Don't assume it works. Show real output.

### Step 4: Be concise
Direct answers. 1-3 sentences for simple questions. Show real command output when running tests. No intro/outro paragraphs.

---

## Project-specific guidance

### Project Overview
Video-Frame-Extractor: a Python pipeline that extracts frames from videos, analyzes them with CLIP, suggests music attributes via LLM, and generates music with MusicGen. Scripts live in `scripts/` directory.

### Entry Points
- **Full pipeline**: `run_me.py` — orchestrates the entire pipeline
- **Individual scripts**: `scripts/extract_frames.py`, `scripts/describe_video.py`, `scripts/suggest_song_attributes.py`, `scripts/generate_music.py`, `scripts/analyze_music.py`
- **Shared utilities**: `scripts/_device.py` — GPU/CPU device management shared across scripts

### GPU Toggle
All scripts support `--gpu` and `--no-gpu` flags. Default: auto-detect CUDA availability. GPU is recommended for CLIP analysis and MusicGen.

### Key New Features (v2)
- Batch processing in `describe_video.py` (CLIP frames in batches of 8)
- Frame caching in `~/.cache/video-frame-extractor/frame_cache.json`
- Scene-change detection in `extract_frames.py`
- Duplicate frame dedup via perceptual hash in `extract_frames.py`
- Rich prompt engineering with `INSTRUMENT_HINTS` and `MOOD_ADJECTIVES` in `generate_music.py`
- Seed control (`--seed`) and variations (`--variations`) in `generate_music.py`
- Parallel processing in `analyze_music.py`
- Dry-run mode in `run_me.py`

### Testing
```bash
python3 -m py_compile scripts/_device.py
python3 -m py_compile scripts/extract_frames.py
python3 -m py_compile scripts/describe_video.py
python3 -m py_compile scripts/suggest_song_attributes.py
python3 -m py_compile scripts/generate_music.py
python3 -m py_compile scripts/analyze_music.py
python3 -m py_compile run_me.py
```

### Architecture Notes
- `_device.py` exports `DeviceConfig`, `resolve_device()`, `add_gpu_args()` — imported by all model scripts
- `describe_video.py` imports `_device`; `run_me.py` imports `_device` indirectly
- All scripts return int exit codes (0=success, 1=error)
- Library match uses keyword overlap scoring; v2 index has `version: 2` with `bpm`, `energy`, `key_mode` per track
- `generate_music.py` uses MusicPrompt dataclass for structured prompts
- tqdm used for progress bars; check `sys.stderr.isatty()` for disabled output in batch contexts

### Quirks
- `analyze_music.py` requires librosa (optional, install with `uv add librosa`)
- `generate_music.py` uses `do_sample=True` even on CPU (non-deterministic by default, use `--seed` for reproducibility)
- `describe_video.py` batch processing loads all images in a batch into memory; batch_size=8 is conservative for CPU
- Frame cache is keyed by MD5 hash of raw bytes — any resize/recompress invalidates it