"""
One-shot pipeline: extract frames → describe video → suggest music attributes →
pick the best match from a library OR generate new music.

Usage:
    python run_me.py --video input.mp4 [output_dir]
    python run_me.py --video input.mp4 --interval 1.0
    python run_me.py --video input.mp4 --library /path/to/library.json
    python run_me.py --video input.mp4 --no-library   # always generate
    python run_me.py --video input.mp4 --gpu          # force GPU
    python run_me.py --video input.mp4 --no-gpu       # force CPU
    python run_me.py --video input.mp4 --scene-change  # detect scene changes
    python run_me.py --video input.mp4 --max-frames 30 # limit frames
    python run_me.py --video input.mp4 --seed 42       # reproducible music
    python run_me.py --video input.mp4 --variations 3  # generate 3 variations
    python run_me.py --video input.mp4 --dry-run       # preview without generating
    python run_me.py --video input.mp4 --config pipeline.yaml  # use config file
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent / "scripts"
FRAMES_DIR_DEFAULT = "frames"
MUSIC_OUT_DEFAULT = "background_music.wav"
LIBRARY_DEFAULT = "library.json"
CONFIG_DEFAULT = "pipeline.yaml"


@dataclass
class PipelineConfig:
    interval: float = 2.0
    duration: int = 10
    max_frames: Optional[int] = None
    dedup: bool = True
    scene_change: bool = False
    use_gpu: Optional[bool] = None
    no_library: bool = False
    library_path: Path = Path(LIBRARY_DEFAULT)
    output_path: Path = Path(MUSIC_OUT_DEFAULT)
    seed: Optional[int] = None
    variations: int = 1
    dry_run: bool = False
    lofi: bool = False
    no_interactive: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        return cls(
            interval=args.interval,
            duration=args.duration,
            max_frames=args.max_frames,
            dedup=args.dedup,
            scene_change=args.scene_change,
            use_gpu=args.use_gpu,
            no_library=args.no_library,
            library_path=Path(args.library),
            output_path=Path(args.output),
            seed=args.seed,
            variations=args.variations,
            dry_run=args.dry_run,
            lofi=args.lofi,
            no_interactive=args.no_interactive,
        )


@dataclass
class VideoDescription:
    description: str
    scene_counter: Dict[str, int] = field(default_factory=dict)
    action_counter: Dict[str, int] = field(default_factory=dict)
    mood_counter: Dict[str, int] = field(default_factory=dict)
    frame_count: int = 0


@dataclass
class MusicAttributes:
    Genre: str
    Tempo: str
    Mood: str

    def to_dict(self) -> Dict[str, Any]:
        return {"Genre": self.Genre, "Tempo": self.Tempo, "Mood": self.Mood}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MusicAttributes":
        return cls(
            Genre=str(d.get("Genre", "")),
            Tempo=str(d.get("Tempo", "")),
            Mood=str(d.get("Mood", "")),
        )


def run(script_name: str, *args, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    cfg_args = []
    if "--no-gpu" in args or "--gpu" in args:
        pass
    cmd = [sys.executable, str(SCRIPT_DIR / script_name), *args]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)


def gpu_arg(use_gpu: Optional[bool]) -> List[str]:
    if use_gpu is True:
        return ["--gpu"]
    if use_gpu is False:
        return ["--no-gpu"]
    return []


def score(attr: Dict[str, Any], track: Dict[str, Any]) -> float:
    score_val = 0.0
    if attr.get("Genre", "").lower() in track.get("Genre", "").lower():
        score_val += 2.0
    if attr.get("Tempo") == track.get("Tempo"):
        score_val += 1.0
    mood_attr = attr.get("Mood", "").lower()
    mood_track = track.get("Mood", "").lower()
    if mood_attr and mood_track:
        attr_words = set(mood_attr.replace("_", " ").split())
        track_words = set(mood_track.replace("_", " ").split())
        shared = attr_words & track_words
        score_val += len(shared) * 0.5
    return score_val


def best_match(attrs: Dict[str, Any], library_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(library_path, "r", encoding="utf-8") as f:
            library = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read library: {exc}", file=sys.stderr)
        return None

    tracks = library.get("tracks", [])
    if not tracks:
        return None

    scored = [(score(attrs, t), t) for t in tracks]
    best_scored, best_track = max(scored, key=lambda x: x[0])

    if best_scored <= 0:
        print(f"No close match in library (best score: {best_scored:.2f}).", file=sys.stderr)
        return None

    return best_track


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full video-to-music pipeline in one command."
    )
    parser.add_argument("-v", "--video", dest="video", help="Path to the input .mp4 video file.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=FRAMES_DIR_DEFAULT,
        help=f"Directory to save extracted frames (default: {FRAMES_DIR_DEFAULT}).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Interval in seconds between saved frames (default: 2.0).",
    )
    parser.add_argument(
        "--frames-dir",
        dest="output_dir",
        help="Alias for positional output_dir (for clarity).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract.",
    )
    parser.add_argument(
        "--dedup/--no-dedup",
        dest="dedup",
        action="store_true",
        default=True,
        help="Skip duplicate frames (default: True).",
    )
    parser.add_argument(
        "--scene-change",
        action="store_true",
        help="Detect scene changes instead of uniform sampling.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip the lo-fi follow-up question in suggest_song_attributes.",
    )
    parser.add_argument(
        "--lofi",
        action="store_true",
        help="Skip straight to a lo-fi-styled music suggestion.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Length of generated music in seconds (default: 10).",
    )
    parser.add_argument(
        "--output",
        default=MUSIC_OUT_DEFAULT,
        help=f"Output WAV file path (default: {MUSIC_OUT_DEFAULT}).",
    )
    parser.add_argument(
        "--library",
        default=LIBRARY_DEFAULT,
        help=f"Path to the mood index JSON (default: {LIBRARY_DEFAULT}).",
    )
    parser.add_argument(
        "--no-library",
        action="store_true",
        help="Skip the library — always generate new music.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible music generation.",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=1,
        help="Number of music variations to generate (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the pipeline steps without generating audio.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file for pipeline defaults.",
    )

    from _device import add_gpu_args
    add_gpu_args(parser)

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        return 1

    frames_dir = Path(args.output_dir)
    music_out = Path(args.output)
    library_path = Path(args.library)

    cfg = PipelineConfig.from_args(args)

    if cfg.dry_run:
        print("=== DRY RUN MODE ===")
        print(f"Video: {video_path}")
        print(f"Frames dir: {frames_dir}")
        print(f"Interval: {cfg.interval}s")
        print(f"Max frames: {cfg.max_frames}")
        print(f"Dedup: {cfg.dedup}")
        print(f"Scene change: {cfg.scene_change}")
        print(f"GPU: {cfg.use_gpu}")
        print(f"Library: {library_path}")
        print(f"Output: {music_out}")
        print(f"Duration: {cfg.duration}s")
        print(f"Seed: {cfg.seed}")
        print(f"Variations: {cfg.variations}")
        print()

    try:
        print("=== Step 1: Extract frames ===")
        if cfg.dry_run:
            print(f"[Dry run] Would run: extract_frames.py {video_path} {frames_dir}")
        else:
            extract_args = [str(video_path), str(frames_dir), f"--interval={cfg.interval}"]
            if cfg.max_frames:
                extract_args.append(f"--max-frames={cfg.max_frames}")
            if not cfg.dedup:
                extract_args.append("--no-dedup")
            if cfg.scene_change:
                extract_args.append("--scene-change")
            extract_args.extend(gpu_arg(cfg.use_gpu))
            r = run("extract_frames.py", *extract_args)
            print(r.stderr, flush=True)

        print("=== Step 2: Describe video with CLIP ===")
        if cfg.dry_run:
            print(f"[Dry run] Would run: describe_video.py {frames_dir}")
        else:
            describe_args = [str(frames_dir)]
            describe_args.extend(gpu_arg(cfg.use_gpu))
            r = run("describe_video.py", *describe_args, capture=True)
            print(r.stderr, flush=True)
            raw_stdout = r.stdout

            desc_marker = "\n=== Per-frame score breakdown ==="
            if desc_marker in raw_stdout:
                plain_desc, _ = raw_stdout.split(desc_marker, 1)
            else:
                plain_desc = raw_stdout

            description = plain_desc.strip()
            if not description:
                print("Error: describe_video produced no output.", file=sys.stderr)
                return 1

        print("=== Step 3: Suggest music attributes ===")
        attr_args = [f"--description={description}"]
        if cfg.lofi:
            attr_args.append("--lofi")
        if cfg.no_interactive:
            attr_args.append("--no-interactive")
        attr_args.extend(gpu_arg(cfg.use_gpu))

        if cfg.dry_run:
            print(f"[Dry run] Would run: suggest_song_attributes.py {description}")
        else:
            r = run("suggest_song_attributes.py", *attr_args, capture=True)
            print(r.stderr, flush=True)
            attrs_raw = r.stdout.strip()
            try:
                attrs = MusicAttributes.from_dict(json.loads(attrs_raw))
            except json.JSONDecodeError:
                print(
                    f"Error: suggest_song_attributes returned invalid JSON:\n{attrs_raw}",
                    file=sys.stderr,
                )
                return 1

        print("=== Step 4: Pick best match / generate music ===")
        if cfg.dry_run:
            print(f"[Dry run] Would run: generate_music.py or library match")
            return 0

        if not cfg.no_library:
            match = best_match(attrs.to_dict(), library_path)
            if match:
                track_path = Path(match["path"])
                if track_path.is_file():
                    print(f"Using best match: {match['file']} → {music_out}")
                    import shutil
                    shutil.copy2(track_path, music_out)
                    print(f"Copied {music_out}")
                else:
                    print(
                        f"Library track not found on disk: {track_path}. Falling back to generation.",
                        file=sys.stderr,
                    )
                    match = None
            if not match:
                gen_args = [
                    f"--attributes={json.dumps(attrs.to_dict())}",
                    f"--output={music_out}",
                    f"--duration={cfg.duration}",
                ]
                if cfg.seed is not None:
                    gen_args.append(f"--seed={cfg.seed}")
                if cfg.variations > 1:
                    gen_args.append(f"--variations={cfg.variations}")
                gen_args.extend(gpu_arg(cfg.use_gpu))
                r = run("generate_music.py", *gen_args)
                print(r.stderr, flush=True)
        else:
            gen_args = [
                f"--attributes={json.dumps(attrs.to_dict())}",
                f"--output={music_out}",
                f"--duration={cfg.duration}",
            ]
            if cfg.seed is not None:
                gen_args.append(f"--seed={cfg.seed}")
            if cfg.variations > 1:
                gen_args.append(f"--variations={cfg.variations}")
            gen_args.extend(gpu_arg(cfg.use_gpu))
            r = run("generate_music.py", *gen_args)
            print(r.stderr, flush=True)

        print(f"\n=== Done! Music saved to {music_out} ===")

    except subprocess.CalledProcessError as exc:
        print(f"Error: script failed with exit code {exc.returncode}", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())