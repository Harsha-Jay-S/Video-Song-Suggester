"""
One-shot pipeline: extract frames → describe video → suggest music attributes →
pick the best match from a library OR generate new music.

Usage:
    python run_me.py --video input.mp4 [output_dir]
    python run_me.py --video input.mp4 --interval 1.0
    python run_me.py --video input.mp4 --library /path/to/library.json
    python run_me.py --video input.mp4 --no-library   # always generate
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent / "scripts"
FRAMES_DIR_DEFAULT = "frames"
MUSIC_OUT_DEFAULT = "background_music.wav"
LIBRARY_DEFAULT = "library.json"


def run(script_name: str, *args, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT_DIR / script_name), *args]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=True,
    )


def score(attr: dict, track: dict) -> float:
    score = 0.0
    if attr.get("Genre", "").lower() in track.get("Genre", "").lower():
        score += 2.0
    if attr.get("Tempo") == track.get("Tempo"):
        score += 1.0
    mood_attr = attr.get("Mood", "").lower()
    mood_track = track.get("Mood", "").lower()
    if mood_attr and mood_track:
        attr_words = set(mood_attr.replace("_", " ").split())
        track_words = set(mood_track.replace("_", " ").split())
        shared = attr_words & track_words
        score += len(shared) * 0.5
    return score


def best_match(attrs: dict, library_path: Path) -> dict | None:
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
        print(
            f"No close match in library (best score: {best_scored:.2f}).",
            file=sys.stderr,
        )
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
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        return 1

    frames_dir = Path(args.output_dir)
    music_out = Path(args.output)
    library_path = Path(args.library)

    try:
        print("=== Step 1: Extract frames ===")
        r = run(
            "extract_frames.py",
            str(video_path),
            str(frames_dir),
            f"--interval={args.interval}",
        )
        print(r.stdout, flush=True)

        print("=== Step 2: Describe video with CLIP ===")
        r = run("describe_video.py", str(frames_dir), capture=True)
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
        if args.lofi:
            attr_args.append("--lofi")
        if args.no_interactive:
            attr_args.append("--no-interactive")
        r = run("suggest_song_attributes.py", *attr_args, capture=True)
        print(r.stderr, flush=True)
        attrs_raw = r.stdout.strip()
        try:
            attrs = json.loads(attrs_raw)
        except json.JSONDecodeError:
            print(
                f"Error: suggest_song_attributes returned invalid JSON:\n{attrs_raw}",
                file=sys.stderr,
            )
            return 1

        print("=== Step 4: Pick best match / generate music ===")
        if not args.no_library:
            match = best_match(attrs, library_path)
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
                r = run(
                    "generate_music.py",
                    f"--attributes={json.dumps(attrs)}",
                    f"--output={music_out}",
                    f"--duration={args.duration}",
                )
                print(r.stderr, flush=True)
        else:
            r = run(
                "generate_music.py",
                f"--attributes={json.dumps(attrs)}",
                f"--output={music_out}",
                f"--duration={args.duration}",
            )
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