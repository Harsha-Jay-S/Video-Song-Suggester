"""
Scan a folder of audio files and build a mood index mapping each track to
its Genre / Tempo / Mood profile.

Usage:
    python scripts/analyze_music.py <folder> [--index-file library.json]
    python scripts/analyze_music.py <folder> --no-gpu  # force CPU
    python scripts/analyze_music.py <folder> --dry-run   # preview without writing
    python scripts/analyze_music.py <folder> --parallel 4  # parallel processing
"""

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import librosa
except ImportError:
    print("Error: librosa not installed. Run: uv add librosa", file=sys.stderr)
    sys.exit(1)

from _device import add_gpu_args, resolve_device, DeviceConfig

SLOW_MAX = 90
FAST_MIN = 140


@dataclass
class TrackProfile:
    file: str
    path: str
    Genre: str
    Tempo: str
    Mood: str
    bpm: Optional[int] = None
    energy: Optional[float] = None
    key_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "path": self.path,
            "Genre": self.Genre,
            "Tempo": self.Tempo,
            "Mood": self.Mood,
        }


def estimate_tempo(y: List[float], sr: int) -> int:
    try:
        onset_env = librosa.onset.onset_envelope(y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope= onset_env, sr=sr)
        bpm = int(round(float(tempo)))
        return max(40, min(220, bpm))
    except Exception:
        return 120


def estimate_energy(y: List[float], sr: int) -> float:
    rms = librosa.feature.rms(y=y)[0]
    return float(math.nanmean(rms))


def estimate_tonality(y: List[float], sr: int) -> str:
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.08, 2.02, 5.19]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 3.92, 2.69, 4.02, 5.19, 2.39, 3.98, 2.69, 5.32]
        key_profile = major_profile if chroma.mean() > 0.3 else minor_profile
        return "major" if key_profile[0] > 5.5 else "minor"
    except Exception:
        return "major"


def tempo_label(bpm: int) -> str:
    if bpm < SLOW_MAX:
        return "Slow"
    if bpm > FAST_MIN:
        return "Fast"
    return "Medium"


def energy_label(energy: float) -> str:
    if energy < 0.03:
        return "Calm"
    if energy > 0.08:
        return "Energetic"
    return "Balanced"


def infer_mood(energy: float, bpm: int, key_mode: str) -> str:
    if energy < 0.02 and bpm < 80:
        return "Calm and peaceful"
    if key_mode == "minor" and energy < 0.05:
        return "Melancholic"
    if bpm > FAST_MIN and energy > 0.06:
        return "Energetic and exciting"
    if energy > 0.05 and bpm < 80:
        return "Uplifting"
    return "Balanced"


def analyze_track(path: Path) -> Dict[str, Any]:
    try:
        y, sr = librosa.load(path, duration=30.0, offset=5.0)
        bpm = estimate_tempo(y, sr)
        energy = estimate_energy(y, sr)
        key_mode = estimate_tonality(y, sr)
        tempo_cat = tempo_label(bpm)
        mood = infer_mood(energy, bpm, key_mode)
        return TrackProfile(
            file=path.name,
            path=str(path),
            Genre=path.stem.replace("_", " ").replace("-", " ").title(),
            Tempo=tempo_cat,
            Mood=mood,
            bpm=bpm,
            energy=energy,
            key_mode=key_mode,
        ).to_dict()
    except Exception as exc:
        print(f"Warning: failed to analyze {path.name}: {exc}", file=sys.stderr)
        return TrackProfile(
            file=path.name,
            path=str(path),
            Genre=path.stem.replace("_", " ").replace("-", " ").title(),
            Tempo="Medium",
            Mood="Balanced",
        ).to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan audio files and build a mood index."
    )
    parser.add_argument("folder", help="Folder containing .mp3 / .wav / .flac files.")
    parser.add_argument(
        "--index-file",
        default="library.json",
        help="Output index file (default: library.json).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview results without writing the index file.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential).",
    )
    add_gpu_args(parser)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        return 1

    device_cfg = resolve_device(args.use_gpu)
    print(f"Using device: {device_cfg.device}", file=sys.stderr)

    exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    pattern = "**/*" if args.recursive else "*"
    paths = sorted(
        p for p in folder.glob(pattern)
        if p.suffix.lower() in exts and p.is_file()
    )
    if not paths:
        print(f"No audio files found in {folder}.", file=sys.stderr)
        return 1

    tracks: List[Dict[str, Any]] = []

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(analyze_track, p): p for p in paths}
            for idx, future in enumerate(as_completed(futures), 1):
                print(f"[{idx}/{len(paths)}] Analyzing...", file=sys.stderr, flush=True)
                tracks.append(future.result())
    else:
        for idx, path in enumerate(paths, 1):
            print(f"[{idx}/{len(paths)}] {path.name}...", file=sys.stderr, flush=True)
            tracks.append(analyze_track(path))

    index = {"version": 2, "tracks": tracks, "count": len(tracks)}
    print(f"\nIndex built with {len(tracks)} tracks.", file=sys.stderr)

    if args.dry_run:
        print("\n=== Dry run — preview ===", file=sys.stderr)
        print(json.dumps(index, indent=2))
        print("\n[Dry run] Skipping file write.")
        return 0

    out = Path(args.index_file)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\nIndex saved to {out} ({len(tracks)} tracks).", file=sys.stderr)
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())