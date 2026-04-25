"""
Extract frames from a video using smarter sampling:
- Uniform interval extraction (default)
- Per-frame hashing to skip visually identical/similar frames
- Keyframe detection using scene-change analysis

Usage:
    python scripts/extract_frames.py <input_video.mp4> [output_dir]
    python scripts/extract_frames.py <input_video.mp4> [output_dir] --interval 2.0
    python scripts/extract_frames.py <input_video.mp4> --no-gpu  # force CPU
    python scripts/extract_frames.py <input_video.mp4> --scene-change  # detect scene changes
    python scripts/extract_frames.py <input_video.mp4> --dedup   # skip duplicate frames
    python scripts/extract_frames.py <input_video.mp4> --max-frames 30  # limit max frames
"""

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
from tqdm import tqdm


DEFAULT_SIMILARITY_THRESHOLD = 0.95


def image_similarity_hash(img1, img2) -> float:
    if img1.shape != img2.shape:
        return 0.0
    diff = cv2.absdiff(img1, img2).astype(float) / 255.0
    max_diff = math.sqrt(float((diff ** 2).sum()))
    return max(0.0, 1.0 - max_diff / math.sqrt(float(img1.size)))


def detect_scene_changes(cap: cv2.VideoCapture, threshold: float = 30.0) -> list[int]:
    keyframe_indices = [0]
    ret, prev_frame = cap.read()
    if not ret:
        return keyframe_indices
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(prev_gray, gray)
        mean_diff = float(diff.mean())

        if mean_diff > threshold:
            keyframe_indices.append(frame_idx)
            prev_gray = gray
        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return keyframe_indices


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 2.0,
    max_frames: Optional[int] = None,
    dedup: bool = True,
    scene_change: bool = False,
    use_gpu: bool = False,
) -> int:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if scene_change:
        frame_indices = detect_scene_changes(cap)
    else:
        frame_interval = max(1, int(round(fps * interval_seconds)))
        frame_indices = list(range(0, total_frames, frame_interval))

    if max_frames and len(frame_indices) > max_frames:
        step = len(frame_indices) / max_frames
        frame_indices = [frame_indices[int(i * step)] for i in range(max_frames)]

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f} | Total frames: {total_frames}", file=sys.stderr)
    print(f"Will save {len(frame_indices)} frames (max_frames={max_frames}, dedup={dedup})", file=sys.stderr)

    saved_count = 0
    last_frame = None
    last_hash = None

    target_positions = sorted(set(frame_indices))

    for target_pos in tqdm(target_positions, desc="Extracting frames", unit="frame", disable=not sys.stderr.isatty()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
        ret, frame = cap.read()
        if not ret:
            continue

        if dedup:
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            if frame_hash == last_hash:
                continue
            last_hash = frame_hash

            if last_frame is not None:
                sim = image_similarity_hash(last_frame, frame)
                if sim > DEFAULT_SIMILARITY_THRESHOLD:
                    continue

        timestamp_seconds = target_pos / fps
        filename = f"frame_{saved_count:04d}_t{timestamp_seconds:07.2f}s.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)
        saved_count += 1
        last_frame = frame

    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}", file=sys.stderr)
    return saved_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frames from a video with smart sampling.")
    parser.add_argument("video", help="Path to the input .mp4 video file.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="frames",
        help="Directory to save extracted frames (default: ./frames).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Interval in seconds between saved frames (default: 2.0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=True,
        help="Skip visually similar/duplicate frames (default: True).",
    )
    parser.add_argument(
        "--no-dedup",
        dest="dedup",
        action="store_false",
        help="Disable duplicate frame detection.",
    )
    parser.add_argument(
        "--scene-change",
        action="store_true",
        help="Detect scene changes instead of uniform interval sampling.",
    )
    args = parser.parse_args()

    try:
        extract_frames(
            args.video,
            args.output_dir,
            interval_seconds=args.interval,
            max_frames=args.max_frames,
            dedup=args.dedup,
            scene_change=args.scene_change,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())