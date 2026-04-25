"""
Extract one frame every 2 seconds from an .mp4 video using OpenCV.

Usage:
    python scripts/extract_frames.py <input_video.mp4> [output_dir]

If output_dir is not provided, frames are saved to ./frames/.
"""

import argparse
import os
import sys

import cv2


def extract_frames(video_path: str, output_dir: str, interval_seconds: float = 2.0) -> int:
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

    frame_interval = max(1, int(round(fps * interval_seconds)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f} | Total frames: {total_frames} | Saving every {frame_interval} frames")

    saved_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp_seconds = frame_index / fps
            filename = f"frame_{saved_count:04d}_t{timestamp_seconds:07.2f}s.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")
    return saved_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a frame every N seconds from a video.")
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
    args = parser.parse_args()

    try:
        extract_frames(args.video, args.output_dir, args.interval)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
