"""Device management utilities for GPU/CPU selection."""

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DeviceConfig:
    device: torch.device
    dtype: torch.dtype
    use_gpu: bool

    def __repr__(self) -> str:
        return f"DeviceConfig(device={self.device}, dtype={self.dtype}, use_gpu={self.use_gpu})"


def resolve_device(requested: Optional[bool], quiet: bool = False) -> DeviceConfig:
    if requested is None:
        use_gpu = torch.cuda.is_available()
    elif requested:
        use_gpu = torch.cuda.is_available()
        if not use_gpu and not quiet:
            print("Warning: GPU requested but CUDA not available. Using CPU.", file=sys.stderr)
    else:
        use_gpu = False

    if use_gpu:
        return DeviceConfig(device=torch.device("cuda"), dtype=torch.float16, use_gpu=True)
    return DeviceConfig(device=torch.device("cpu"), dtype=torch.float32, use_gpu=False)


def add_gpu_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gpu",
        dest="use_gpu",
        action="store_true",
        default=None,
        help="Explicitly enable GPU (if available).",
    )
    group.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        default=None,
        help="Explicitly disable GPU, use CPU only.",
    )