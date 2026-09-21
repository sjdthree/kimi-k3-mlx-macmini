"""Shared test guards.

Ten of the suites compare against `k3ref`, which reads the real 1.42 TiB Kimi-K3
checkpoint. Without it they used to die in `K3Config.load()` with a FileNotFoundError
naming a volume that only ever existed on the author's machine. They now skip, so a
fresh clone runs the checkpoint-free suites green and says plainly what the rest need.

Point `K3_CKPT` at the checkpoint to run them:

    K3_CKPT=/Volumes/YourDrive/Kimi-K3 python tests/test_moe_mlx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
CKPT = Path(os.environ.get("K3_CKPT", "Kimi-K3"))

__all__ = ["CKPT", "checkpoint_available", "require_checkpoint"]


def checkpoint_available() -> bool:
    return (CKPT / "config.json").is_file()


def require_checkpoint() -> None:
    """Exit 0 with an explanation when the checkpoint is not reachable."""
    if not checkpoint_available():
        print(f"SKIP: no Kimi-K3 checkpoint at {CKPT}")
        print("      set K3_CKPT=/path/to/Kimi-K3 to run this suite")
        raise SystemExit(0)
