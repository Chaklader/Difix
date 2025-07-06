#!/usr/bin/env python3
"""Quick introspection utility for Difix3D / 3DGS checkpoints.

Example
-------
$ python inspect_ckpt.py --ckpt /path/to/ckpt_40099_rank0.pt

It prints the top-level keys and, if a ``splats`` dict is present, the
name, dtype, shape and basic stats of every tensor inside it.  This
helps debugging export errors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _tensor_stats(t):
    if not torch.is_tensor(t):
        return "non-tensor"
    return f"{t.dtype}, {tuple(t.shape)}, min={t.min():.3f}, max={t.max():.3f}"


def inspect_checkpoint(path: Path):
    print(f"Loading {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print("Top-level keys:", list(ckpt.keys()))

    if "splats" in ckpt and isinstance(ckpt["splats"], dict):
        print("\n=== splats tensors ===")
        for k, v in ckpt["splats"].items():
            print(f"{k:12s}: {_tensor_stats(v)}")
    else:
        print("No 'splats' dict found – this may not be a GS checkpoint.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Inspect a Difix3D checkpoint")
    p.add_argument("--ckpt", required=True, type=Path, help="Path to ckpt_*.pt")
    args = p.parse_args(argv)

    inspect_checkpoint(args.ckpt)


if __name__ == "__main__":
    main()
