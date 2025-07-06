#!/usr/bin/env python3
"""
Convert a Difix3D / 3DGS training checkpoint (``ckpt_xxxxx_rank0.pt``)
to an uncompressed PLY file that can be opened in Meshlab / Blender or
converted to other Gaussian-splat formats.

The checkpoint is expected to have been produced by
``examples/gsplat/simple_trainer_difix3d.py`` and therefore contains a
``splats`` dict with entries such as ``means``, ``scales``, ``quats``,
``opacities`` and spherical–harmonic coefficients (``sh0`` / ``shN``).

For a quick visualisation the script exports for every Gaussian only:
    • xyz position (``means``)
    • RGB colour derived from the first SH coefficient (``sh0``)

Additional parameters (scale, opacity …) could be added as PLY
properties later if desired.

Usage
-----
python ckpt_to_ply.py \
    --ckpt   /path/to/ckpt_40099_rank0.pt \
    --output /path/to/model_40100.ply

The output format is ASCII PLY for maximum compatibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# Constant for the SH l=0 basis ( 1 / sqrt(4 * pi) )
_SH_C0 = 0.28209479177387814


def _load_splats(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load the *splats* dict from a Difix3D checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "splats" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain a 'splats' key – is this a Difix3D GS checkpoint?"
        )
    splats: dict[str, torch.Tensor] = ckpt["splats"]
    return splats


def _splats_to_ply_arrays(splats: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Return vertex positions (N,3) and colours (N,3 uint8) ready for PLY writing."""
    if "means" not in splats:
        raise KeyError("'means' tensor not found in splats – cannot export positions.")

    means = splats["means"]  # (N,3)
    if means.ndim != 2 or means.size(1) != 3:
        raise ValueError("'means' tensor expected shape (N,3)")
    xyz = means.cpu().numpy().astype(np.float32)

    # ----- colour -----
    if "sh0" in splats:
        sh0 = splats["sh0"]  # (N,1,3) or (N,3)
        if sh0.ndim == 3:
            sh0 = sh0.squeeze(1)  # (N,3)
        rgb_linear = (_SH_C0 * sh0).cpu().numpy()  # convert SH0 to linear RGB
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    elif "colors" in splats:  # fallback
        colors = splats["colors"]  # (N,3)
        rgb_linear = np.clip(colors.cpu().numpy(), 0.0, 1.0)
    else:
        print("Warning: no 'sh0' or 'colors' in checkpoint – using grey colour.")
        rgb_linear = np.full_like(xyz, 0.5, dtype=np.float32)

    # Ensure we have 3 colour channels
    if rgb_linear.ndim == 1:
        rgb_linear = np.repeat(rgb_linear[:, None], 3, axis=1)
    elif rgb_linear.shape[1] == 1:
        rgb_linear = np.repeat(rgb_linear, 3, axis=1)

    rgb_uint8 = (rgb_linear * 255).astype(np.uint8)
    return xyz, rgb_uint8


def _write_ascii_ply(out_path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write an ASCII PLY with x y z r g b."""
    n = xyz.shape[0]
    print(f"Writing {n} vertices to {out_path}")

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with out_path.open("w") as f:
        f.write(header)
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    print("Done.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Difix3D GS checkpoint to ASCII PLY")
    parser.add_argument("--ckpt", required=True, type=Path, help="Path to ckpt_*.pt file")
    parser.add_argument("--output", required=True, type=Path, help="Output .ply path")
    args = parser.parse_args(argv)

    splats = _load_splats(args.ckpt)
    xyz, rgb = _splats_to_ply_arrays(splats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_ascii_ply(args.output, xyz, rgb)


if __name__ == "__main__":
    main()
