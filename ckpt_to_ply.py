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


def _write_binary_ply_3dgs(out_path: Path, splats: dict[str, torch.Tensor]) -> None:
    """Write PLY in GraphDeco 3DGS layout (binary_little_endian)."""
    N = splats["means"].shape[0]
    # Prepare columns -----------------------------------------------------
    xyz = splats["means"].cpu().numpy().astype("<f4")  # (N,3)

    scales = np.exp(splats["scales"].cpu().numpy()).astype("<f4")  # (N,3)

    quats = splats["quats"].cpu().numpy().astype("<f4")  # (N,4) assumed (x,y,z,w)
    # reorder to rot_1 rot_2 rot_3 rot_0 (x y z w)
    rot = quats[:, [0, 1, 2, 3]]

    opacity = torch.sigmoid(splats["opacities"]).cpu().numpy().astype("<f4").reshape(N, 1)

    sh0 = (_SH_C0 * splats["sh0"].squeeze(1)).cpu().numpy().astype("<f4")  # (N,3)

    shN = splats["shN"].cpu().numpy().astype("<f4").reshape(N, -1)  # (N,45)

    # Stack all fields
    data = np.concatenate([xyz, scales, rot, opacity, sh0, shN], axis=1)

    assert data.shape[1] == 59, f"Expected 59 properties per vertex, got {data.shape[1]}"

    # Header --------------------------------------------------------------
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property float rot_0",
        "property float opacity",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    # f_rest_0 … f_rest_44
    header_lines += [f"property float f_rest_{i}" for i in range(45)]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with out_path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    print(f"Binary PLY written to {out_path} with {N} vertices.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Difix3D GS checkpoint to ASCII PLY")
    parser.add_argument("--ckpt", required=True, type=Path, help="Path to ckpt_*.pt file")
    parser.add_argument("--output", required=True, type=Path, help="Output .ply path")
    parser.add_argument("--format", choices=["ascii", "3dgs"], default="ascii",
                        help="PLY flavour: simple ascii or 3dgs binary (default: ascii)")
    args = parser.parse_args(argv)

    splats = _load_splats(args.ckpt)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "ascii":
        xyz, rgb = _splats_to_ply_arrays(splats)
        _write_ascii_ply(args.output, xyz, rgb)
    else:
        _write_binary_ply_3dgs(args.output, splats)


if __name__ == "__main__":
    main()
