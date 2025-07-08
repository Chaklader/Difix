#!/usr/bin/env python3
"""
Convert a Difix3D / 3DGS training checkpoint (``ckpt_xxxxx_rank0.pt``)
to a binary PLY file compatible with your working PLY format.

The checkpoint is expected to have been produced by Difix3D training
and contains a ``splats`` dict with entries such as ``means``, ``scales``, 
``quats``, ``opacities`` and spherical–harmonic coefficients (``sh0`` / ``shN``).

Usage
-----
python ckpt_to_ply.py \
    --ckpt   /path/to/ckpt_40099_rank0.pt \
    --output NeRF.ply

The output format matches your working PLY format exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


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


def _print_splat_info(splats: dict[str, torch.Tensor]) -> None:
    """Print diagnostic information about the splats."""
    N = splats["means"].shape[0]
    print(f"\n=== Splat Information ===")
    print(f"Total Gaussians: {N:,}")
    
    for key, tensor in splats.items():
        print(f"{key}: {tensor.shape}")
    
    if "opacities" in splats:
        opacities = torch.sigmoid(splats["opacities"]).squeeze()
        print(f"Opacity range: {opacities.min():.3f} - {opacities.max():.3f}")
        high_opacity = (opacities > 0.05).sum().item()
        print(f"High opacity (>0.05): {high_opacity:,} ({high_opacity/N:.1%})")
    
    if "scales" in splats:
        scales = torch.exp(splats["scales"])
        print(f"Scale range: {scales.min():.3f} - {scales.max():.3f}")
        avg_scale = scales.mean()
        print(f"Average scale: {avg_scale:.3f}")
    
    if "quats" in splats:
        quats = splats["quats"]
        print(f"Quaternion range: {quats.min():.3f} - {quats.max():.3f}")
        # Check if first component looks like w (should be close to ±1 for unit quaternions)
        first_component_abs_mean = torch.abs(quats[:, 0]).mean().item()
        print(f"First quat component abs mean: {first_component_abs_mean:.3f}")
        if first_component_abs_mean > 0.7:
            print("  -> Likely (w,x,y,z) format")
        else:
            print("  -> Likely (x,y,z,w) format")
    print("=" * 25)


def _write_binary_ply_compatible(out_path: Path, splats: dict[str, torch.Tensor], scale_offset: float = 0.0) -> None:
    """Write PLY in the EXACT format that matches your working model, with optional scale adjustment."""
    N = splats["means"].shape[0]
    print(f"\n=== Converting {N:,} Gaussians to Compatible PLY ===")
    
    # Positions
    xyz = splats["means"].cpu().numpy().astype("<f4")  # (N,3)
    
    # Adjust scales with optional offset (add to log-scale for larger Gaussians)
    adjusted_scales = splats["scales"].cpu().numpy().astype("<f4") + scale_offset  # (N,3) - keep in log space!
    
    # Generate normal vectors (fake them - typically zeros are fine)
    normals = np.zeros((N, 3), dtype="<f4")  # (N,3) - zeros work fine
    
    # Colors (DC coefficients) - DON'T multiply by SH_C0
    sh0 = splats["sh0"].squeeze(1).cpu().numpy().astype("<f4")  # (N,3)
    
    # Higher order SH coefficients 
    shN = splats["shN"].cpu().numpy().astype("<f4")
    if shN.ndim == 3:  # (N, 15, 3) format
        print("Reshaping SH coefficients from (N,15,3) to (N,45)")
        shN_reorganized = np.zeros((N, 45), dtype="<f4")
        for c in range(3):
            shN_reorganized[:, c*15:(c+1)*15] = shN[:, :, c]
        shN = shN_reorganized
    elif shN.shape[1] != 45:
        print(f"Reshaping SH coefficients from ({N},{shN.shape[1]}) to ({N},45)")
        shN = shN.reshape(N, -1)
        if shN.shape[1] > 45:
            shN = shN[:, :45]  # Truncate if too many
        elif shN.shape[1] < 45:
            # Pad with zeros if too few
            padding = np.zeros((N, 45 - shN.shape[1]), dtype="<f4")
            shN = np.concatenate([shN, padding], axis=1)
    
    # CRITICAL: Use RAW opacities (don't apply sigmoid!) - keep original values
    opacity = splats["opacities"].cpu().numpy().astype("<f4").reshape(N, 1)  # Raw values!
    
    # CRITICAL: Use RAW quaternions (DON'T normalize!) - keep original values
    quats = splats["quats"].cpu().numpy().astype("<f4")  # (N,4) - RAW values!
    
    # Handle quaternion format detection WITHOUT normalization
    first_component_abs_mean = np.abs(quats[:, 0]).mean()
    if first_component_abs_mean > 0.7:
        # Likely (w,x,y,z) format - keep as (w,x,y,z) for good PLY format
        rot = quats[:, [0, 1, 2, 3]]  # Keep as (w,x,y,z) = (rot_0, rot_1, rot_2, rot_3)
        print("Quaternions detected as (w,x,y,z) - keeping order")
    else:
        # (x,y,z,w) format - reorder to (w,x,y,z) for good PLY format
        rot = quats[:, [3, 0, 1, 2]]  # Reorder to (w,x,y,z)
        print("Quaternions detected as (x,y,z,w) - reordering to (w,x,y,z)")
    
    # CRITICAL: Match the EXACT field order from good PLY
    # Order: x,y,z, nx,ny,nz, f_dc_0,f_dc_1,f_dc_2, f_rest_0...f_rest_44, opacity, scale_0,scale_1,scale_2, rot_0,rot_1,rot_2,rot_3
    data = np.concatenate([
        xyz,           # x, y, z (3)
        normals,       # nx, ny, nz (3)  
        sh0,           # f_dc_0, f_dc_1, f_dc_2 (3)
        shN,           # f_rest_0 through f_rest_44 (45)
        opacity,       # opacity (1)
        adjusted_scales,       # scale_0, scale_1, scale_2 (3)
        rot            # rot_0, rot_1, rot_2, rot_3 (4)
    ], axis=1)
    
    print(f"Final data shape: {data.shape} (expected: {N} x 62)")
    assert data.shape[1] == 62, f"Expected 62 properties, got {data.shape[1]}"
    
    # Header - EXACT field order from good PLY
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y", 
        "property float z",
        "property float nx",      # Normal vectors
        "property float ny",
        "property float nz",
        "property float f_dc_0",  # DC coefficients first
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    
    # f_rest_0 through f_rest_44
    header_lines += [f"property float f_rest_{i}" for i in range(45)]
    
    # Then opacity, scales, rotations
    header_lines += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1", 
        "property float scale_2",
        "property float rot_0",    # w component
        "property float rot_1",    # x component
        "property float rot_2",    # y component
        "property float rot_3"     # z component
    ]
    
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with out_path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    print(f"Compatible PLY written to {out_path} with {N:,} vertices.")
    print("=" * 45)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert Difix3D GS checkpoint to compatible PLY format")
    parser.add_argument("--ckpt", required=True, type=Path, help="Path to ckpt_*.pt file")
    parser.add_argument("--output", required=True, type=Path, help="Output .ply path")
    
    # Filtering options (optional, can be removed for no filtering)
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--real-estate", action="store_true", 
                        help="Apply real estate optimized filtering (recommended)")
    filter_group.add_argument("--smart-lod", type=float, metavar="PERCENTAGE",
                        help="Smart LOD filtering keeping percentage of most important Gaussians (e.g., 0.4 for 40%%)")
    filter_group.add_argument("--custom-filter", action="store_true",
                        help="Use custom opacity/scale thresholds")
    filter_group.add_argument("--scene-bounds", type=float, metavar="DISTANCE", default=None,
                        help="Filter by scene bounds - keep Gaussians within DISTANCE from center")
    filter_group.add_argument("--opacity-preserving", action="store_true",
                        help="Advanced filtering that preserves opacity transitions (recommended for quality)")
    
    # Custom filter parameters
    parser.add_argument("--min-opacity", type=float, default=0.005,
                        help="Minimum opacity threshold for custom filtering (default: 0.005)")
    parser.add_argument("--max-scale", type=float, default=0.1,
                        help="Maximum scale threshold for custom filtering (default: 0.1)")
    
    # Optional scale offset to adjust Gaussian sizes
    parser.add_argument("--scale-offset", type=float, default=0.0,
                        help="Add to log-scales to make Gaussians larger (default: 0.0)")
    
    args = parser.parse_args(argv)

    splats = _load_splats(args.ckpt)
    _print_splat_info(splats)
    
    # Apply filtering based on selected option, if any
    if args.opacity_preserving:
        splats = _filter_opacity_preserving(splats)
    elif args.scene_bounds is not None:
        splats = _filter_by_scene_bounds(splats, args.scene_bounds)
    elif args.real_estate:
        splats = _filter_for_real_estate_viewing(splats)
    elif args.smart_lod is not None:
        splats = _filter_smart_lod(splats, args.smart_lod)
    elif args.custom_filter:
        splats = _filter_splats_for_export(splats, args.min_opacity, args.max_scale)
    else:
        print("\nNo filtering applied - using full model")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_ply_compatible(args.output, splats, args.scale_offset)


if __name__ == "__main__":
    main()
