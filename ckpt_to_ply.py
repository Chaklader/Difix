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


def _filter_by_scene_bounds(splats: dict[str, torch.Tensor], max_distance: float = 15.0) -> dict[str, torch.Tensor]:
    """Filter Gaussians to match compact scene bounds of working model."""
    N_original = splats["means"].shape[0]
    
    means = splats["means"]
    center = means.mean(dim=0)
    distances = torch.norm(means - center, dim=1)
    mask = distances < max_distance
    
    filtered_splats = {}
    for key, tensor in splats.items():
        filtered_splats[key] = tensor[mask]
    
    N_filtered = filtered_splats["means"].shape[0]
    reduction = (N_original - N_filtered) / N_original
    print(f"\n=== Scene Bounds Filtering Results ===")
    print(f"Original: {N_original:,} Gaussians")
    print(f"Filtered: {N_filtered:,} Gaussians")
    print(f"Reduction: {reduction:.1%}")
    print(f"Max distance from center: {max_distance}")
    print("=" * 35)
    return filtered_splats


def _filter_opacity_preserving(splats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Filter while preserving opacity distribution similar to working model."""
    N_original = splats["means"].shape[0]
    
    # Get raw opacity values (not sigmoid)
    raw_opacities = splats["opacities"].squeeze()
    sigmoid_opacities = torch.sigmoid(raw_opacities)
    scales = torch.exp(splats["scales"])
    
    # Create importance score that preserves low-opacity transitions
    # This mimics the bimodal opacity distribution from the working model
    importance_high = sigmoid_opacities ** 2  # Favor high opacity
    importance_low = (1.0 - sigmoid_opacities) * 0.3  # Keep some low opacity for transitions
    importance_combined = importance_high + importance_low
    
    # Add scale component (smaller scales are more important for detail)
    avg_scale = scales.mean(dim=1)
    scale_importance = 1.0 / (avg_scale + 1e-6)
    
    # Combined importance score
    importance = importance_combined * scale_importance
    
    # Keep top 75% to preserve more transitions
    target_count = int(N_original * 0.75)
    _, top_indices = torch.topk(importance, target_count)
    
    filtered_splats = {}
    for key, tensor in splats.items():
        filtered_splats[key] = tensor[top_indices]
    
    N_filtered = filtered_splats["means"].shape[0]
    reduction = (N_original - N_filtered) / N_original
    print(f"\n=== Opacity-Preserving Filtering Results ===")
    print(f"Original: {N_original:,} Gaussians")
    print(f"Filtered: {N_filtered:,} Gaussians")
    print(f"Reduction: {reduction:.1%}")
    print("Preserved opacity transitions for better visual quality")
    print("=" * 42)
    return filtered_splats


def _filter_for_real_estate_viewing(splats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Filtering optimized for real estate virtual tours."""
    N_original = splats["means"].shape[0]
    
    opacities = torch.sigmoid(splats["opacities"]).squeeze()
    scales = torch.exp(splats["scales"])
    
    # For real estate: keep medium-high opacity and reasonable scales
    opacity_mask = opacities > 0.015  # Lower threshold to preserve transitions
    scale_mask = scales.max(dim=1)[0] < 2.0  # Remove only very large artifacts
    
    # Combine filters
    final_mask = opacity_mask & scale_mask
    
    filtered_splats = {}
    for key, tensor in splats.items():
        filtered_splats[key] = tensor[final_mask]
    
    N_filtered = filtered_splats["means"].shape[0]
    reduction = (N_original - N_filtered) / N_original
    print(f"\n=== Real Estate Filtering Results ===")
    print(f"Original: {N_original:,} Gaussians")
    print(f"Filtered: {N_filtered:,} Gaussians")
    print(f"Reduction: {reduction:.1%}")
    print("=" * 35)
    return filtered_splats


def _filter_splats_for_export(splats: dict[str, torch.Tensor], 
                             min_opacity: float = 0.005,
                             max_scale: float = 0.1) -> dict[str, torch.Tensor]:
    """Custom filtering with user-specified thresholds."""
    N_original = splats["means"].shape[0]
    
    # Opacity filtering - remove very transparent Gaussians
    opacities = torch.sigmoid(splats["opacities"]).squeeze()
    opacity_mask = opacities > min_opacity
    
    # Scale filtering - remove very large Gaussians that might be artifacts
    scales = torch.exp(splats["scales"])
    max_scale_per_gaussian = scales.max(dim=1)[0]
    scale_mask = max_scale_per_gaussian < max_scale
    
    # Combine filters
    final_mask = opacity_mask & scale_mask
    
    # Apply filtering
    filtered_splats = {}
    for key, tensor in splats.items():
        filtered_splats[key] = tensor[final_mask]
    
    N_filtered = filtered_splats["means"].shape[0]
    reduction = (N_original - N_filtered) / N_original
    print(f"\n=== Custom Filtering Results ===")
    print(f"Original: {N_original:,} Gaussians")
    print(f"Filtered: {N_filtered:,} Gaussians")
    print(f"Reduction: {reduction:.1%}")
    print(f"Opacity threshold: {min_opacity}")
    print(f"Scale threshold: {max_scale}")
    print("=" * 30)
    
    return filtered_splats


def _filter_smart_lod(splats: dict[str, torch.Tensor], target_percentage: float = 0.4) -> dict[str, torch.Tensor]:
    """Smart level-of-detail filtering for web viewing."""
    N_original = splats["means"].shape[0]
    
    # Calculate importance score for each Gaussian
    opacities = torch.sigmoid(splats["opacities"]).squeeze()
    scales = torch.exp(splats["scales"])
    avg_scale = scales.mean(dim=1)  # Average scale per Gaussian
    
    # Importance = opacity^2 * (1 / average_scale) 
    # High opacity + small scale = more important for detail
    importance = (opacities ** 2) * (1.0 / (avg_scale + 1e-6))
    
    # Keep top percentage of most important Gaussians
    target_count = int(N_original * target_percentage)
    _, top_indices = torch.topk(importance, target_count)
    
    filtered_splats = {}
    for key, tensor in splats.items():
        filtered_splats[key] = tensor[top_indices]
    
    N_filtered = filtered_splats["means"].shape[0]
    print(f"\n=== Smart LOD Filtering Results ===")
    print(f"Original: {N_original:,} Gaussians")
    print(f"Filtered: {N_filtered:,} Gaussians")
    print(f"Kept: {target_percentage:.1%} most important")
    print("=" * 32)
    return filtered_splats


def _write_binary_ply_compatible(out_path: Path, splats: dict[str, torch.Tensor]) -> None:
    """Write PLY in the EXACT format that matches your working model."""
    N = splats["means"].shape[0]
    print(f"\n=== Converting {N:,} Gaussians to Compatible PLY ===")
    
    # Positions
    xyz = splats["means"].cpu().numpy().astype("<f4")  # (N,3)
    
    # CRITICAL: Use RAW scales (don't apply exp!) - keep in log space
    scales = splats["scales"].cpu().numpy().astype("<f4")  # (N,3) - keep in log space!
    
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
        scales,        # scale_0, scale_1, scale_2 (3)
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
    
    # Filtering options
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
    
    args = parser.parse_args(argv)

    splats = _load_splats(args.ckpt)
    _print_splat_info(splats)
    
    # Apply filtering based on selected option
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
    _write_binary_ply_compatible(args.output, splats)


if __name__ == "__main__":
    main()
