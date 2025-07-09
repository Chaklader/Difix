#!/usr/bin/env python3
"""
Convert Nerfstudio Splatfacto checkpoint to Difix3D gsplat format (fixed).
Scales are stored in log-space and opacities in logit-space as expected by Difix.
Usage:
    python convert_checkpoint_2.py \
        --nerfstudio_ckpt path/to/step-000029999.ckpt \
        --output_gsplat_ckpt converted.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

EPS = 1e-4  # numerical safety


def convert_nerfstudio_to_gsplat(nerfstudio_ckpt_path: Path, output_path: Path) -> None:
    """Convert a Nerfstudio Splatfacto checkpoint into a Difix3D-compatible gsplat checkpoint."""

    print(f"Loading Nerfstudio checkpoint: {nerfstudio_ckpt_path}")
    ckpt = torch.load(nerfstudio_ckpt_path, map_location="cpu", weights_only=False)

    pipeline_state = ckpt["pipeline"]

    # Extract Gaussian parameters in linear space
    means = pipeline_state["_model.gauss_params.means"]
    scales = pipeline_state["_model.gauss_params.scales"]  # linear σ
    quats = pipeline_state["_model.gauss_params.quats"]
    features_dc = pipeline_state["_model.gauss_params.features_dc"]  # [N,3]
    features_rest = pipeline_state["_model.gauss_params.features_rest"]  # [N,15,3]
    opacities = pipeline_state["_model.gauss_params.opacities"]  # Already in logit space

    # Convert to Difix representation
    scales_log = torch.log(torch.clamp(scales, min=EPS))  # log σ
    # Opacities in Nesrfstudio checkpoints are already parameterised as logit (unbounded).
    # Keep them unchanged except for squeezing the trailing dim.
    opacities_logit = opacities.squeeze(-1)

    gsplat_ckpt = {
        "splats": {
            "means": means.clone(),
            "scales": scales_log.clone(),
            "quats": quats.clone(),
            "opacities": opacities_logit.clone(),
            "sh0": features_dc.clone().unsqueeze(1),  # [N,1,3]
            "shN": features_rest.clone(),
        },
        "step": ckpt.get("step", 29999),
        "global_step": ckpt.get("step", 29999),
        "metadata": {
            "source": "nerfstudio_splatfacto",
            "num_gaussians": means.shape[0],
            "converted_from": str(nerfstudio_ckpt_path),
        },
    }

    print(f"Saving gsplat checkpoint to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gsplat_ckpt, output_path)
    print("Conversion completed ✔")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Nerfstudio checkpoint to gsplat format (fixed)")
    parser.add_argument("--nerfstudio_ckpt", required=True, type=Path, help="Path to Nerfstudio .ckpt file")
    parser.add_argument("--output_gsplat_ckpt", required=True, type=Path, help="Destination .pt file")
    args = parser.parse_args()

    if not args.nerfstudio_ckpt.exists():
        parser.error(f"Checkpoint not found: {args.nerfstudio_ckpt}")

    convert_nerfstudio_to_gsplat(args.nerfstudio_ckpt, args.output_gsplat_ckpt)


if __name__ == "__main__":
    main()
