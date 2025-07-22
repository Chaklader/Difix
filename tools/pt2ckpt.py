#!/usr/bin/env python3
"""
Merge an aligned Difix3D / gsplat ``.pt`` file back into a full Nerfstudio
Lightning checkpoint ``.ckpt`` so it can be used by ``ns-train difix3d``.

Typical workflow:
1. Train a Splatfacto run (produces ``step-XXXXX.ckpt``).
2. Convert that checkpoint to a minimal ``.pt`` (splats-only) or extract the
   splats via ``tools/ckpt2pt.py``.
3. Align the splats to COLMAP metric space with ``tools/align_checkpoint.py``
   (produces ``NeRF.pt``).
4. Merge the aligned splats back into the original Lightning checkpoint:

   ```bash
   python tools/pt2ckpt.py \
       --orig_ckpt step-000029999.ckpt \
       --aligned_pt NeRF.pt \
       --output_ckpt step-000029999_aligned.ckpt
   ```

The resulting ``*_aligned.ckpt`` contains *all* original training state plus
aligned Gaussian tensors under the expected flat keys so that
``run_difix3d.sh`` can load it with ``--load-checkpoint``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge_pt_into_ckpt(orig_ckpt: Path, aligned_pt: Path, output_ckpt: Path) -> None:
    """Replace Gaussian tensors in *orig_ckpt* with those from *aligned_pt* and
    save to *output_ckpt*.
    """

    print(f"Loading original checkpoint: {orig_ckpt}")
    ck = torch.load(orig_ckpt, map_location="cpu")

    print(f"Loading aligned splats: {aligned_pt}")
    aligned = torch.load(aligned_pt, map_location="cpu")
    if "splats" not in aligned:
        raise KeyError("Aligned .pt file must contain a top-level 'splats' key")
    spl = aligned["splats"]

    # Nerfstudio Lightning ckpts store tensors under flat keys inside the
    # 'pipeline' dict; overwrite them with the aligned versions.
    pipe = ck["pipeline"]

    pipe["_model.gauss_params.means"] = spl["means"]
    pipe["_model.gauss_params.scales"] = spl["scales"]
    pipe["_model.gauss_params.quats"] = spl["quats"]

    # Ensure opacities have shape [N,1]
    opacities = spl["opacities"]
    if opacities.ndim == 1:
        opacities = opacities.unsqueeze(-1)
    pipe["_model.gauss_params.opacities"] = opacities

    # SH coefficients
    sh0 = spl["sh0"]
    if sh0.ndim == 3:  # [N,1,3] -> [N,3]
        sh0 = sh0.squeeze(1)
    pipe["_model.gauss_params.features_dc"] = sh0
    pipe["_model.gauss_params.features_rest"] = spl["shN"]

    print(f"Saving merged checkpoint to: {output_ckpt}")
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, output_ckpt)
    print("Done ✔")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge aligned splats (.pt) back into a Lightning checkpoint (.ckpt)"
    )
    parser.add_argument("--orig_ckpt", type=Path, required=True, help="Original .ckpt path")
    parser.add_argument("--aligned_pt", type=Path, required=True, help="Aligned splats .pt path")
    parser.add_argument("--output_ckpt", type=Path, required=True, help="Destination .ckpt path")
    args = parser.parse_args()

    if not args.orig_ckpt.exists():
        parser.error(f"Original checkpoint not found: {args.orig_ckpt}")
    if not args.aligned_pt.exists():
        parser.error(f"Aligned .pt file not found: {args.aligned_pt}")

    merge_pt_into_ckpt(args.orig_ckpt, args.aligned_pt, args.output_ckpt)


if __name__ == "__main__":
    main()
