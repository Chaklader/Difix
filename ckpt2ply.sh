#!/usr/bin/env bash
# Export a point cloud PLY from a finished Difix3D/Nerfacto run.
# Usage: ./export_pointcloud.sh <RUN_NAME>

set -e

if [[ -z "$1" ]]; then
  echo "Usage: $0 <RUN_NAME>" >&2
  exit 1
fi

RUN_NAME="$1"
CFG=$(find ./outputs -type f -name config.yml | grep "${RUN_NAME}" | head -n 1 || true)
if [[ -z "${CFG}" ]]; then
  echo "Could not find config for run ${RUN_NAME}" >&2
  exit 1
fi

OUT_DIR="./exports/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

# usage: ns-export [-h]
#                  {pointcloud,tsdf,poisson,marching-cubes,cameras,gaussian-splat}

# ╭─ options ─────────────────────────────────────────────────────────────╮
# │ -h, --help        show this help message and exit                     │
# ╰───────────────────────────────────────────────────────────────────────╯
# ╭─ subcommands ─────────────────────────────────────────────────────────╮
# │ {pointcloud,tsdf,poisson,marching-cubes,cameras,gaussian-splat}       │
# │     pointcloud    Export NeRF as a point cloud.                       │
# │     tsdf          Export a mesh using TSDF processing.                │
# │     poisson       Export a mesh using poisson surface reconstruction. │
# │     marching-cubes                                                    │
# │                   Export a mesh using marching cubes.                 │
# │     cameras       Export camera poses to a .json file.                │
# │     gaussian-splat                                                    │
# │                   Export 3D Gaussian Splatting model to a .ply        │
# ╰───────────────────────────────────────────────────────────────────────╯

echo "Exporting point cloud to ${OUT_DIR}/pointcloud.ply"

# ```bash
# $ ns-export pointcloud --load-config outputs/abq/nerfacto/2023-05-31_152244/config.yml --output-dir exports/pcd/ --num-points 1000000 --remove-outliers True --normal-method model_output --normal-output-name normals --use-bounding-box True --bounding-box-min -1 -1 -1 --bounding-box-max 1 1 1
# ```

# And the error message:
# ```
# usage: ns-export [-h] {pointcloud,tsdf,poisson,marching-cubes,cameras}

# ns-export: error: unrecognized arguments: --normal-method model_output --normal-output-name normals
# ```

# The error indicates that the `--normal-method` and `--normal-output-name` arguments are not recognized in this version of ns-export.

# Example: export a point cloud for a *specific* checkpoint
#   • --normal-method open3d  – compute per-vertex normals via Open3D (install with `pip install open3d`)
#   • You can add `--step <N>` instead of supplying an explicit checkpoint path.
#   • Remove the flag or use `--normal-method none` to skip normals.
#
# This one-liner worked successfully on the A100 box:
# ns-export pointcloud --load-config outputs/difix3d_20250626_073554/difix3d/2025-06-26_073600/config.yml --output-dir exports/ --normal-method open3d
#  for EVAL 
# from repo root or with PYTHONPATH set
# ns-eval \
#   --load-config outputs/difix3d_20250626_073554/difix3d/config.yml \
#   --step 29999 \                    # omit for latest if your version supports --step
#   --output-path eval_results
# ns-eval --load-config outputs/difix3d_20250626_073554/difix3d/config.yml  --output-path eval_results

# ns-export gaussian-splat \
#   --load-config outputs/difix3d_20250627_174106/splatfacto/2025-06-27_174113/config.yml \
#   --output-dir exports/gaussian_splat/ \
#   --output-filename Nerf.ply

# PLY location on A100 box
# /home/azureuser/github/Difix3D/exports/gaussian_splat/Nerf.ply

# ns-eval \
#   --load-config outputs/difix3d_20250627_174106/splatfacto/2025-06-27_174113/config.yml \
#   --output-path eval_results/gaussian_model.json

# (difix) azureuser@PipelineGPU:~/github/Difix3D$ cat eval_results/gaussian_model.json 
# {
#   "experiment_name": "difix3d_20250627_174106",
#   "method_name": "splatfacto",
#   "checkpoint": "outputs/difix3d_20250627_174106/splatfacto/2025-06-27_174113/nerfstudio_models/step-000029999.ckpt",
#   "results": {
#     "psnr": 24.536195755004883,
#     "psnr_std": 2.76033091545105,
#     "ssim": 0.9524361491203308,
#     "ssim_std": 0.039308369159698486,
#     "lpips": 0.1788865178823471,
#     "lpips_std": 0.07015663385391235,
#     "num_rays_per_sec": 72083408.0,
#     "num_rays_per_sec_std": 10386467.0,
#     "fps": 5.915403842926025,
#     "fps_std": 0.8523479700088501
#   }


# ##
# Here are your model's evaluation metrics:

# ## Quality Metrics:

# **PSNR: 24.54 dB (±2.76)**
# - Measures pixel-level accuracy
# - 24.5 is decent, 25-30 is good, 30+ is excellent
# - Your model has moderate quality

# **SSIM: 0.952 (±0.039)**
# - Measures structural similarity (0-1, higher is better)
# - 0.95 is **very good** - maintains structure well
# - Close to ground truth visually

# **LPIPS: 0.179 (±0.070)**
# - Perceptual similarity (lower is better)
# - 0.179 is moderate - some perceptual differences
# - <0.1 is excellent, 0.1-0.2 is good

# ## Performance Metrics:

# **Rays/sec: 72M (±10M)**
# - Very fast ray processing
# - Good for real-time applications

# **FPS: 5.9 (±0.85)**
# - Rendering speed in frames per second
# - Depends on resolution and hardware

# ## Summary:
# Your model has **good visual quality** (high SSIM) with **moderate pixel accuracy** (PSNR). It renders **very fast** 
# (72M rays/sec). The high standard deviations suggest quality varies across different test views.

ns-export gaussian-splat \
  --load-config "${CFG}" \
  --output-dir "${OUT_DIR}" \
  --output-filename Nerf.ply \
  "$@"

echo "Done."
