#!/usr/bin/env bash
# Difix3D training launcher (Nerfstudio plug-in)
# Usage: Adjust DATA_DIR if your COLMAP workspace is elsewhere, then run
#   chmod +x run_difix3d_train.sh
#   ./run_difix3d_train.sh > train.log 2>&1 &
# -----------------------------------------------------------------------------
set -euo pipefail

# Make local src/ importable when running outside editable install
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# -----------------------------------------------------------------------------
DATA_DIR="/home/azureuser/datasets/colmap_processed"   # <- processed dataset
RUN_NAME="difix3d_$(date +%Y%m%d_%H%M%S)"

# downscale_factor selects which image resolution folder to load:
#   1 → images/ (original resolution)
#   2 → images_2/ (half resolution)
#   4 → images_4/ (quarter resolution)
# Ensure this value equals 2^NUM_DOWNSCALES used in process_dataset.sh.
# Using --downscale_factor 2 means Nerfstudio will read from images_2/,
# i.e. every dimension is halved (4032×3024 → 2016×1512).


# Ray batch size guidance (VRAM vs speed):
#   24 GB GPU with downscale_factor 2 → 2 048 – 8 192 rays/batch
#   downscale_factor 4 (smaller imgs) → 8 192 – 16 384 rays/batch
# -----------------------------------------------------------------------------
# Train a Gaussian-Splat model (Splatfacto) on the one-shot-cleaned images
# Note: one_shot_clean.sh must have been run beforehand so ${DATA_DIR}
# already points at the pre-cleaned, COLMAP-processed dataset.

# ns-train splatfacto \
# 		--pipeline.datamanager.images-on-gpu True \
# 		--pipeline.model.use_bilateral_grid True \
# 		--pipeline.datamanager.train-cameras-sampling-strategy fps \
#         --data ./pd \
#         --machine.num-devices 1 \
# 		--vis $vis \
#         --viewer.quit-on-train-completion True \
#         --experiment-name $N3D_JOBID \
# 		--project-name $proj_name \
#         nerfstudio-data \
#         --train-split-fraction $splitfrac

ns-train splatfacto \
  --machine.num-devices 1 \
  --vis tensorboard \
  --viewer.quit-on-train-completion True \
  --max-num-iterations 30000 \
  --pipeline.datamanager.images-on-gpu True \
  --pipeline.datamanager.train-cameras-sampling-strategy fps \
  --pipeline.model.use_bilateral_grid True \
  --pipeline.model.densify-grad-thresh 0.0002 \
  --pipeline.model.refine-every 50 \
  --pipeline.model.opacity-loss-weight 0.0005 \
  --optimizer.lr 0.002 \
  --pipeline.model.sched-decay-steps 20000 \
  --pipeline.model.camera-optimizer.mode off \
  --experiment-name "${RUN_NAME}" \
  --project-name splatfacto \
  nerfstudio-data \
  --data "${DATA_DIR}" \
  --downscale_factor 1 \
  --train-split-fraction 0.9 \
  --orientation-method none \
  --center_method none \
  --auto-scale-poses False


# After training completes you can export the Gaussian-Splat model with:
#   ns-export gaussian-splat \
#     --load-config "$(find ./outputs -type f -name config.yml | grep "${RUN_NAME}" | head -n 1)" \
#     --output-dir "exports/${RUN_NAME}"
# (export_pointcloud.sh is still available if you merely want a point cloud.)
