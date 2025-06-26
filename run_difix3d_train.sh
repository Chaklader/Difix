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


# -----------------------------------------------------------------------------
ns-train difix3d \
  --machine.num-devices 1 \
  --vis tensorboard \
  --max_num_iterations 30000 \
  --pipeline.model.appearance-embed-dim 0 \
  --pipeline.model.camera-optimizer.mode off \
  --pipeline.datamanager.train-num-images-to-sample-from -1 \
  --pipeline.datamanager.eval-num-images-to-sample-from -1 \
  --pipeline.datamanager.train-num-rays-per-batch 4096 \
  --pipeline.datamanager.patch-size 1 \
  --experiment-name "${RUN_NAME}" \
  --project-name difix3d \
  nerfstudio-data \
  --data "${DATA_DIR}" \
  --downscale_factor 2 \
  --train-split-fraction 0.9 \
  --orientation-method none \
  --center_method none \
  --auto-scale-poses False

# Export Gaussian splat after training finishes
CFG=$(find ./outputs -type f -name config.yml | grep "${RUN_NAME}" | head -n 1 || true)
if [[ -n "${CFG}" ]]; then
  ns-export gaussian-splat \
    --load-config "${CFG}" \
    --output-dir "./exports/${RUN_NAME}"
fi
