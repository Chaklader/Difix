#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

DATA_DIR="/home/azureuser/datasets/colmap_processed"   
RUN_NAME="splatfacto_sean_set3_$(date +%Y%m%d_%H%M%S)"

ns-train splatfacto \
  --machine.num-devices 1 \
  --vis wandb \
  --viewer.quit-on-train-completion True \
  --max_num_iterations 30000 \
  --pipeline.datamanager.images-on-gpu True \
  --pipeline.datamanager.train-cameras-sampling-strategy fps \
  --pipeline.model.use_bilateral_grid True \
  --experiment-name "${RUN_NAME}" \
  --project-name DC-DEV \
  nerfstudio-data \
  --data "${DATA_DIR}" \
  --downscale_factor 1 \
  --train-split-fraction 0.9

