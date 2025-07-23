#!/usr/bin/env bash
# Nerfacto training launcher (30k iterations)
# Usage:
#   chmod +x riun_nerfacto.sh
#   ./run_nerfacto.sh > nerfacto.log 2>&1 &
# -----------------------------------------------------------------------------
set -euo pipefail

# Allow local src/ to be importable if running outside editable install
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# -----------------------------------------------------------------------------
DATA_DIR="/home/azureuser/datasets/colmap_processed"   # processed dataset
RUN_NAME="nerfacto_$(date +%Y%m%d_%H%M%S)"

# -----------------------------------------------------------------------------
# Launch Nerfacto training
# Notes:
#   • 30 000 iterations as requested
#   • WandB visualisation enabled
#   • Project name fixed to DC-DEV
#   • 95% of images used for training, 5% for validation/test
#   • Viewer closed automatically on completion
# -----------------------------------------------------------------------------
ns-train nerfacto \
  --machine.num-devices 1 \
  --vis wandb \
  --viewer.quit-on-train-completion True \
  --max-num-iterations 30000 \
  --experiment-name "${RUN_NAME}" \
  --project-name DC-DEV \
  nerfstudio-data \
  --data "${DATA_DIR}" \
  --downscale_factor 1 \
  --train-split-fraction 0.9
