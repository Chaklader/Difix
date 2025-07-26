#!/usr/bin/env bash
# Splatfacto fine-tune launcher initialised from Nerfacto point-cloud
# Usage:
#   chmod +x run_test.sh
#   ./run_test.sh > splat.log 2>&1 &
# -----------------------------------------------------------------------------
set -euo pipefail

# Allow local src/ to be importable if running outside editable install
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# -----------------------------------------------------------------------------
DATA_DIR="/mnt/nvme0n1/azureuser/datasets/colmap_processed"
INIT_PLY="$(pwd)/assets/NeRF.ply"   # point-cloud exported from Nerfacto
RUN_NAME="splatfacto_from_nerfacto_arefe_$(date +%Y%m%d_%H%M%S)"

    
ns-train splatfacto \
  --machine.num-devices 1 \
  --vis tensorboard \
  --viewer.quit-on-train-completion True \
  --data "${DATA_DIR}" \
  --pipeline.model.init-points "${INIT_PLY}" \
  --pipeline.model.densify-grad-threshold 0.0002 \
  --pipeline.model.densification-interval 100 \
  --pipeline.model.opacity-reset-interval 3000 \
  --experiment-name "${RUN_NAME}" \
  --project-name DC-DEV \
  --train-split-fraction 0.9 \
  --max-num-iterations 30000

echo "Splatfacto fine-tune completed – ${RUN_NAME}"
