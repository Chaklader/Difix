#!/usr/bin/env bash
# ============================================================
# Difix3D – RESUME script
# ------------------------------------------------------------
# Run only the remaining phases (B: ½-res, C: full-res)
# starting from an existing checkpoint, typically one written
# at the end of Phase A (≈45 k) or later.
#
# Usage:
#   bash run_difix.sh /path/to/ckpt_54999_rank0.pt
#
# Logs are sent to stdout; pipe/tee/ nohup as desired.
# ============================================================
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "USAGE: $0 <checkpoint.pt>" >&2
  exit 1
fi

CKPT_PATH="$1"
if [ ! -f "$CKPT_PATH" ]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

# ------------------------
# Scene-specific variables
# ------------------------
SCENE_ID="difix_sprint"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/${SCENE_ID}"

# Limit DataLoader workers to reduce RAM usage
export DIFIX_NUM_WORKERS=2
mkdir -p "$OUTPUT_DIR"

# Helper: current step stored in checkpoint
ckpt_step() {
  python - <<'PY' "$1"
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu").get("step", 0))
PY
}

CKPT_STEP=$(ckpt_step "$CKPT_PATH")

echo "[INFO] Resuming from $CKPT_PATH  (step $CKPT_STEP)"

############################
# Phase B – half-resolution
############################
if (( CKPT_STEP < 70000 )); then
  MAX2=$(( CKPT_STEP + 25000 ))
  echo "[Phase 2] ½-res refinement → up to step $MAX2"
  CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
    --data_dir "$DATA_DIR" \
    --result_dir "$OUTPUT_DIR" \
    --ckpt "$CKPT_PATH" \
    --no-normalize-world-space \
    --data_factor 2 \
    --batch_size 12 \
    --test_every 2 \
    --max_steps "$MAX2" \
    --eval_steps "$MAX2" \
    --disable_viewer
  CKPT_PATH=$(ls -1t "$OUTPUT_DIR"/ckpts/ckpt_*_rank0.pt | head -n1)
  CKPT_STEP=$(ckpt_step "$CKPT_PATH")
else
  echo "[Phase 2] already completed (current step $CKPT_STEP)"
fi

############################
# Phase C – full-resolution
############################
if (( CKPT_STEP < 90000 )); then
  MAX3=$(( CKPT_STEP + 20000 ))
  echo "[Phase 3] Full-res polish → up to step $MAX3"
  CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
    --data_dir "$DATA_DIR" \
    --result_dir "$OUTPUT_DIR" \
    --ckpt "$CKPT_PATH" \
    --no-normalize-world-space \
    --data_factor 1 \
    --batch_size 8 \
    --test_every 2 \
    --max_steps "$MAX3" \
    --eval_steps "$MAX3" \
    --fix_steps "$MAX3" \
    --disable_viewer
else
  echo "[Phase 3] already completed (current step $CKPT_STEP)"
fi

echo "===== Difix3D resume run finished – final checkpoint: $(ls -1t "$OUTPUT_DIR"/ckpts/ckpt_*_rank0.pt | head -n1) ====="
