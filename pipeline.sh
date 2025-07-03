#!/usr/bin/env bash
# -------------------------------------------------------------
# Difix3D end-to-end pipeline
# 1. One-shot clean images
# 2. Prepare Nerfstudio dataset (COLMAP processing)
# 3. Train Gaussian-Splat model
#
# Usage: ./pipeline.sh [extra args forwarded to training]
# -------------------------------------------------------------
set -euo pipefail
START_TS=$(date +%s)
export START_TS

# User-adjustable paths for checkpoint conversion
NERFSTUDIO_CKPT=${NERFSTUDIO_CKPT:-"/path/to/nerfstudio/step-000029999.ckpt"}
GSPLAT_CKPT=${GSPLAT_CKPT:-"converted_checkpoints/gsplat_model.pt"}

# Step 1 ───────────────────────────────────────────────────────

echo "[1/4] Preparing Nerfstudio dataset (COLMAP)…"
./process_dataset.sh ~/datasets/colmap_workspace/images ~/datasets/colmap_processed 0

# Forward any arguments to the training launcher so you can
# override default flags, e.g. ./pipeline.sh --vis viewer

echo "[2/4] Training Splatfacto model… (logging to train.log)"
./run_difix3d_train.sh "$@" > train.log 2>&1 &
TRAIN_PID=$!
echo "Training running in background (PID $TRAIN_PID). Follow progress with: tail -100f train.log"
# Spawn a timer that waits for training to finish and logs total duration
(
  wait $TRAIN_PID
  END_TS=$(date +%s)
  TOTAL=$(( END_TS - START_TS ))
  printf "\n[PIPELINE] Training completed in %d hr %d min %d sec (total)\n" $((TOTAL/3600)) $((TOTAL%3600/60)) $((TOTAL%60)) | tee -a train.log
)&
ELAPSED=$(( $(date +%s) - START_TS ))
printf "\n[PIPELINE] Dataset prep + base training completed in %d min %d sec\n" $((ELAPSED/60)) $((ELAPSED%60)) | tee -a train.log

echo "[3/4] Converting Nerfstudio checkpoint to gsplat format…"
python convert_checkpoint.py --nerfstudio_ckpt "$NERFSTUDIO_CKPT" --output_gsplat_ckpt "$GSPLAT_CKPT"

echo "[4/4] Fine-tuning GS model… (logging to finetune.log)"
./run_difix3d_gsplat_finetune.sh > finetune.log 2>&1 &
FINETUNE_PID=$!
echo "Fine-tuning running in background (PID $FINETUNE_PID). Follow progress with: tail -100f finetune.log"
# Optional: wait for fine-tuning to end and time it
(
  wait $FINETUNE_PID
  END_TS=$(date +%s)
  TOTAL=$(( END_TS - START_TS ))
  printf "\n[PIPELINE] Fine-tuning completed in %d hr %d min %d sec (total elapsed)\n" $((TOTAL/3600)) $((TOTAL%3600/60)) $((TOTAL%60)) | tee -a finetune.log
)&

echo "✓ Pipeline finished (fine-tuning is now running in background). Check outputs/ for results.
# Tip: override NERFSTUDIO_CKPT and GSPLAT_CKPT env vars if you
#      want to convert a different checkpoint:
# NERFSTUDIO_CKPT=...</path/to/ckpt> GSPLAT_CKPT=out.pt ./pipeline.sh"
