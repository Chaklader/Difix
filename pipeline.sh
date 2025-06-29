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

# Step 1 ───────────────────────────────────────────────────────
echo "[1/3] Cleaning images with Difix…"
./one_shot_clean.sh

echo "[2/3] Preparing Nerfstudio dataset (COLMAP)…"
./process_dataset.sh ~/datasets/colmap_workspace/images_clean ~/datasets/colmap_processed 0

# Forward any arguments to the training launcher so you can
# override default flags, e.g. ./pipeline.sh --vis viewer

echo "[3/3] Training Splatfacto model… (logging to train.log)"
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
printf "\n[PIPELINE] Pre-training stages completed in %d min %d sec\n" $((ELAPSED/60)) $((ELAPSED%60)) | tee -a train.log

echo "✓ Pipeline finished. Check outputs/ for results."
