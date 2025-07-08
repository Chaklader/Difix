#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# parameters.sh
# ------------------------------------------------------------------------------
# Central place to set training schedule hyper-parameters for Difix3D GSplat runs.
# Modify the values below to experiment without touching library code.
# All values are integers; step lists are written as space-separated strings.
# ------------------------------------------------------------------------------
# Universal step lists — always passed to the trainer
EVAL_STEPS="10000 20000 30000 35000 40000 45000 50000 55000 60000"
SAVE_STEPS="10000 20000 30000 40000 45000 50000 55000 60000"
FIX_STEPS="3000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000 52000 54000 56000 58000 60000"
# ------------------------------------------------------------------------------

# Default maximum training steps (overridden by presets below)
MAX_STEPS=60000

# per-GPU batch size
BATCH_SIZE=2

# ------------------------------------------------------------
# Training schedule preset selector
# 1 → quick run to 35 k (+100 bake-in)
# 2 → short run to 40 k (+100 bake-in)
# 3 → medium run to 50 k (+100 bake-in)
# 4 → full run to 60 k (default comprehensive schedule)
# ------------------------------------------------------------
# Select schedule preset here (1=35k, 2=40k, 3=50k, 4=60k)
SCHEDULE_PRESET=1

# Select training length (k indicates thousands of base SGD steps; +100-step bake-in is implicit)
case "$SCHEDULE_PRESET" in
  1) MAX_STEPS=32100 ;;  # ~32k quick run
  2) MAX_STEPS=34100 ;;  # ~34k short run
  3) MAX_STEPS=40100 ;;  # 40k medium run
  4) MAX_STEPS=50100 ;;  # 50k long run
  *) MAX_STEPS=60000 ;;  # 60k full run (default)
esac

