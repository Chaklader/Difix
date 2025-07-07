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

# ------------------------------------------------------------
# Training schedule preset selector
# 1 → quick run to 35 k (+100 bake-in)
# 2 → short run to 40 k (+100 bake-in)
# 3 → medium run to 50 k (+100 bake-in)
# 4 → full run to 60 k (default comprehensive schedule)
# ------------------------------------------------------------
# Select schedule preset here (1=35k, 2=40k, 3=50k, 4=60k)
SCHEDULE_PRESET=1

case "$SCHEDULE_PRESET" in
    1)
        # 35 k run (4 fix cycles, then 100-step bake-in)
        MAX_STEPS=35100
        EVAL_STEPS="10000 20000 30000"
        SAVE_STEPS="5000 10000 20000 30000 35100"
        FIX_STEPS="5000 10000 20000 30000"
        ;;
    2)
        # 40 k run (fix once, then bake-in)
        MAX_STEPS=40100
        EVAL_STEPS="10000 20000 30000 35000 40000"
        SAVE_STEPS="10000 20000 30000 35000 40000 40100"
        FIX_STEPS="3000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000"
        ;;
    3)
        # 50 k run (longer training, +100-step bake-in)
        MAX_STEPS=50100
        EVAL_STEPS="10000 20000 30000 35000 40000 45000 50000"
        SAVE_STEPS="10000 20000 30000 35000 40000 45000 50000 50100"
        FIX_STEPS="3000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000"
        ;;
    4|*)
        # 60 k run (original comprehensive schedule)
        MAX_STEPS=60000
        EVAL_STEPS="10000 20000 30000 35000 40000 45000 50000 55000 60000"
        SAVE_STEPS="10000 20000 30000 40000 45000 50000 55000 60000"
        FIX_STEPS="3000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000 52000 54000 56000 58000 60000"
        ;;
esac

# per-GPU batch size
BATCH_SIZE=2

