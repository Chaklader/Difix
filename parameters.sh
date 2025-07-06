#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# parameters.sh
# ------------------------------------------------------------------------------
# Central place to set training schedule hyper-parameters for Difix3D GSplat runs.
# Modify the values below to experiment without touching library code.
# All values are integers; step lists are written as space-separated strings.
# ------------------------------------------------------------------------------

# Total optimisation steps
# MAX_STEPS=60000

# # Global steps at which to run validation rendering (eval())
# EVAL_STEPS="10000 20000 30000 35000 40000 45000 50000 55000 60000"

# # Global steps at which to write model checkpoints
# SAVE_STEPS="10000 20000 30000 40000 45000 50000 55000 60000"

# # Global steps at which to run the image fixer / novel render pipeline
# FIX_STEPS="3000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000 52000 54000 56000 58000 60000"


MAX_STEPS=40000

# per-GPU batch size
BATCH_SIZE=2

# do an eval, save ckpt, and run fixer only once
EVAL_STEPS="40000"
SAVE_STEPS="35000 40000"
FIX_STEPS="40000"
