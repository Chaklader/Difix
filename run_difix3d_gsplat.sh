#!/usr/bin/env bash
# Quick debug run: load NeRF.pt at step 29999, run 20 SGD steps with one eval, no fixer.
# Outputs will be written to /mnt/nvme0n1/azureuser/finetune/difix_debug

SCENE_ID="difix_debug"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="NeRF.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/${SCENE_ID}"

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
    --data_dir "${DATA_DIR}" \
    --data_factor 1 \
    --result_dir "${OUTPUT_DIR}" \
    --test_every 8 \
    --max_steps 30020 \
    --eval_steps 30000 \
    --fix_steps 99999 \
    --ckpt "${CKPT_PATH}"
