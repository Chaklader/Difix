#!/usr/bin/env bash
# Longer fine-tune run: start from NeRF splats @ 29 999 and train ~4 000 steps with periodic evals.
# Outputs will be written to /mnt/nvme0n1/azureuser/finetune/difix_debug

SCENE_ID="difix_debug"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="/mnt/nvme0n1/azureuser/finetune/difix_debug/ckpts/ckpt_40099_rank0.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/${SCENE_ID}"

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
    --data_dir "${DATA_DIR}" \
    --data_factor 1 \
    --result_dir "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test_every 2 \
    --max_steps 60100 \
    --eval_steps 42000 44000 46000 48000 50000 52000 54000 56000 58000 60000 \
    --fix_steps 99999 \
    --ckpt "${CKPT_PATH}" \
    --batch_size 4
