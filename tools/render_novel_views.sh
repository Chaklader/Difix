#!/bin/bash

# Novel view rendering script for Difix3D
# Generates synthetic renders from spiral/interpolated camera trajectories

SCENE_ID="difix3d_novel_renders"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="NeRF.pt"  # Your aligned checkpoint
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/novel_views/${SCENE_ID}"

echo "Rendering novel views from checkpoint: ${CKPT_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

# Render spiral trajectory (good for 360 scenes)
CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --ckpt "${CKPT_PATH}" \
    --data_dir "${DATA_DIR}" \
    --no-normalize-world-space \
    --result_dir "${OUTPUT_DIR}" \
    --max_steps 3 \
    --eval_steps 2 \
    --save_steps 999999 \
    --fix_steps 2 \
    --render_traj_path spiral

echo "Novel view rendering completed"
echo "Rendered images saved to: ${OUTPUT_DIR}/renders/"
echo "Next: Use Difix diffusion pipeline to clean these synthetic renders"
