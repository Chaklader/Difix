#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Difix3D + Nerfstudio finetuning launcher
# -----------------------------------------------------------------------------
# 1.  Set SCENE_ID to the folder name that contains the processed Nerfstudio
#     dataset (colmap/ + images_2/ images_4/ …)
# 2.  Set DATA_DIR and CKPT_DIR to where you keep the processed data and the
#     pretrained NeRF checkpoints, respectively.
# -----------------------------------------------------------------------------

set -euo pipefail

# ------------------------------ user inputs ----------------------------------
SCENE_ID="difix3d_2025_07_22"
DATA_FACTOR=1                                         # 1 = full-res, 2/4/8 downsamples
DATA="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="step-000029999_aligned.ckpt"  
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/${SCENE_ID}"

# ------------------------------ sanity checks -------------------------------
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATA}" ]]; then
  echo "[ERROR] Dataset folder not found: ${DATA}" >&2
  exit 1
fi

# ------------------------------ launch ---------------------------------------
# Set CUDA memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# SCENE_ID=032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7
# DATA=DATA_DIR/${SCENE_ID}
# DATA_FACTOR=4
# CKPT_PATH=CKPR_DIR/${SCENE_ID}/nerfacto/nerfstudio_models/step-000029999.ckpt # Path to the pretrained checkpoint file
# OUTPUT_DIR=outputs/difix3d/nerfacto/${SCENE_ID}

# CUDA_VISIBLE_DEVICES=0 ns-train difix3d \
#     --data ${DATA} \
#     --pipeline.model.appearance-embed-dim 0 \
#     --pipeline.model.camera-optimizer.mode off \
#     --save_only_latest_checkpoint False \
#     --vis viewer \
#     --output_dir ${OUTPUT_DIR} \
#     --experiment_name ${SCENE_ID} \
#     --timestamp '' \
#     --load-checkpoint ${CKPT_PATH} \
#     --max_num_iterations 30000 \
#     --steps_per_eval_all_images 0 \
#     --steps_per_eval_batch 0 \
#     --steps_per_eval_image 0 \
#     --steps_per_save 2000 \
#     --viewer.quit-on-train-completion True \
#     nerfstudio-data \
#     --orientation-method none \
#     --center_method none \
#     --auto-scale-poses False \
#     --downscale_factor ${DATA_FACTOR} \
#     --eval_mode filename \

    # ------------------------------ launch ---------------------------------------
CUDA_VISIBLE_DEVICES=0 \
ns-train difix3d \
    --data "${DATA}" \
    --pipeline.model.appearance-embed-dim 0 \
    --pipeline.model.camera-optimizer.mode off \
    --save_only_latest_checkpoint False \
    --vis viewer \
    --output_dir "${OUTPUT_DIR}" \
    --experiment_name "${SCENE_ID}" \
    --timestamp '' \
    --load-checkpoint "${CKPT_PATH}" \
    --max_num_iterations 250 \
    --steps_per_eval_all_images 0 \
    --steps_per_eval_batch 0 \
    --steps_per_eval_image 0 \
    --steps_per_save 200 \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --orientation-method none \
    --center_method none \
    --auto-scale-poses False \
    --downscale_factor "${DATA_FACTOR}" \
    --eval_mode fraction \
    --train_split_fraction 0.9

echo "Difix3D training completed for 250 steps"