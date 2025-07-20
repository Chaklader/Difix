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
SCENE_ID="difix3d_2025_07_21"
DATA_FACTOR=1                                         # 1 = full-res, 2/4/8 downsamples
DATA="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="outputs/difix3d_20250717_164447/splatfacto/2025-07-17_164454/nerfstudio_models/step-000029999.ckpt"  
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
CUDA_VISIBLE_DEVICES=0 \
ns-train difix3d \
    --data "${DATA}" \
    --pipeline.model.appearance-embed-dim 0 \
    --pipeline.model.camera-optimizer.mode off \
    --save_only_latest_checkpoint False \
    --vis viewer \
    --output_dir "${OUTPUT_DIR}" \
    --experiment_name "${SCENE_ID}" \
    --load-checkpoint "${CKPT_PATH}" \
    --max_num_iterations 30000 \
    --steps_per_eval_all_images 0 \
    --steps_per_eval_batch 0 \
    --steps_per_eval_image 0 \
    --steps_per_save 2000 \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --orientation-method none \
    --center_method none \
    --auto-scale-poses False \
    --downscale_factor "${DATA_FACTOR}" \
    --eval_mode fraction \
    --train_split_fraction 0.9

echo "Difix3D training completed for 30,000 steps"