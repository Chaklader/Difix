#!/usr/bin/env bash

# Run Difix3D fine-tuning on a Gaussian-Splat (splatfacto) checkpoint and export directory.
# Usage: ./run_difix3d_gsplat_finetune.sh <SCENE_ID> <DATASET_DIR> <CKPT_DIR> [DATA_FACTOR]
# Example:
#   ./run_difix3d_gsplat_finetune.sh \
#       032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7 \
#       /mnt/datasets \
#       /mnt/checkpoints 4
#
# Arguments
#   SCENE_ID      – Folder name identifying the scene (also used in output paths)
#   DATASET_DIR   – Root path that contains ${SCENE_ID}/gaussian_splat/
#   CKPT_DIR      – Root path that contains ${SCENE_ID}/ckpts/ckpt_29999_rank0.pt
#   DATA_FACTOR   – Optional downscale factor (default 4)

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <SCENE_ID> <DATASET_DIR> <CKPT_DIR> [DATA_FACTOR]" >&2
  exit 1
fi

SCENE_ID="$1"
DATASET_DIR="$2"
CKPT_DIR="$3"
DATA_FACTOR="${4:-4}"

DATA="${DATASET_DIR}/${SCENE_ID}/gaussian_splat"
CKPT_PATH="${CKPT_DIR}/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt"
OUTPUT_DIR="outputs/difix3d/gsplat/${SCENE_ID}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python examples/gsplat/simple_trainer_difix3d.py default \
  --data_dir "${DATA}" \
  --data_factor "${DATA_FACTOR}" \
  --result_dir "${OUTPUT_DIR}" \
  --no-normalize-world-space \
  --test_every 1 \
  --ckpt "${CKPT_PATH}"
