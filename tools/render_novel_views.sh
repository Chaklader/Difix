#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# render_novel_views.sh
# -----------------------------------------------------------------------------
# Generate novel-view renders from a GSplat checkpoint **without** training.
# These synthetic views can later be cleaned with the Difix diffusion fixer and
# merged back into the COLMAP dataset to augment training.
#
# How to use:
#   1. Set SCENE_ID and CKPT_PATH below (unaligned or aligned as appropriate).
#   2. Choose a trajectory type: spiral | interp | ellipse.
#   3. Run the script – it will export PNGs into
#        ${OUTPUT_DIR}/renders/novel/1/Pred/  (Difix default).
#   4. Clean the images (see comments at end of file).
# -----------------------------------------------------------------------------

set -euo pipefail

# -------- USER CONFIG --------------------------------------------------------
SCENE_ID="difix3d_novel_renders_synthetic"   # folder tag
DATA_DIR="/home/azureuser/datasets/colmap_processed"      # original dataset
CKPT_PATH="NeRF.pt"                               # GSplat checkpoint
TRAJ="spiral"                                             # spiral | interp | ellipse

# Where to write renders
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/novel_views/${SCENE_ID}"
mkdir -p "${OUTPUT_DIR}"

# -------- RENDER-ONLY CALL ---------------------------------------------------
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir "${DATA_DIR}" \
    --ckpt "${CKPT_PATH}" \
    --no-normalize-world-space \  # checkpoint already aligned – remove if CKPT is unaligned
    --render_traj_path "${TRAJ}" \
    --max_steps 30002 \                # single dummy step
    --eval_steps 30001 \
    --save_steps 999999 \
    --fix_steps 999999 \
    --result_dir "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# NEXT STEPS
# -----------------------------------------------------------------------------
# 1. Clean the renders (remove blur / holes / light-rays) with the diffusion
#    fixer:
#       python src/inference_difix.py \
#           --input_dir  "${OUTPUT_DIR}/renders/novel/1/Pred/" \
#           --output_dir "/mnt/nvme0n1/azureuser/finetune/cleaned_novel_views/" \
#           --model_id   "nvidia/difix_ref"
#
# 2. Copy the cleaned PNGs into your COLMAP dataset images folder, e.g.:
#       cp /mnt/nvme0n1/azureuser/finetune/cleaned_novel_views/*.png \
#          ${DATA_DIR}/images/
#
# 3. Append the corresponding camera poses to ${DATA_DIR}/transforms.json.
#    The poses are written by the renderer to:
#       ${OUTPUT_DIR}/renders/novel/1/poses.json
#    Copy each new entry into the "frames" list of transforms.json.
#
# 4. Re-run Difix3D training with the augmented dataset.
# -----------------------------------------------------------------------------
