#!/usr/bin/env bash
# ============================================================
# Difix3D "sprint-to-30dB" fine-tuning pipeline for a COLMAP
# scene starting from a NeRF-studio GSplat checkpoint that has
# already been aligned to the COLMAP frame (e.g. NeRF_aligned.pt).
#
# The run is split into three trainer invocations so we can change
# resolution and batch size without touching the codebase.
# Each phase resumes from the checkpoint written by the previous
# one and writes into the SAME result directory so TensorBoard
# charts remain continuous.
#
# Estimated 1-GPU (A100-80G) wall-clock ≈ 7–8 h.
# ============================================================
set -euo pipefail

########################################
# Scene-specific paths (EDIT AS NEEDED)
########################################
SCENE_ID="difix_sprint"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_INIT="$(pwd)/NeRF_aligned.pt"          # 30k Nerfstudio checkpoint, already aligned
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/${SCENE_ID}"
mkdir -p "${OUTPUT_DIR}"

########################################
# Helper – resolve latest checkpoint
########################################
latest_ckpt() {
  ls -1t "${OUTPUT_DIR}"/ckpts/ckpt_*_rank0.pt | head -n 1
}

########################################
# 0. One-off PNG compression (≈10 min)
########################################
if [[ ! -f "${OUTPUT_DIR}/ckpts/ckpt_comp_rank0.pt" ]]; then
  echo "[Phase 0] Running PNG compression pass …"
  # Figure out the step stored in the initial checkpoint so we can run **exactly
  # one more** training iteration (step -> step+1). This ensures:
  #   1. The training loop executes once and triggers the PNG compression.
  #   2. A fresh checkpoint is saved (condition step==max_steps-1).
  CKPT_STEP=$(python -c 'import torch,sys; print(torch.load(sys.argv[1], map_location="cpu").get("step",0))' "${CKPT_INIT}")
  NEXT_STEP=$((CKPT_STEP + 1))

  CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
    --data_dir "${DATA_DIR}" \
    --result_dir "${OUTPUT_DIR}" \
    --compression png \
    --max_steps "${NEXT_STEP}" \
    --eval_steps "${NEXT_STEP}" \
    --ckpt "${CKPT_INIT}" \
    --no-normalize-world-space \
    --disable_viewer || true

  # Rename the freshly saved compressed checkpoint for clarity
  mv "$(latest_ckpt)" "${OUTPUT_DIR}/ckpts/ckpt_comp_rank0.pt"
fi
CKPT_PATH="${OUTPUT_DIR}/ckpts/ckpt_comp_rank0.pt"

########################################
# 1. Phase A – ¼-res (factor 4), batch 16, 15 k steps (≈1 h)
########################################
echo "[Phase 1] Low-res fast shrink …"
CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
  --data_dir "${DATA_DIR}" \
  --result_dir "${OUTPUT_DIR}" \
  --ckpt "${CKPT_PATH}" \
  --no-normalize-world-space \
  --data_factor 4 \
  --batch_size 16 \
  --test_every 2 \
  --max_steps 15000 \
  --eval_steps 15000 \
  --disable_viewer
CKPT_PATH="$(latest_ckpt)"

########################################
# 2. Phase B – ½-res (factor 2), batch 12, +25 k steps (≈2.5 h)
########################################
echo "[Phase 2] Mid-res refinement …"
CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
  --data_dir "${DATA_DIR}" \
  --result_dir "${OUTPUT_DIR}" \
  --ckpt "${CKPT_PATH}" \
  --no-normalize-world-space \
  --data_factor 2 \
  --batch_size 12 \
  --test_every 2 \
  --max_steps 40000 \
  --eval_steps 30000 40000 \
  --disable_viewer
CKPT_PATH="$(latest_ckpt)"

########################################
# 3. Phase C – full-res (factor 1) polish, batch 8, +20 k steps + fixer (≈3.5 h)
########################################
echo "[Phase 3] Half-res polish + fixer …"
CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
  --data_dir "${DATA_DIR}" \
  --result_dir "${OUTPUT_DIR}" \
  --ckpt "${CKPT_PATH}" \
  --no-normalize-world-space \
  --data_factor 1 \
  --batch_size 8 \
  --test_every 2 \
  --max_steps 60000 \
  --eval_steps 50000 60000 \
  --fix_steps 60000 \
  --disable_viewer

echo "===== Sprint complete. Final checkpoint: $(latest_ckpt) ====="
