# Set up the Difix3D enhancement command
SCENE_ID="difix3d_2025_07_14"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="NeRF.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/difix3d_enhanced/${SCENE_ID}"

# run_difix3d_train.sh  (only the argument lines changed)

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data-dir      "${DATA_DIR}" \
    --data-factor   1 \
    --batch-size    8 \
    --result-dir    "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test-every    2 \
    --max-steps     5000 \
    --eval-steps    2000 4000 5000 \
    --ckpt          "${CKPT_PATH}" 