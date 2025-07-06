# ------------------------------------------------------------------------------
# Difix3D GSplat fine-tune launcher
# Uses hyper-parameters defined in parameters.sh so they can be tweaked easily
# ------------------------------------------------------------------------------

# Load schedule parameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/parameters.sh"

# Set up the Difix3D enhancement command
SCENE_ID="difix3d_20250706_113551"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="/mnt/nvme0n1/azureuser/finetune/gsplat_init.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/difix3d_enhanced/${SCENE_ID}"

# Run Difix3D enhancement
CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir "${DATA_DIR}" \
    --data_factor 1 \
    --result_dir "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test_every 8 \
    --ckpt "${CKPT_PATH}" \
    --max_steps "${MAX_STEPS}" \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --fix_steps  ${FIX_STEPS} \
    --batch_size ${BATCH_SIZE}
