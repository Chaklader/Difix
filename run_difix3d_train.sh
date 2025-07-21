SCENE_ID="charles_difix3d_2025_07_21"
DATA_DIR="/home/azureuser/datasets/colmap_processed_charles"
CKPT_PATH="NeRF.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune_charles/difix3d_enhanced/${SCENE_ID}"

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir "${DATA_DIR}" \
    --data_factor 1 \
    --batch_size 8 \
    --result_dir "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test_every 10 \
    --max_steps 40000 \
    --eval_steps 32100 34100 35100 37100 39100 \
    --save_steps 32100 34100 35100 37100 39100 \
    --fix_steps 32000 34000 35000 37000 39000 \
    --ckpt "${CKPT_PATH}"

echo "Difix3D training completed for 40.000 steps"

