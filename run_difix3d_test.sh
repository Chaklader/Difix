SCENE_ID="difix3d_2025_07_14"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="NeRF.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/difix3d_enhanced/${SCENE_ID}"

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir "${DATA_DIR}" \
    --data_factor 2 \
    --batch_size 4 \
    --result_dir "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test_every 10 \
    --max_steps 32000 \
    --eval_steps 31999 \
    --save_steps 31999 \
    --fix_steps 31800 \
    --ckpt "${CKPT_PATH}"

echo "Difix3D smoke test completed for +2000 steps"

