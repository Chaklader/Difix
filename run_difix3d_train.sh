SCENE_ID="difix3d_2025_07_14"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="NeRF.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/difix3d_enhanced/${SCENE_ID}"

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir "${DATA_DIR}" \
    --data_factor 1 \
    --batch_size 8 \
    --result_dir "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test_every 2 \
    --max_steps 35500 \
    --eval_steps 32100 34100 35100 \
    --save_steps 32100 34100 35100 \
    --fix_steps 32000 34000 35000 \
    --compression png \
    --ckpt "${CKPT_PATH}"

echo "Difix3D training completed."

