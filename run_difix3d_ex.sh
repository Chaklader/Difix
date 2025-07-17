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
    --test_every 10 \
    --max_steps 32000 \
    --eval_steps 31999 \
    --save_steps 31999 \
    --fix_steps 31800 \
    --ckpt "${CKPT_PATH}" \
    --opacity_reg 0.01 \
    --scale_reg 0.001 \
    --random_bkgd

echo "Difix3D training completed for 32.000 steps"

