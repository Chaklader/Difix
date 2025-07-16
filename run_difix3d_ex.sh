SCENE_ID="difix3d_2025_07_14"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="NeRF_unaligned.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/difix3d_enhanced/${SCENE_ID}"

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir "${DATA_DIR}" \
    --data_factor 1 \
    --batch_size 4 \
    --result_dir "${OUTPUT_DIR}" \
    --normalize-world-space \
    --test_every 10 \
    --max_steps 32000 \
    --eval_steps 31999 \
    --save_steps 31999 \
    --fix_steps 31800 \
    --pipeline.model.densify-grad-thresh 0.0003 \
    --pipeline.model.spatial-lr-scale 0.1 \
    --pipeline.model.densify-interval 200 \
    --pipeline.model.prune-interval 200 \
    --pipeline.model.prune-start-iter 180 \
    --ckpt "${CKPT_PATH}"

echo "Difix3D training completed for 32.000 steps"

