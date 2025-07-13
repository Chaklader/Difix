# Set up the Difix3D enhancement command
SCENE_ID="difix3d_20250701_195855"
DATA_DIR="/home/azureuser/datasets/colmap_processed"
CKPT_PATH="NeRF.pt"
OUTPUT_DIR="/mnt/nvme0n1/azureuser/finetune/difix3d_enhanced/${SCENE_ID}"

# Run Difix3D enhancement
CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py mcmc \
    --data_dir "${DATA_DIR}" \
    --data_factor 1 \
    --result_dir "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test_every 8 \
    --max_steps 34100 \
    --ckpt "${CKPT_PATH}"
