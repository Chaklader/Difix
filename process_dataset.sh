#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# process_dataset.sh
# ------------------------------------------------------------------------------
# Utility script to convert a raw COLMAP workspace into a Nerfstudio-formatted
# dataset ("nerfstudio-data" format) that Difix3D can train on.
#
# Usage:
#   ./process_dataset.sh <COLMAP_WORKSPACE_DIR> <OUTPUT_DIR> [NUM_DOWNSCALES]
#   example:
#   ./process_dataset.sh ~/datasets/colmap_workspace/images_clean ~/datasets/colmap_processed 0
#   NUM_DOWNSCALES controls additional 2× down-scales of the images:
#     0 → keep original resolution (folder: images/)
#     1 → half resolution (creates images_2/)
#     2 → quarter resolution (creates images_4/), and so on.
#   IMPORTANT: set --downscale_factor in run_difix3d_train.sh to 2^NUM_DOWNSCALES
#   so the training script loads from the matching images_K/ folder.
#   COLMAP_WORKSPACE_DIR : Path to the original COLMAP workspace that contains
#                          the `images/` folder and (optionally) the COLMAP
#                          database/sparse outputs.
#   OUTPUT_DIR           : Destination directory for the processed dataset.
#   NUM_DOWNSCALES       : (Optional) Number of 2× downscales to apply to the
#                          images. 0 keeps full resolution, 1 halves it, 2
#                          quarters it, etc. Default: 0.
#
# ./process_dataset.sh ~/datasets/colmap_workspace/images_clean ~/datasets/colmap_processed 0
#
# Notes:
#   • The script forces Qt into off-screen mode (required on headless servers).
#   • The --no-gpu flag disables OpenGL so COLMAP can run on machines without a
#     display or GPU context.  Remove it if you prefer GPU feature extraction /
#     matching and an X server is available.
# ------------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $(basename "$0") <COLMAP_WORKSPACE_DIR> <OUTPUT_DIR> [NUM_DOWNSCALES]" >&2
  exit 1
fi

RAW_DATA_DIR=$(realpath "$1")
OUTPUT_DIR=$(realpath "$2")
NUM_DOWNSCALES=${3:-0}

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "[process_dataset] Converting $RAW_DATA_DIR → $OUTPUT_DIR (downscales=$NUM_DOWNSCALES)"

# Headless environment settings
export QT_QPA_PLATFORM=offscreen

# Run ns-process-data (CPU-only via --no-gpu)
ns-process-data images \
  --data        "$RAW_DATA_DIR" \
  --output-dir  "$OUTPUT_DIR" \
  --num-downscales "$NUM_DOWNSCALES" \
  --no-gpu

echo "[process_dataset] Conversion finished. Key files:"
ls -lh "$OUTPUT_DIR"/transforms.json 

echo "[process_dataset] Done."
