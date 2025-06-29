#!/usr/bin/env bash
# Run the full conversion pipeline:
# 1. Convert PLY ➜ SPZ
# 2. Compute rotation matrix
# 3. Generate boundary JSON + PNG
#
# Usage: ./converter.sh
# (Run from the repository root)
set -euo pipefail

echo "[1/3] Converting PLY to SPZ…"
python src/spz_converter.py

echo "[2/3] Computing rotation matrix…"
python src/rotation_corrction.py

echo "[3/3] Generating boundary…"
python src/boundary.py

echo "✓ All steps completed. Output files are in the exports/ directory."
