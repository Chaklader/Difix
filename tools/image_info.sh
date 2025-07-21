#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# image_info.sh
# -----------------------------------------------------------------------------
# Analyzes images in a folder and prints info useful for debugging VAE errors.
# For each image: filename, width, height, and if dims are multiples of 8.
#
# Usage:
#   ./tools/image_info.sh <IMAGES_DIR>
#
#   IMAGES_DIR : Path to folder with images (e.g., ~/datasets/colmap_processed/images/).
#
# Requires: ImageMagick (install with 'sudo apt install imagemagick' on Ubuntu).
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <IMAGES_DIR>" >&2
  exit 1
fi

IMAGES_DIR="$(realpath "$1")"

if ! command -v identify &> /dev/null; then
  echo "[ERROR] ImageMagick not installed. Install with 'sudo apt install imagemagick'." >&2
  exit 1
fi

echo "Analyzing images in ${IMAGES_DIR}"
echo "Filename | Width | Height | Multiple of 8?"
echo "------------------------------------------"

for img in "${IMAGES_DIR}"/*.[pj][np][gg]; do
  if [[ ! -f "$img" ]]; then
    continue
  fi
  info=$(identify -format "%w %h" "$img")
  width=$(echo "$info" | cut -d' ' -f1)
  height=$(echo "$info" | cut -d' ' -f2)
  filename=$(basename "$img")
  if (( width % 8 == 0 && height % 8 == 0 )); then
    status="Yes"
  else
    status="No (width: $((width % 8)), height: $((height % 8)))"
  fi
  printf "%s | %d | %d | %s\n" "$filename" "$width" "$height" "$status"
done

echo "\nSummary:"
total=$(ls "${IMAGES_DIR}"/*.[pj][np][gg] 2>/dev/null | wc -l)
problematic=$(grep -c "No" <(for img in "${IMAGES_DIR}"/*.[pj][np][gg]; do identify -format "%w %h\n" "$img" | awk '{if ($1 % 8 != 0 || $2 % 8 != 0) print "No"}'; done))
echo "Total images: ${total}"
echo "Problematic (not multiple of 8): ${problematic}"
if (( problematic > 0 )); then
  echo "Suggestion: Resize or pad images to multiples of 8 before training."
fi 