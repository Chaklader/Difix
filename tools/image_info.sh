#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# image_info.sh – list image dimensions and flag those not divisible by a given
# divisor (default 8). Handy for diagnosing VAE / tiled decode issues.
# -----------------------------------------------------------------------------
# Usage:  ./tools/image_info.sh <IMAGES_DIR> [DIVISOR]
# Example: ./tools/image_info.sh ~/datasets/colmap_processed/images 256
# -----------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <IMAGES_DIR> [DIVISOR]" >&2
  exit 1
fi

IMAGES_DIR="$(realpath "$1")"
DIVISOR="${2:-8}"

if ! command -v identify &>/dev/null; then
  echo "Error: 'identify' (ImageMagick) not found. Please install it." >&2
  exit 1
fi

printf "Analyzing %s (divisor=%s)\n" "$IMAGES_DIR" "$DIVISOR"
printf "%-24s | %6s | %6s | %s\n" "Filename" "Width" "Height" "OK?"
printf -- "---------------------------------------------------------------\n"

total=0; bad=0
while IFS= read -r -d '' img; do
  (( total++ ))
  read -r w h <<< "$(identify -format '%w %h' "$img")"
  fname=$(basename "$img")
  if (( w % DIVISOR == 0 && h % DIVISOR == 0 )); then
    status="yes"
  else
    status="NO"
    (( bad++ ))
  fi
  printf "%-24s | %6d | %6d | %s\n" "$fname" "$w" "$h" "$status"
done < <(find "$IMAGES_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0)

printf "\nSummary: %d files  |  %d NOT divisible by %s\n" "$total" "$bad" "$DIVISOR"
