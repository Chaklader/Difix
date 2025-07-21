#!/usr/bin/env bash
# ----------------------------------------------------------------------------- 
# image_info.sh – list image sizes and show which are NOT multiples of 8
# -----------------------------------------------------------------------------
# Usage:  ./tools/image_info.sh <IMAGES_DIR>
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <IMAGES_DIR> [DIVISOR]" >&2
  exit 1
fi

IMAGES_DIR="$(realpath "$1")"
DIVISOR="${2:-8}"

printf "Analyzing images in %s (divisor=%s)\n" "$IMAGES_DIR" "$DIVISOR"
printf "Filename |  W  |  H  | multiple-of-%s?\n" "$DIVISOR"
printf -- "-------------------------------------------\n"

# enable better globbing
shopt -s nullglob nocaseglob extglob

# shell array to store problematic files
bad=()
total=0

print_line () { printf "%s | %4d | %4d | %s\n" "$1" "$2" "$3" "$4"; }

if command -v identify &>/dev/null; then
  # ---------- fast path: ImageMagick present ----------
  for img in "$IMAGES_DIR"/*.@(png|jpg|jpeg); do
    (( total++ ))
    read -r w h <<<"$(identify -format "%w %h" "$img")"
    base=$(basename "$img")
    if (( w % DIVISOR == 0 && h % DIVISOR == 0 )); then
      print_line "$base" "$w" "$h" "yes"
    else
      print_line "$base" "$w" "$h" "NO"
      bad+=("$base")
    fi
  done
else
  echo "Error: 'identify' (ImageMagick) not found. Please install ImageMagick or adjust the script." >&2
  exit 1
fi

printf "\nSummary : %d files  |  %d not multiple-of-%s\n" "$total" "${#bad[@]}" "$DIVISOR"
if ((${#bad[@]})); then
  printf "Hint    : pad/resize those images or resize inputs so both dimensions are divisible by %s\n" "$DIVISOR"
fi