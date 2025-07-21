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

# ---------- use Python (Pillow) for broader compatibility ----------
python - <<'PY' "$IMAGES_DIR" "$DIVISOR"
import sys, pathlib, json
from PIL import Image

root = pathlib.Path(sys.argv[1])
mod = int(sys.argv[2])
print(f"Analyzing {root} (divisor={mod})")
print(f"{'Filename':24} | {'Width':6} | {'Height':6} | OK?")
print("-"*63)

total = bad = 0
for img in sorted(root.glob('*')):
    if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        continue
    total += 1
    try:
        w, h = Image.open(img).size
    except Exception as e:
        print(f"{img.name:24} |  error opening: {e}")
        bad += 1
        continue
    ok = (w % mod == 0 and h % mod == 0)
    print(f"{img.name:24} | {w:6d} | {h:6d} | {'yes' if ok else 'NO'}")
    if not ok:
        bad += 1
print(f"\nSummary: {total} files  |  {bad} NOT divisible by {mod}")
PY
exit 0
for img in "$IMAGES_DIR"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
  [[ -f $img ]] || continue  # skip if glob didn't expand
  (( total++ ))
  read -r w h <<< "$(identify -format '%w %h' "$img")"
  fname=$(basename "$img")
  if (( w % DIVISOR == 0 && h % DIVISOR == 0 )); then
    status="yes"
  else
    status="NO"; (( bad++ ))
  fi
  printf "%-24s | %6d | %6d | %s\n" "$fname" "$w" "$h" "$status"
done

printf "\nSummary: %d files  |  %d NOT divisible by %s\n" "$total" "$bad" "$DIVISOR"
