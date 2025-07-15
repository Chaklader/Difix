#!/usr/bin/env bash
# Split a COLMAP images directory into 90% train / 10% eval by creating
# symbolic links under images/train and images/eval.
# Usage:  split_train_eval.sh /path/to/colmap_processed/images
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <images_dir>" >&2
  exit 1
fi

IMG_DIR="$1"
cd "$IMG_DIR"

# 1) create sub-directories if they do not exist
mkdir -p train eval

# 2) count images and pick 10 % for eval
mapfile -t all_imgs < <(ls -1 *.png)
TOTAL=${#all_imgs[@]}
if [[ $TOTAL -eq 0 ]]; then
  echo "No .png images found in $IMG_DIR" >&2
  exit 1
fi
EVAL_N=$(( TOTAL / 10 ))

printf "%s\n" "${all_imgs[@]}" | shuf -n "$EVAL_N" > /tmp/eval_list.txt

# 3) symlink the selected eval frames
while read -r f; do
  ln -s "../$f" "eval/$(basename "${f%.*}")_eval_${f##*_}" || true
done < /tmp/eval_list.txt

# 4) symlink the remaining frames into train
for f in "${all_imgs[@]}"; do
  if ! grep -qx "$f" /tmp/eval_list.txt; then
    ln -s "../$f" "train/$(basename "${f%.*}")_train_${f##*_}" || true
  fi
done

TRAIN_N=$(( TOTAL - EVAL_N ))
rm /tmp/eval_list.txt

echo "Created $TRAIN_N train and $EVAL_N eval symlinks inside $IMG_DIR/{train,eval}"
