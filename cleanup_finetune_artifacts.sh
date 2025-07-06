#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# cleanup_finetune_artifacts.sh
# ------------------------------------------------------------------------------
# Periodically removes bulky, no-longer-needed artefacts produced by Difix3D
# fine-tuning runs so that the disk does not fill up.
#
# WHAT IT REMOVES
#  1. Entire validation render folders: renders/val/<step>/
#  2. In renders/novel/<step>/ it deletes "Pred" and "Ref" once "Fixed" exists.
#  3. It keeps novel "Fixed" and "Alpha" folders but you can choose to retain
#     only the N most-recent steps (default 3) to save even more space.
#
# Usage:
#   ./cleanup_finetune_artifacts.sh <run_dir> [interval_sec] [keep_steps]
#
#   run_dir      : path to the active run folder, e.g.
#                  finetune/difix3d_enhanced/difix3d_20250706_113551
#   interval_sec : how often to sweep (default 300 seconds)
#   keep_steps   : how many of the newest novel steps to keep (default 3)
#
# Example (after starting fine-tuning):
#   nohup ./cleanup_finetune_artifacts.sh \
#       /mnt/nvme0n1/azureuser/finetune/difix3d_enhanced/difix3d_20250706_113551 \
#       300 3 &
# ------------------------------------------------------------------------------
set -euo pipefail

RUN_DIR="${1:?provide run_dir}"
INTERVAL="${2:-300}"
KEEP="${3:-3}"

while true; do
    # ------------------------------------------------------------------
    # 1. Blow away ALL validation renders (never used again after written)
    # ------------------------------------------------------------------
    if [[ -d "$RUN_DIR/renders/val" ]]; then
        find "$RUN_DIR/renders/val" -mindepth 1 -maxdepth 1 -type d -print0 \
            | xargs -0r rm -rf --
    fi

    # ------------------------------------------------------------------
    # 2. Inside each novel step:
    #    a) if Fixed exists -> delete Pred & Ref
    #    b) optionally keep only the newest $KEEP steps overall
    # ------------------------------------------------------------------
    NOVEL_DIR="$RUN_DIR/renders/novel"
    if [[ -d "$NOVEL_DIR" ]]; then
        # a) delete Pred & Ref once Fixed exists
        find "$NOVEL_DIR" -mindepth 2 -maxdepth 2 -type d -name Fixed -print0 \
            | while IFS= read -r -d '' fixed; do
                  stepdir="$(dirname "$fixed")"
                  rm -rf -- "$stepdir/Pred" "$stepdir/Ref" 2>/dev/null || true
              done

        # b) keep only the newest $KEEP steps (based on step folder name)
        mapfile -t step_dirs < <(ls -1d "$NOVEL_DIR"/*/ 2>/dev/null | sort -n)
        if (( ${#step_dirs[@]} > KEEP )); then
            del_count=$(( ${#step_dirs[@]} - KEEP ))
            for (( i=0; i<del_count; i++ )); do
                rm -rf -- "${step_dirs[i]}"
            done
        fi
    fi

    sleep "$INTERVAL"
done
