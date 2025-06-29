#!/usr/bin/env bash
# One-shot Difix image cleaner wrapper
# Usage: ./one_shot_clean.sh [extra python args]
set -euo pipefail
PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src" python one_shot_clean.py "$@"
exit 0