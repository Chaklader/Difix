#!/usr/bin/env bash
# Quick utility to display GPU memory summary (CSV format).
# Usage:  ./check_gpu_memory.sh

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
