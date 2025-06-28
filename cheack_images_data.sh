#!/usr/bin/env bash

python3 - <<'PY'
from pathlib import Path
from PIL import Image
folder = Path('~/datasets/colmap_workspace/images').expanduser()
# include .jpg and .jpeg (case sensitive)
for img in sorted(folder.glob('*.jp*g'))[:5]:
    w, h = Image.open(img).size
    print(img.name, f'{w}x{h}')
PY