python - <<'PY'
from pathlib import Path
from PIL import Image
folder = Path('~/datasets/colmap_workspace/images').expanduser()
for img in sorted(folder.glob('*.jpg'))[:5]:
    w, h = Image.open(img).size
    print(img.name, f'{w}x{h}')
PY