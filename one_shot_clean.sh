PYTHONPATH=$PWD/src python - <<'PY'
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers.utils import load_image
from pipeline_difix import DifixPipeline
import torch

# paths
scene     = Path("/home/azureuser/datasets/colmap_workspace")
img_dir   = scene / "images"
clean_dir = scene / "images_clean"
clean_dir.mkdir(exist_ok=True)

# load Difix (FP16 on GPU)
pipe = DifixPipeline.from_pretrained(
          "nvidia/difix",
          torch_dtype=torch.float16,
          trust_remote_code=True,
          low_cpu_mem_usage=True,
      ).to("cuda")
pipe.enable_attention_slicing()      # lighter VRAM
# DO NOT call pipe.enable_vae_tiling() – it breaks skip sizes

TARGET = 3072                         # max side fed to VAE
# NOTE: Images larger than TARGET are temporarily downscaled for the Difix
#       model, then **upsampled back** so the file saved in images_clean/
#       has exactly the same width × height as the original. No permanent
#       resolution change occurs.

for p in tqdm(sorted(img_dir.glob("*.jp*g"))):
    orig = load_image(str(p)).convert("RGB")
    w, h = orig.size
    if max(w, h) > TARGET:
        scale = TARGET / max(w, h)
        inp   = orig.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    else:
        scale = 1.0
        inp   = orig

    clean = pipe(
        "remove degradation",
        image=inp,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0,
    ).images[0]

    if scale != 1.0:                 # restore original resolution
        clean = clean.resize((w, h), Image.LANCZOS)

    clean.save(clean_dir / p.name)
PY
