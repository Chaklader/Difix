#!/usr/bin/env python3
"""One-shot Difix image cleaner.

Extracted from the previous one_shot_clean.sh inline Python so it can be run
directly:

    PYTHONPATH=$PWD/src python one_shot_clean.py

Adjust SCENE, TARGET or TIMESTEP below as needed.
"""
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers.utils import load_image
from pipeline_difix import DifixPipeline
import torch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SCENE = Path("/home/azureuser/datasets/colmap_workspace")
IMG_DIR = SCENE / "images"
CLEAN_DIR = SCENE / "images_clean"
CLEAN_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Load Difix model (FP16 on GPU)
# ----------------------------------------------------------------------------
pipe = DifixPipeline.from_pretrained(
    "nvidia/difix",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to("cuda")
pipe.enable_attention_slicing()  # lighter VRAM
# DO NOT call pipe.enable_vae_tiling() – it breaks skip sizes

# ----------------------------------------------------------------------------
# Cleaning parameters
# ----------------------------------------------------------------------------
# max side fed to VAE (valid multiples of 128 up to ≈4 K: 2048, 2304, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096)
TARGET = 2688
TIMESTEP = 199  # denoise strength (higher = milder)

# ----------------------------------------------------------------------------
# Processing loop
# ----------------------------------------------------------------------------
for p in tqdm(sorted(IMG_DIR.glob("*.jp*g"))):
    orig = load_image(str(p)).convert("RGB")
    w, h = orig.size
    if max(w, h) > TARGET:
        scale = TARGET / max(w, h)
        inp = orig.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        scale = 1.0
        inp = orig

    clean = pipe(
        "remove degradation",
        image=inp,
        num_inference_steps=1,
        timesteps=[TIMESTEP],
        guidance_scale=0.0,
    ).images[0]

    if scale != 1.0:  # restore original resolution
        clean = clean.resize((w, h), Image.LANCZOS)

    clean.save(CLEAN_DIR / p.name)




# from pathlib import Path
# from PIL import Image
# from tqdm import tqdm
# from diffusers.utils import load_image
# from pipeline_difix import DifixPipeline
# import torch

# # paths
# scene     = Path("/home/azureuser/datasets/colmap_workspace")
# img_dir   = scene / "images"
# clean_dir = scene / "images_clean"
# clean_dir.mkdir(exist_ok=True)

# # load Difix (FP16 on GPU)
# pipe = DifixPipeline.from_pretrained(
#           "nvidia/difix",
#           torch_dtype=torch.float16,
#           trust_remote_code=True,
#           low_cpu_mem_usage=True,
#       ).to("cuda")
# pipe.enable_attention_slicing()      # lighter VRAM
# # DO NOT call pipe.enable_vae_tiling() – it breaks skip sizes

# TARGET = 2688                         # max side fed to VAE
# # NOTE: Images larger than TARGET are temporarily downscaled for the Difix
# #       model, then **upsampled back** so the file saved in images_clean/
# #       has exactly the same width × height as the original. No permanent
# #       resolution change occurs.

# for p in tqdm(sorted(img_dir.glob("*.jp*g"))):
#     orig = load_image(str(p)).convert("RGB")
#     w, h = orig.size
#     if max(w, h) > TARGET:
#         scale = TARGET / max(w, h)
#         inp   = orig.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
#     else:
#         scale = 1.0
#         inp   = orig

#     clean = pipe(
#         "remove degradation",
#         image=inp,
#         num_inference_steps=1,
#         timesteps=[199],
#         guidance_scale=0.0,
#     ).images[0]

#     if scale != 1.0:                 # restore original resolution
#         clean = clean.resize((w, h), Image.LANCZOS)

#     clean.save(clean_dir / p.name)

