#!/usr/bin/env python
"""Align a Nerfstudio/Splatfacto Gaussian-Splat checkpoint to the COLMAP
poses used by Difix3D.

Given
1.  `transforms.json` produced by Nerfstudio (contains camera-to-world
    matrices for each training image) and
2.  the COLMAP dataset directory that Difix3D reads,
this script solves a global similarity transform (scale *s*, rotation *R*,
translation *t*) that best aligns the Nerfstudio camera centres to the
COLMAP camera centres (Umeyama/Procrustes alignment).

It then applies that same similarity to all splat parameters in the
checkpoint (means, scales, quaternions) and writes a new, aligned
checkpoint that can be loaded with `--no-normalize-world-space`.

Usage example:

    python tools/align_checkpoint.py \
        --transforms_json transforms.json \
        --colmap_dir /path/to/colmap_processed \
        --ckpt_in NeRF.pt \
        --ckpt_out NeRF_aligned.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rs

# Lazily add project root to path so that imports in checkpoint dicts still work
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root

from pycolmap import SceneManager
import os


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Return similarity that maps *src* -> *dst* (both 3xN).

    Returns (scale, R(3x3), t(3,)). Implementation adapted from
    *Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns", PAMI 1991*.
    """
    assert src.shape == dst.shape and src.shape[0] == 3, "inputs 3xN"

    n = src.shape[1]
    mu_src = src.mean(axis=1, keepdims=True)
    mu_dst = dst.mean(axis=1, keepdims=True)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = dst_c @ src_c.T / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1.0
    R = (U * d) @ Vt

    if with_scale:
        s = (S * d).sum() / (src_c ** 2).sum()
    else:
        s = 1.0
    t = mu_dst.squeeze() - s * R @ mu_src.squeeze()
    return s, R, t


def load_nerf_cam_centers(json_path: Path):
    js = json.load(open(json_path))
    centres, ids = [], []
    for f in js["frames"]:
        Tcw = np.array(f["transform_matrix"], dtype=np.float32)
        centres.append(Tcw[:3, 3])
        ids.append(int(f.get("colmap_im_id", -1)))
    return np.stack(centres), np.array(ids, dtype=int)


def load_colmap_cam_centers(data_dir: Path):
    """Return camera centres (world coordinate) as array shape (N_im, 3).
    The function looks for COLMAP sparse dir at
        data_dir/sparse/0  or  data_dir/colmap/sparse/0 .
    """
    colmap_dir = data_dir / "sparse/0"
    if not colmap_dir.exists():
        colmap_dir = data_dir / "colmap/sparse/0"
    if not colmap_dir.exists():
        raise FileNotFoundError(f"Could not find COLMAP sparse model under {data_dir}.")

    mgr = SceneManager(str(colmap_dir))
    mgr.load_images()

    centres = {}
    for im_id, im in mgr.images.items():
        # Camera centre C = -R^T * t
        R = im.R()
        t = im.tvec
        C = -R.T @ t
        centres[im_id] = C.astype(np.float32)
    return centres


def main(opt):
    ns_xyz, ns_ids = load_nerf_cam_centers(opt.transforms_json)
    cm_xyz = load_colmap_cam_centers(opt.colmap_dir)

    # Match by COLMAP image id if available, else by order intersection
    if (ns_ids >= 0).any():
        sel_src, sel_dst = [], []
        for p, im_id in zip(ns_xyz, ns_ids):
            if im_id in cm_xyz:
                sel_src.append(p)
                sel_dst.append(cm_xyz[im_id])
        if len(sel_src) < 3:
            raise RuntimeError("Too few matching image IDs between transforms.json and COLMAP.")
        ns_xyz = np.stack(sel_src)
        cm_xyz = np.stack(sel_dst)
    else:
        # Fallback: rely on ordering, take min length
        cm_list = np.stack(list(cm_xyz.values()))
        min_len = min(len(ns_xyz), len(cm_list))
        ns_xyz = ns_xyz[:min_len]
        cm_xyz = cm_list[:min_len]

    src = ns_xyz.T  # 3xN
    dst = cm_xyz.T  # 3xN
    scale, R, t = umeyama(src, dst, with_scale=True)

    print("Solved similarity:\n scale:", scale, "\n R:\n", R, "\n t:", t)

    ckpt = torch.load(opt.ckpt_in, map_location="cpu")

    # Ensure computations stay in original dtype (usually float32)
    m = ckpt["splats"]["means"]                 # (N,3) float32
    dtype = m.dtype
    R_t = torch.tensor(R, dtype=dtype)
    t_t = torch.tensor(t, dtype=dtype)
    s_t = torch.tensor(scale, dtype=dtype)

    ckpt["splats"]["means"] = (m @ R_t.T) * s_t + t_t

    # Scales (log-space) – add log(s) in same dtype
    ckpt["splats"]["scales"] += torch.tensor(np.log(scale), dtype=ckpt["splats"]["scales"].dtype)

    # Quaternions: rotate by R then renormalise
    q = ckpt["splats"]["quats"]  # [N,4] (x,y,z,w)
    qR = torch.tensor(Rs.from_matrix(R).as_quat(), dtype=q.dtype)  # (x,y,z,w)
    x1, y1, z1, w1 = qR
    x2, y2, z2, w2 = q.T
    qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q_new = torch.stack([qx, qy, qz, qw], dim=1)
    ckpt["splats"]["quats"] = q_new / torch.linalg.norm(q_new, dim=1, keepdim=True)

    torch.save(ckpt, opt.ckpt_out)
    print("Aligned checkpoint written to", opt.ckpt_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms_json", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--ckpt_in", type=Path, required=True)
    parser.add_argument("--ckpt_out", type=Path, required=True)
    main(parser.parse_args())
