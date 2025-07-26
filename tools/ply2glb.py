#!/usr/bin/env python3
"""Convert a PLY point-/triangle-mesh to a Draco-compressed GLB.

Default behaviour (no arguments):
    python tools/ply2glb.py
converts     assets/NeRF.ply  →  assets/NeRF.glb    (Draco, loss-less)

Optional CLI:
    --in    /path/to/input.ply
    --out   /path/to/output.glb

Requirements:
    pip install trimesh pygltflib
    sudo apt install meshoptimizer (gltfpack)   # or place `gltfpack` in $PATH

The script first exports an uncompressed GLB with trimesh, then – if the
`gltfpack` binary is available – repacks it with Draco compression using
`-tc -noq` (Draco, no quantisation → loss-less).  If `gltfpack` is not
found, it leaves the raw GLB in place and prints a warning.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import trimesh



def export_raw_glb(ply_path: Path, tmp_glb: Path) -> None:
    """Export every point from the input PLY directly into a GLB.

    The glTF standard supports POINTS primitives, so we simply keep all
    vertices (and any vertex colours) intact.  This avoids geometry
    distortion and removes heavy dependencies like SciPy.
    """
    # Load PLY; this may return PointCloud, Trimesh, or Scene
    obj = trimesh.load(ply_path)
    if isinstance(obj, trimesh.Scene):
        # Merge all geometry into a single cloud/mesh
        obj = trimesh.util.concatenate(tuple(g for g in obj.geometry.values()))
    cloud = obj  # now PointCloud or Trimesh

    if cloud.is_empty:
        raise RuntimeError(f"No vertices found in {ply_path}")

    print(f"[ply2glb] Loaded {len(cloud.vertices)} points – exporting GLB…")
    cloud.export(tmp_glb, file_type='glb')


def draco_compress(input_glb: Path, output_glb: Path) -> bool:
    """Run gltfpack with loss-less Draco compression."""
    gltfpack = shutil.which('gltfpack')
    if gltfpack is None:
        return False

    cmd = [gltfpack, '-i', str(input_glb), '-o', str(output_glb), '-tc', '-noq']
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ply2glb] gltfpack failed ({e}). Keeping raw GLB.", file=sys.stderr)
        return False


def convert(ply_path: Path, output_glb: Path) -> None:
    ply_path = ply_path.expanduser().resolve()
    output_glb = output_glb.expanduser().resolve()
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_glb = Path(tmpdir) / 'temp.glb'
        print(f"[ply2glb] Converting {ply_path} to GLB...")
        export_raw_glb(ply_path, tmp_glb)

        # Attempt Draco compression
        if draco_compress(tmp_glb, output_glb):
            print(f"[ply2glb] ✅ Draco-compressed GLB written to: {output_glb}")
        else:
            shutil.move(tmp_glb, output_glb)
            print(f"[ply2glb] ✅ Uncompressed GLB written to: {output_glb}")
            print("[ply2glb] Install gltfpack for Draco compression: sudo apt install meshoptimizer")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PLY point cloud to GLB mesh')
    parser.add_argument('--in', dest='inp', default='assets/NeRF.ply', help='Input PLY file')
    parser.add_argument('--out', dest='out', default='assets/NeRF.glb', help='Output GLB file')
    args = parser.parse_args()

    try:
        convert(Path(args.inp), Path(args.out))
        print('[ply2glb] Done!')
    except Exception as exc:
        print(f'[ply2glb] Error: {exc}', file=sys.stderr)
        sys.exit(1)