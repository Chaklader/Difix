#!/usr/bin/env bash
# Export a point cloud PLY from a finished Difix3D/Nerfacto run.
# Usage: ./export_pointcloud.sh <RUN_NAME>

set -e

if [[ -z "$1" ]]; then
  echo "Usage: $0 <RUN_NAME>" >&2
  exit 1
fi

RUN_NAME="$1"
CFG=$(find ./outputs -type f -name config.yml | grep "${RUN_NAME}" | head -n 1 || true)
if [[ -z "${CFG}" ]]; then
  echo "Could not find config for run ${RUN_NAME}" >&2
  exit 1
fi

OUT_DIR="./exports/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

# usage: ns-export [-h]
#                  {pointcloud,tsdf,poisson,marching-cubes,cameras,gaussian-splat}

# ╭─ options ─────────────────────────────────────────────────────────────╮
# │ -h, --help        show this help message and exit                     │
# ╰───────────────────────────────────────────────────────────────────────╯
# ╭─ subcommands ─────────────────────────────────────────────────────────╮
# │ {pointcloud,tsdf,poisson,marching-cubes,cameras,gaussian-splat}       │
# │     pointcloud    Export NeRF as a point cloud.                       │
# │     tsdf          Export a mesh using TSDF processing.                │
# │     poisson       Export a mesh using poisson surface reconstruction. │
# │     marching-cubes                                                    │
# │                   Export a mesh using marching cubes.                 │
# │     cameras       Export camera poses to a .json file.                │
# │     gaussian-splat                                                    │
# │                   Export 3D Gaussian Splatting model to a .ply        │
# ╰───────────────────────────────────────────────────────────────────────╯

echo "Exporting point cloud to ${OUT_DIR}/pointcloud.ply"

# Example: export a point cloud for a *specific* checkpoint
#   • --normal-method open3d  – compute per-vertex normals via Open3D (install with `pip install open3d`)
#   • You can add `--step <N>` instead of supplying an explicit checkpoint path.
#   • Remove the flag or use `--normal-method none` to skip normals.
#
# This one-liner worked successfully on the A100 box:
# ns-export pointcloud --load-config outputs/difix3d_20250626_073554/difix3d/2025-06-26_073600/config.yml --output-dir exports/ --normal-method open3d

ns-export pointcloud \
  --load-config "${CFG}" \
  --output-dir "${OUT_DIR}" \
  --normal-method open3d 

echo "Done."
