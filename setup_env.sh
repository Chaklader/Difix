#!/usr/bin/env bash
conda create -y -n difix --file env_difix.lock  # or: conda env create -f environment.yml
conda activate difix
export CUDA_HOME=$CONDA_PREFIX        # preserve nvcc headers path
