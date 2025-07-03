#!/usr/bin/env python3
"""
Convert Nerfstudio splatfacto checkpoint to gsplat format for Difix3D
Usage: python convert_checkpoint.py --nerfstudio_ckpt path/to/step-000029999.ckpt --output_gsplat_ckpt converted.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json

def convert_nerfstudio_to_gsplat(nerfstudio_ckpt_path, output_path):
    """
    Convert Nerfstudio splatfacto checkpoint to gsplat format
    """
    print(f"Loading Nerfstudio checkpoint: {nerfstudio_ckpt_path}")
    
    # Load Nerfstudio checkpoint with weights_only=False (for compatibility)
    checkpoint = torch.load(nerfstudio_ckpt_path, map_location='cpu', weights_only=False)
    
    # Extract pipeline state
    pipeline_state = checkpoint['pipeline']
    
    print("Extracting Gaussian parameters...")
    
    # Extract Gaussian Splatting parameters from Nerfstudio format
    try:
        # Extract parameters with the exact naming pattern found
        means = pipeline_state['_model.gauss_params.means']
        scales = pipeline_state['_model.gauss_params.scales']
        quats = pipeline_state['_model.gauss_params.quats']
        features_dc = pipeline_state['_model.gauss_params.features_dc']
        features_rest = pipeline_state['_model.gauss_params.features_rest']
        opacities = pipeline_state['_model.gauss_params.opacities']
        
        print(f"Found means: shape {means.shape}")
        print(f"Found scales: shape {scales.shape}")
        print(f"Found quats: shape {quats.shape}")
        print(f"Found features_dc: shape {features_dc.shape}")
        print(f"Found features_rest: shape {features_rest.shape}")
        print(f"Found opacities: shape {opacities.shape}")
        
        num_gaussians = means.shape[0]
        print(f"Successfully extracted {num_gaussians} Gaussians")
        
        # Create Difix3D-compatible checkpoint (expects a `splats` dict with sh0/shN)
        gsplat_checkpoint = {
            'splats': {
                'means': means.clone(),
                'scales': scales.clone(),
                'quats': quats.clone(),
                'opacities': opacities.clone().squeeze(-1),
                # SH coefficients: degree-0 in sh0, higher degrees in shN
                'sh0': features_dc.clone().unsqueeze(1),     # [N,1,3]
                'shN': features_rest.clone(),                 # [N,15,3]
            },
            'step': checkpoint.get('step', 29999),
            'global_step': checkpoint.get('step', 29999),
        }
        
        # Add metadata
        gsplat_checkpoint['metadata'] = {
            'source': 'nerfstudio_splatfacto',
            'num_gaussians': num_gaussians,
            'converted_from': str(nerfstudio_ckpt_path),
            'original_step': checkpoint.get('step', 29999),
        }
        
        print(f"Saving gsplat checkpoint to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gsplat_checkpoint, output_path)
        
        print("Conversion completed successfully!")
        print(f"Converted {num_gaussians} Gaussians")
        print(f"Output saved to: {output_path}")
        
        return gsplat_checkpoint
        
    except KeyError as e:
        print(f"Missing required parameter: {e}")
        print("Available Gaussian parameters:")
        for key in pipeline_state.keys():
            if 'gauss_params' in key:
                print(f"  {key}: {pipeline_state[key].shape}")
        raise
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise

def inspect_checkpoint(ckpt_path):
    """
    Inspect checkpoint structure for debugging
    """
    print(f"Inspecting checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    print("Top-level keys:")
    for key in checkpoint.keys():
        print(f"  {key}: {type(checkpoint[key])}")
    
    print(f"Training step: {checkpoint.get('step', 'unknown')}")
    
    if 'pipeline' in checkpoint:
        pipeline_state = checkpoint['pipeline']
        print(f"\nPipeline keys: {len(pipeline_state)} total")
        
        # Show Gaussian parameters specifically
        gauss_params = {}
        for key in pipeline_state.keys():
            if 'gauss_params' in key:
                gauss_params[key] = pipeline_state[key]
        
        if gauss_params:
            print("\nGaussian Parameters:")
            for key, tensor in gauss_params.items():
                print(f"  {key}: {tensor.shape} ({tensor.dtype})")
        
        # Show other model components
        other_keys = [k for k in pipeline_state.keys() if 'gauss_params' not in k]
        if other_keys:
            print(f"\nOther pipeline components: {len(other_keys)} items")
            # Show first few for reference
            for key in sorted(other_keys)[:5]:
                if hasattr(pipeline_state[key], 'shape'):
                    print(f"  {key}: {pipeline_state[key].shape}")
                else:
                    print(f"  {key}: {type(pipeline_state[key])}")
            if len(other_keys) > 5:
                print(f"  ... and {len(other_keys) - 5} more")

def main():
    parser = argparse.ArgumentParser(description='Convert Nerfstudio checkpoint to gsplat format')
    parser.add_argument('--nerfstudio_ckpt', type=str, required=True, 
                       help='Path to Nerfstudio checkpoint (.ckpt)')
    parser.add_argument('--output_gsplat_ckpt', type=str, required=False,
                       help='Output path for gsplat checkpoint (.pt)')
    parser.add_argument('--inspect', action='store_true',
                       help='Only inspect checkpoint structure without conversion')
    
    args = parser.parse_args()
    
    nerfstudio_ckpt_path = Path(args.nerfstudio_ckpt)
    
    if not nerfstudio_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {nerfstudio_ckpt_path}")
    
    if args.inspect:
        inspect_checkpoint(nerfstudio_ckpt_path)
    else:
        if not args.output_gsplat_ckpt:
            raise ValueError("--output_gsplat_ckpt is required for conversion")
        output_path = Path(args.output_gsplat_ckpt)
        convert_nerfstudio_to_gsplat(nerfstudio_ckpt_path, output_path)

if __name__ == '__main__':
    main()
