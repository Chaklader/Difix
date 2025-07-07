#!/usr/bin/env python3
"""
Comprehensive PLY file analyzer and comparison tool.
Compares two PLY files to identify differences that might cause rendering issues.

Usage:
python analyze_ply.py --ply1 good_model.ply --ply2 problematic_model.ply
"""

import argparse
import struct
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class PLYAnalyzer:
    def __init__(self, ply_path: Path):
        self.ply_path = ply_path
        self.header_info = {}
        self.data = {}
        self.properties = []
        self.num_vertices = 0
        
    def parse_header(self) -> Dict[str, Any]:
        """Parse PLY header and extract metadata."""
        print(f"\n{'='*50}")
        print(f"ANALYZING: {self.ply_path.name}")
        print(f"{'='*50}")
        
        header_lines = []
        with open(self.ply_path, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
        
        # Extract key information
        for line in header_lines:
            if line.startswith('format'):
                self.header_info['format'] = line
            elif line.startswith('element vertex'):
                self.num_vertices = int(line.split()[2])
                self.header_info['num_vertices'] = self.num_vertices
            elif line.startswith('property'):
                prop_parts = line.split()
                if len(prop_parts) >= 3:
                    prop_type = prop_parts[1]
                    prop_name = prop_parts[2]
                    self.properties.append((prop_name, prop_type))
        
        self.header_info['properties'] = self.properties
        self.header_info['num_properties'] = len(self.properties)
        
        # Print header analysis
        print(f"Format: {self.header_info.get('format', 'Unknown')}")
        print(f"Number of vertices: {self.num_vertices:,}")
        print(f"Number of properties: {len(self.properties)}")
        print("\nProperties:")
        for i, (name, ptype) in enumerate(self.properties):
            print(f"  {i:2d}: {name:15s} ({ptype})")
        
        return self.header_info
    
    def load_data(self) -> Dict[str, np.ndarray]:
        """Load binary PLY data."""
        print(f"\nLoading binary data...")
        
        # Skip header
        with open(self.ply_path, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8').strip()
                if line == 'end_header':
                    break
            
            # Read binary data
            num_floats = self.num_vertices * len(self.properties)
            binary_data = f.read(num_floats * 4)  # 4 bytes per float
            
            if len(binary_data) != num_floats * 4:
                raise ValueError(f"Expected {num_floats * 4} bytes, got {len(binary_data)}")
        
        # Convert to numpy array
        all_data = np.frombuffer(binary_data, dtype='<f4').reshape(self.num_vertices, len(self.properties))
        
        # Map to property names
        for i, (prop_name, _) in enumerate(self.properties):
            self.data[prop_name] = all_data[:, i]
        
        print(f"Data shape: {all_data.shape}")
        return self.data
    
    def analyze_gaussian_parameters(self) -> Dict[str, Any]:
        """Analyze Gaussian Splatting specific parameters."""
        print(f"\n{'='*30}")
        print("GAUSSIAN PARAMETERS ANALYSIS")
        print(f"{'='*30}")
        
        analysis = {}
        
        # Position analysis
        if all(key in self.data for key in ['x', 'y', 'z']):
            positions = np.column_stack([self.data['x'], self.data['y'], self.data['z']])
            analysis['positions'] = {
                'mean': positions.mean(axis=0),
                'std': positions.std(axis=0),
                'min': positions.min(axis=0),
                'max': positions.max(axis=0),
                'range': positions.max(axis=0) - positions.min(axis=0)
            }
            print("Positions (x, y, z):")
            print(f"  Mean: [{analysis['positions']['mean'][0]:8.3f}, {analysis['positions']['mean'][1]:8.3f}, {analysis['positions']['mean'][2]:8.3f}]")
            print(f"  Std:  [{analysis['positions']['std'][0]:8.3f}, {analysis['positions']['std'][1]:8.3f}, {analysis['positions']['std'][2]:8.3f}]")
            print(f"  Min:  [{analysis['positions']['min'][0]:8.3f}, {analysis['positions']['min'][1]:8.3f}, {analysis['positions']['min'][2]:8.3f}]")
            print(f"  Max:  [{analysis['positions']['max'][0]:8.3f}, {analysis['positions']['max'][1]:8.3f}, {analysis['positions']['max'][2]:8.3f}]")
            print(f"  Range:[{analysis['positions']['range'][0]:8.3f}, {analysis['positions']['range'][1]:8.3f}, {analysis['positions']['range'][2]:8.3f}]")
        
        # Scale analysis
        scale_keys = [key for key in self.data.keys() if key.startswith('scale_')]
        if scale_keys:
            scales = np.column_stack([self.data[key] for key in sorted(scale_keys)])
            analysis['scales'] = {
                'mean': scales.mean(axis=0),
                'std': scales.std(axis=0),
                'min': scales.min(axis=0),
                'max': scales.max(axis=0),
                'median': np.median(scales, axis=0)
            }
            print(f"\nScales ({len(scale_keys)} components):")
            print(f"  Mean:   [{', '.join(f'{x:8.4f}' for x in analysis['scales']['mean'])}]")
            print(f"  Std:    [{', '.join(f'{x:8.4f}' for x in analysis['scales']['std'])}]")
            print(f"  Min:    [{', '.join(f'{x:8.4f}' for x in analysis['scales']['min'])}]")
            print(f"  Max:    [{', '.join(f'{x:8.4f}' for x in analysis['scales']['max'])}]")
            print(f"  Median: [{', '.join(f'{x:8.4f}' for x in analysis['scales']['median'])}]")
            
            # Scale distribution analysis
            avg_scales = scales.mean(axis=1)
            analysis['scale_distribution'] = {
                'very_small': (avg_scales < 0.001).sum(),
                'small': ((avg_scales >= 0.001) & (avg_scales < 0.01)).sum(),
                'normal': ((avg_scales >= 0.01) & (avg_scales < 0.1)).sum(),
                'large': ((avg_scales >= 0.1) & (avg_scales < 1.0)).sum(),
                'very_large': (avg_scales >= 1.0).sum()
            }
            print(f"  Scale distribution:")
            for category, count in analysis['scale_distribution'].items():
                percentage = (count / self.num_vertices) * 100
                print(f"    {category:10s}: {count:8,} ({percentage:5.1f}%)")
        
        # Rotation analysis (quaternions)
        rot_keys = [key for key in self.data.keys() if key.startswith('rot_')]
        if rot_keys:
            rotations = np.column_stack([self.data[key] for key in sorted(rot_keys)])
            # Calculate quaternion magnitudes
            quat_mags = np.linalg.norm(rotations, axis=1)
            analysis['rotations'] = {
                'mean_magnitude': quat_mags.mean(),
                'std_magnitude': quat_mags.std(),
                'min_magnitude': quat_mags.min(),
                'max_magnitude': quat_mags.max(),
                'near_unit': ((quat_mags > 0.99) & (quat_mags < 1.01)).sum()
            }
            print(f"\nRotations (quaternions):")
            print(f"  Mean magnitude: {analysis['rotations']['mean_magnitude']:.6f}")
            print(f"  Std magnitude:  {analysis['rotations']['std_magnitude']:.6f}")
            print(f"  Min magnitude:  {analysis['rotations']['min_magnitude']:.6f}")
            print(f"  Max magnitude:  {analysis['rotations']['max_magnitude']:.6f}")
            print(f"  Near unit norm: {analysis['rotations']['near_unit']:,} ({analysis['rotations']['near_unit']/self.num_vertices*100:.1f}%)")
        
        # Opacity analysis
        if 'opacity' in self.data:
            opacity = self.data['opacity']
            analysis['opacity'] = {
                'mean': opacity.mean(),
                'std': opacity.std(),
                'min': opacity.min(),
                'max': opacity.max(),
                'median': np.median(opacity)
            }
            print(f"\nOpacity:")
            print(f"  Mean:   {analysis['opacity']['mean']:.6f}")
            print(f"  Std:    {analysis['opacity']['std']:.6f}")
            print(f"  Min:    {analysis['opacity']['min']:.6f}")
            print(f"  Max:    {analysis['opacity']['max']:.6f}")
            print(f"  Median: {analysis['opacity']['median']:.6f}")
            
            # Opacity distribution
            analysis['opacity_distribution'] = {
                'transparent': (opacity < 0.01).sum(),
                'low': ((opacity >= 0.01) & (opacity < 0.1)).sum(),
                'medium': ((opacity >= 0.1) & (opacity < 0.5)).sum(),
                'high': ((opacity >= 0.5) & (opacity < 0.9)).sum(),
                'opaque': (opacity >= 0.9).sum()
            }
            print(f"  Opacity distribution:")
            for category, count in analysis['opacity_distribution'].items():
                percentage = (count / self.num_vertices) * 100
                print(f"    {category:11s}: {count:8,} ({percentage:5.1f}%)")
        
        # Color analysis
        color_keys = [key for key in self.data.keys() if key.startswith('f_dc_')]
        if color_keys:
            colors = np.column_stack([self.data[key] for key in sorted(color_keys)])
            analysis['colors'] = {
                'mean': colors.mean(axis=0),
                'std': colors.std(axis=0),
                'min': colors.min(axis=0),
                'max': colors.max(axis=0)
            }
            print(f"\nColors (DC coefficients):")
            print(f"  Mean: [{', '.join(f'{x:8.4f}' for x in analysis['colors']['mean'])}]")
            print(f"  Std:  [{', '.join(f'{x:8.4f}' for x in analysis['colors']['std'])}]")
            print(f"  Min:  [{', '.join(f'{x:8.4f}' for x in analysis['colors']['min'])}]")
            print(f"  Max:  [{', '.join(f'{x:8.4f}' for x in analysis['colors']['max'])}]")
        
        # Spherical harmonics analysis
        sh_keys = [key for key in self.data.keys() if key.startswith('f_rest_')]
        if sh_keys:
            sh_data = np.column_stack([self.data[key] for key in sorted(sh_keys)])
            analysis['spherical_harmonics'] = {
                'num_coefficients': len(sh_keys),
                'mean_magnitude': np.mean(np.abs(sh_data)),
                'max_magnitude': np.max(np.abs(sh_data)),
                'std_magnitude': np.std(np.abs(sh_data)),
                'near_zero': (np.abs(sh_data) < 1e-6).sum()
            }
            print(f"\nSpherical Harmonics:")
            print(f"  Num coefficients: {analysis['spherical_harmonics']['num_coefficients']}")
            print(f"  Mean |magnitude|: {analysis['spherical_harmonics']['mean_magnitude']:.6f}")
            print(f"  Max |magnitude|:  {analysis['spherical_harmonics']['max_magnitude']:.6f}")
            print(f"  Std |magnitude|:  {analysis['spherical_harmonics']['std_magnitude']:.6f}")
            print(f"  Near zero:        {analysis['spherical_harmonics']['near_zero']:,}")
        
        return analysis
    
    def create_visualizations(self, output_dir: Path):
        """Create visualization plots."""
        output_dir.mkdir(exist_ok=True)
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'PLY Analysis: {self.ply_path.name}', fontsize=16)
        
        # Position scatter plot
        if all(key in self.data for key in ['x', 'y', 'z']):
            axes[0, 0].scatter(self.data['x'], self.data['z'], alpha=0.5, s=0.1)
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Z')
            axes[0, 0].set_title('Position Distribution (Top View)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Scale distribution
        scale_keys = [key for key in self.data.keys() if key.startswith('scale_')]
        if scale_keys:
            scales = np.column_stack([self.data[key] for key in sorted(scale_keys)])
            avg_scales = scales.mean(axis=1)
            axes[0, 1].hist(avg_scales, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Average Scale')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Scale Distribution')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Opacity distribution
        if 'opacity' in self.data:
            axes[0, 2].hist(self.data['opacity'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 2].set_xlabel('Opacity')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].set_title('Opacity Distribution')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Quaternion magnitude distribution
        rot_keys = [key for key in self.data.keys() if key.startswith('rot_')]
        if rot_keys:
            rotations = np.column_stack([self.data[key] for key in sorted(rot_keys)])
            quat_mags = np.linalg.norm(rotations, axis=1)
            axes[1, 0].hist(quat_mags, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Quaternion Magnitude')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Quaternion Magnitude Distribution')
            axes[1, 0].axvline(x=1.0, color='red', linestyle='--', label='Unit norm')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Color distribution
        color_keys = [key for key in self.data.keys() if key.startswith('f_dc_')]
        if color_keys:
            colors = np.column_stack([self.data[key] for key in sorted(color_keys)])
            color_magnitude = np.linalg.norm(colors, axis=1)
            axes[1, 1].hist(color_magnitude, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Color Magnitude')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Color Magnitude Distribution')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Spherical harmonics magnitude
        sh_keys = [key for key in self.data.keys() if key.startswith('f_rest_')]
        if sh_keys:
            sh_data = np.column_stack([self.data[key] for key in sorted(sh_keys)])
            sh_magnitude = np.linalg.norm(sh_data, axis=1)
            axes[1, 2].hist(sh_magnitude, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 2].set_xlabel('SH Magnitude')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_title('Spherical Harmonics Magnitude')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.ply_path.stem}_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_dir / f'{self.ply_path.stem}_analysis.png'}")

def compare_ply_files(ply1_path: Path, ply2_path: Path):
    """Compare two PLY files and highlight differences."""
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    # Analyze both files
    analyzer1 = PLYAnalyzer(ply1_path)
    analyzer2 = PLYAnalyzer(ply2_path)
    
    header1 = analyzer1.parse_header()
    header2 = analyzer2.parse_header()
    
    data1 = analyzer1.load_data()
    data2 = analyzer2.load_data()
    
    analysis1 = analyzer1.analyze_gaussian_parameters()
    analysis2 = analyzer2.analyze_gaussian_parameters()
    
    # Compare key metrics
    print(f"\n{'='*40}")
    print("COMPARISON SUMMARY")
    print(f"{'='*40}")
    
    print(f"{'Metric':<25} {'File 1':<20} {'File 2':<20} {'Ratio':<10}")
    print(f"{'-'*75}")
    
    # Vertex count comparison
    ratio = header2['num_vertices'] / header1['num_vertices'] if header1['num_vertices'] > 0 else 0
    print(f"{'Vertex Count':<25} {header1['num_vertices']:<20,} {header2['num_vertices']:<20,} {ratio:<10.2f}")
    
    # Property count comparison
    ratio = header2['num_properties'] / header1['num_properties'] if header1['num_properties'] > 0 else 0
    print(f"{'Property Count':<25} {header1['num_properties']:<20} {header2['num_properties']:<20} {ratio:<10.2f}")
    
    # File size comparison
    size1 = ply1_path.stat().st_size / (1024*1024)  # MB
    size2 = ply2_path.stat().st_size / (1024*1024)  # MB
    ratio = size2 / size1 if size1 > 0 else 0
    print(f"{'File Size (MB)':<25} {size1:<20.1f} {size2:<20.1f} {ratio:<10.2f}")
    
    # Scale comparison
    if 'scales' in analysis1 and 'scales' in analysis2:
        scale1_avg = np.mean(analysis1['scales']['mean'])
        scale2_avg = np.mean(analysis2['scales']['mean'])
        ratio = scale2_avg / scale1_avg if scale1_avg > 0 else 0
        print(f"{'Average Scale':<25} {scale1_avg:<20.6f} {scale2_avg:<20.6f} {ratio:<10.2f}")
        
        scale1_max = np.max(analysis1['scales']['max'])
        scale2_max = np.max(analysis2['scales']['max'])
        ratio = scale2_max / scale1_max if scale1_max > 0 else 0
        print(f"{'Max Scale':<25} {scale1_max:<20.6f} {scale2_max:<20.6f} {ratio:<10.2f}")
    
    # Opacity comparison
    if 'opacity' in analysis1 and 'opacity' in analysis2:
        ratio = analysis2['opacity']['mean'] / analysis1['opacity']['mean'] if analysis1['opacity']['mean'] > 0 else 0
        print(f"{'Average Opacity':<25} {analysis1['opacity']['mean']:<20.6f} {analysis2['opacity']['mean']:<20.6f} {ratio:<10.2f}")
    
    # Quaternion normalization comparison
    if 'rotations' in analysis1 and 'rotations' in analysis2:
        ratio = analysis2['rotations']['mean_magnitude'] / analysis1['rotations']['mean_magnitude'] if analysis1['rotations']['mean_magnitude'] > 0 else 0
        print(f"{'Quat Magnitude':<25} {analysis1['rotations']['mean_magnitude']:<20.6f} {analysis2['rotations']['mean_magnitude']:<20.6f} {ratio:<10.2f}")
        
        norm1_pct = analysis1['rotations']['near_unit'] / header1['num_vertices'] * 100
        norm2_pct = analysis2['rotations']['near_unit'] / header2['num_vertices'] * 100
        print(f"{'Unit Quaternions %':<25} {norm1_pct:<20.1f} {norm2_pct:<20.1f} {norm2_pct/norm1_pct if norm1_pct > 0 else 0:<10.2f}")
    
    # SH coefficient comparison
    if 'spherical_harmonics' in analysis1 and 'spherical_harmonics' in analysis2:
        ratio = analysis2['spherical_harmonics']['num_coefficients'] / analysis1['spherical_harmonics']['num_coefficients'] if analysis1['spherical_harmonics']['num_coefficients'] > 0 else 0
        print(f"{'SH Coefficients':<25} {analysis1['spherical_harmonics']['num_coefficients']:<20} {analysis2['spherical_harmonics']['num_coefficients']:<20} {ratio:<10.2f}")
        
        ratio = analysis2['spherical_harmonics']['mean_magnitude'] / analysis1['spherical_harmonics']['mean_magnitude'] if analysis1['spherical_harmonics']['mean_magnitude'] > 0 else 0
        print(f"{'SH Magnitude':<25} {analysis1['spherical_harmonics']['mean_magnitude']:<20.6f} {analysis2['spherical_harmonics']['mean_magnitude']:<20.6f} {ratio:<10.2f}")
    
    # Identify potential issues
    print(f"\n{'='*40}")
    print("POTENTIAL RENDERING ISSUES")
    print(f"{'='*40}")
    
    issues = []
    
    # Check for excessive vertex count
    if header2['num_vertices'] > header1['num_vertices'] * 3:
        issues.append(f"File 2 has {header2['num_vertices']/header1['num_vertices']:.1f}x more vertices than File 1")
    
    # Check for quaternion normalization issues
    if 'rotations' in analysis2:
        if analysis2['rotations']['near_unit'] / header2['num_vertices'] < 0.9:
            norm_pct = analysis2['rotations']['near_unit'] / header2['num_vertices'] * 100
            issues.append(f"File 2 has poor quaternion normalization ({norm_pct:.1f}% near unit norm)")
    
    # Check for extreme scales
    if 'scales' in analysis2:
        if np.max(analysis2['scales']['max']) > 10.0:
            issues.append(f"File 2 has very large scales (max: {np.max(analysis2['scales']['max']):.2f})")
        if np.min(analysis2['scales']['min']) < 1e-6:
            issues.append(f"File 2 has very small scales (min: {np.min(analysis2['scales']['min']):.2e})")
    
    # Check SH complexity
    if 'spherical_harmonics' in analysis2 and 'spherical_harmonics' in analysis1:
        if analysis2['spherical_harmonics']['num_coefficients'] > analysis1['spherical_harmonics']['num_coefficients']:
            issues.append(f"File 2 has more complex SH ({analysis2['spherical_harmonics']['num_coefficients']} vs {analysis1['spherical_harmonics']['num_coefficients']} coefficients)")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("No obvious rendering issues detected in the comparison.")
    
    # Create comparative visualizations
    output_dir = Path("ply_analysis")
    analyzer1.create_visualizations(output_dir)
    analyzer2.create_visualizations(output_dir)
    
    print(f"\nAnalysis complete! Check the '{output_dir}' directory for detailed visualizations.")

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare PLY files for rendering issues')
    parser.add_argument('--ply1', type=Path, required=True, help='Path to first PLY file (good rendering)')
    parser.add_argument('--ply2', type=Path, required=True, help='Path to second PLY file (problematic)')
    
    args = parser.parse_args()
    
    if not args.ply1.exists():
        raise FileNotFoundError(f"PLY file not found: {args.ply1}")
    if not args.ply2.exists():
        raise FileNotFoundError(f"PLY file not found: {args.ply2}")
    
    compare_ply_files(args.ply1, args.ply2)

if __name__ == '__main__':
    main()
