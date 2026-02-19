#!/usr/bin/env python3
"""
Visualize SPINEPS masks overlaid on original DICOM/NIfTI

Usage:
    python visualize_overlay.py \
        --nifti results/spineps_segmentation/nifti/1020394063_T2w.nii.gz \
        --instance results/spineps_segmentation/segmentations/1020394063_seg-vert_msk.nii.gz \
        --semantic results/spineps_segmentation/segmentations/1020394063_seg-spine_msk.nii.gz \
        --output overlay_1020394063.png
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path


def create_overlay(nifti_path, instance_path, semantic_path, output_path, slice_idx=None):
    """
    Create overlay visualization of masks on original image.
    """
    # Load data
    img_nii = nib.load(nifti_path)
    img_data = img_nii.get_fdata()
    
    inst_nii = nib.load(instance_path)
    inst_data = inst_nii.get_fdata().astype(int)
    
    sem_nii = nib.load(semantic_path)
    sem_data = sem_nii.get_fdata().astype(int)
    
    # Get mid-sagittal slice if not specified
    if slice_idx is None:
        slice_idx = img_data.shape[2] // 2
    
    # Extract slices
    img_slice = img_data[:, :, slice_idx]
    inst_slice = inst_data[:, :, slice_idx]
    sem_slice = sem_data[:, :, slice_idx]
    
    # Normalize image for display
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. Original image
    axes[0, 0].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. Instance mask overlay (vertebrae, discs, endplates)
    axes[0, 1].imshow(img_slice.T, cmap='gray', origin='lower', alpha=0.7)
    
    # Color vertebrae (19-28) in blue gradient
    vert_mask = (inst_slice >= 19) & (inst_slice <= 28)
    if vert_mask.any():
        vert_overlay = np.zeros((*inst_slice.shape, 4))
        vert_overlay[vert_mask] = [0, 0, 1, 0.5]  # Blue, 50% transparent
        axes[0, 1].imshow(vert_overlay.T, origin='lower')
    
    # Color discs (119-126) in green
    disc_mask = (inst_slice >= 119) & (inst_slice <= 126)
    if disc_mask.any():
        disc_overlay = np.zeros((*inst_slice.shape, 4))
        disc_overlay[disc_mask] = [0, 1, 0, 0.7]  # Green, 70% transparent
        axes[0, 1].imshow(disc_overlay.T, origin='lower')
    
    # Color endplates (200+) in yellow
    endplate_mask = inst_slice >= 200
    if endplate_mask.any():
        endplate_overlay = np.zeros((*inst_slice.shape, 4))
        endplate_overlay[endplate_mask] = [1, 1, 0, 0.5]  # Yellow, 50% transparent
        axes[0, 1].imshow(endplate_overlay.T, origin='lower')
    
    axes[0, 1].set_title('Instance Mask Overlay\n(Blue: Vertebrae, Green: Discs, Yellow: Endplates)', fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. Semantic mask overlay (subregions)
    axes[1, 0].imshow(img_slice.T, cmap='gray', origin='lower', alpha=0.7)
    
    # Highlight costal processes (43/44) in RED - critical for LSTV Type I
    costal_mask = ((sem_slice == 43) | (sem_slice == 44))
    if costal_mask.any():
        costal_overlay = np.zeros((*sem_slice.shape, 4))
        costal_overlay[costal_mask] = [1, 0, 0, 0.8]  # Bright red, 80% transparent
        axes[1, 0].imshow(costal_overlay.T, origin='lower')
    
    # Show other subregions in cyan
    other_sem_mask = (sem_slice > 0) & ~costal_mask
    if other_sem_mask.any():
        other_overlay = np.zeros((*sem_slice.shape, 4))
        other_overlay[other_sem_mask] = [0, 1, 1, 0.3]  # Cyan, 30% transparent
        axes[1, 0].imshow(other_overlay.T, origin='lower')
    
    axes[1, 0].set_title('Semantic Mask Overlay\n(RED: Costal Processes - Critical for Type I!)', fontsize=14)
    axes[1, 0].axis('off')
    
    # 4. Combined overlay
    axes[1, 1].imshow(img_slice.T, cmap='gray', origin='lower', alpha=0.6)
    
    # Show vertebrae in faint blue
    if vert_mask.any():
        axes[1, 1].imshow(vert_overlay.T, origin='lower', alpha=0.3)
    
    # Show discs in green
    if disc_mask.any():
        axes[1, 1].imshow(disc_overlay.T, origin='lower', alpha=0.5)
    
    # Show costal processes in bright red
    if costal_mask.any():
        axes[1, 1].imshow(costal_overlay.T, origin='lower', alpha=0.8)
    
    axes[1, 1].set_title('Combined Overlay\n(Focus: Costal Processes for LSTV Detection)', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved overlay to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize SPINEPS masks on original image')
    parser.add_argument('--nifti', required=True, help='Original NIfTI file')
    parser.add_argument('--instance', required=True, help='Instance mask (seg-vert)')
    parser.add_argument('--semantic', required=True, help='Semantic mask (seg-spine)')
    parser.add_argument('--output', required=True, help='Output PNG file')
    parser.add_argument('--slice', type=int, default=None, help='Slice index (default: middle)')
    
    args = parser.parse_args()
    
    create_overlay(
        Path(args.nifti),
        Path(args.instance),
        Path(args.semantic),
        Path(args.output),
        args.slice
    )


if __name__ == '__main__':
    main()
