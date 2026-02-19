#!/usr/bin/env python3
"""
SPINEPS Segmentation Pipeline - Refactored

Runs SPINEPS on pre-converted NIfTI files.
Computes centroids for ALL structures and generates uncertainty maps.

Usage:
    python 02_run_spineps.py \
        --nifti_dir results/nifti \
        --output_dir results/spineps \
        --mode prod
"""

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import logging

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_all_centroids(instance_mask_path: Path, semantic_mask_path: Path, 
                          ctd_path: Path) -> dict:
    """Compute centroids for ALL structures."""
    if not HAS_NIBABEL:
        return {}
    
    try:
        instance_nii = nib.load(instance_mask_path)
        instance_data = instance_nii.get_fdata().astype(int)
        
        semantic_nii = nib.load(semantic_mask_path)
        semantic_data = semantic_nii.get_fdata().astype(int)
        
        with open(ctd_path) as f:
            ctd_data = json.load(f)
        
        if len(ctd_data) < 2:
            return {}
        
        added_counts = {
            'vertebrae': 0, 'discs': 0, 'endplates': 0, 'subregions': 0
        }
        
        # Instance mask (vertebrae, discs, endplates)
        for label in np.unique(instance_data):
            if label == 0:
                continue
            label_str = str(label)
            if label_str in ctd_data[1]:
                continue
            
            mask = (instance_data == label)
            if mask.sum() == 0:
                continue
            
            centroid = center_of_mass(mask)
            ctd_data[1][label_str] = {'50': list(centroid)}
            
            if label <= 28:
                added_counts['vertebrae'] += 1
            elif 119 <= label <= 126:
                added_counts['discs'] += 1
            elif label >= 200:
                added_counts['endplates'] += 1
        
        # Semantic mask (subregions)
        for label in np.unique(semantic_data):
            if label == 0:
                continue
            label_str = str(label)
            if label_str in ctd_data[1]:
                continue
            
            mask = (semantic_data == label)
            if mask.sum() == 0:
                continue
            
            centroid = center_of_mass(mask)
            ctd_data[1][label_str] = {'50': list(centroid)}
            added_counts['subregions'] += 1
        
        with open(ctd_path, 'w') as f:
            json.dump(ctd_data, f, indent=2)
        
        return added_counts
    
    except Exception as e:
        logger.warning(f"Error computing centroids: {e}")
        return {}


def compute_uncertainty_from_softmax(derivatives_dir: Path, study_id: str, 
                                     seg_dir: Path) -> bool:
    """Compute uncertainty map from softmax logits."""
    if not HAS_NIBABEL:
        return False
    
    try:
        logits_pattern = f"*{study_id}*logit*.npz"
        logits_files = list(derivatives_dir.glob(logits_pattern))
        
        if not logits_files:
            return False
        
        logits_data = np.load(logits_files[0])
        softmax = logits_data['arr_0']
        
        uncertainty = 1.0 - np.max(softmax, axis=-1)
        
        semantic_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
        if not semantic_mask.exists():
            return False
        
        ref_nii = nib.load(semantic_mask)
        unc_nii = nib.Nifti1Image(
            uncertainty.astype(np.float32), 
            ref_nii.affine, 
            ref_nii.header
        )
        
        unc_path = seg_dir / f"{study_id}_unc.nii.gz"
        nib.save(unc_nii, unc_path)
        
        return True
    
    except Exception as e:
        logger.debug(f"Could not compute uncertainty: {e}")
        return False


def run_spineps(nifti_path: Path, seg_dir: Path, study_id: str) -> dict:
    """Run SPINEPS segmentation."""
    try:
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        import os
        env = os.environ.copy()
        env['SPINEPS_SEGMENTOR_MODELS'] = '/app/models'
        env['SPINEPS_ENVIRONMENT_DIR'] = '/app/models'
        
        cmd = [
            'python', '-m', 'spineps.entrypoint', 'sample',
            '-i', str(nifti_path),
            '-model_semantic', 't2w',
            '-model_instance', 'instance',
            '-model_labeling', 't2w_labeling',
            '-save_softmax_logits',
            '-override_semantic',
            '-override_instance',
            '-override_ctd'
        ]
        
        logger.info("  Running SPINEPS...")
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=600, 
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"  SPINEPS failed:\n{result.stderr}")
            return None
        
        derivatives_base = nifti_path.parent / "derivatives_seg"
        if not derivatives_base.exists():
            logger.error(f"  derivatives_seg not found")
            return None
        
        def find_file(glob_pattern: str) -> Path:
            matches = list(derivatives_base.glob(glob_pattern))
            return matches[0] if matches else None
        
        outputs = {}
        
        # Instance mask
        f = find_file("*_seg-vert_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['instance_mask'] = dest
            logger.info("  ✓ Instance mask")
        
        # Semantic mask
        f = find_file("*_seg-spine_msk.nii.gz")
        if f:
            dest = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
            shutil.copy(f, dest)
            outputs['semantic_mask'] = dest
            logger.info("  ✓ Semantic mask")
        
        # Centroids
        f = find_file("*_ctd.json")
        if f:
            dest = seg_dir / f"{study_id}_ctd.json"
            shutil.copy(f, dest)
            outputs['centroid_json'] = dest
            logger.info("  ✓ Centroids")
            
            # Add ALL centroids
            if 'instance_mask' in outputs and 'semantic_mask' in outputs:
                counts = compute_all_centroids(
                    outputs['instance_mask'],
                    outputs['semantic_mask'],
                    dest
                )
                if counts:
                    total = sum(counts.values())
                    logger.info(f"  ✓ Added {total} centroids: "
                              f"{counts.get('discs', 0)} discs, "
                              f"{counts.get('endplates', 0)} endplates, "
                              f"{counts.get('subregions', 0)} subregions")
        
        # Uncertainty map
        if 'semantic_mask' in outputs:
            if compute_uncertainty_from_softmax(derivatives_base, study_id, seg_dir):
                outputs['uncertainty_map'] = seg_dir / f"{study_id}_unc.nii.gz"
                logger.info("  ✓ Uncertainty map")
        
        return outputs if outputs else None
    
    except Exception as e:
        logger.error(f"  SPINEPS error: {e}")
        return None


def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'processed': [], 'success': [], 'failed': []}


def save_progress(progress_file: Path, progress: dict):
    try:
        tmp = progress_file.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(progress, f, indent=2)
        tmp.replace(progress_file)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='SPINEPS Segmentation Pipeline'
    )
    parser.add_argument('--nifti_dir', required=True,
                       help='Directory with NIfTI files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for segmentations')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--mode', choices=['trial', 'debug', 'prod'], 
                       default='prod')
    
    args = parser.parse_args()
    
    nifti_dir = Path(args.nifti_dir)
    output_dir = Path(args.output_dir)
    seg_dir = output_dir / 'segmentations'
    progress_file = output_dir / 'progress.json'
    
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    progress = load_progress(progress_file)
    already_processed = set(progress['processed'])
    
    # Find sagittal T2 NIfTI files
    nifti_files = sorted(nifti_dir.glob("*_sag_t2.nii.gz"))
    
    if args.mode == 'debug':
        nifti_files = nifti_files[:1]
    elif args.mode == 'trial':
        nifti_files = nifti_files[:3]
    elif args.limit:
        nifti_files = nifti_files[:args.limit]
    
    # Extract study IDs and filter
    study_files = []
    for f in nifti_files:
        study_id = f.stem.replace('_sag_t2', '')
        if study_id not in already_processed:
            study_files.append((study_id, f))
    
    logger.info("=" * 70)
    logger.info("SPINEPS SEGMENTATION")
    logger.info("=" * 70)
    logger.info(f"Mode:         {args.mode}")
    logger.info(f"Total:        {len(nifti_files)}")
    logger.info(f"Already done: {len(nifti_files) - len(study_files)}")
    logger.info(f"To process:   {len(study_files)}")
    logger.info(f"Output:       {output_dir}")
    logger.info("=" * 70)
    
    success_count = len(progress['success'])
    error_count = len(progress['failed'])
    
    for study_id, nifti_path in tqdm(study_files, desc="Segmenting"):
        logger.info(f"\n[{study_id}]")
        
        try:
            outputs = run_spineps(nifti_path, seg_dir, study_id)
            
            if outputs:
                progress['processed'].append(study_id)
                progress['success'].append(study_id)
                save_progress(progress_file, progress)
                success_count += 1
                logger.info(f"  ✓ Done ({len(outputs)} outputs)")
            else:
                logger.warning(f"  ✗ Failed")
                progress['processed'].append(study_id)
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                error_count += 1
        
        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted - progress saved")
            break
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            progress['processed'].append(study_id)
            progress['failed'].append(study_id)
            save_progress(progress_file, progress)
            error_count += 1
    
    logger.info("\n" + "=" * 70)
    logger.info("SPINEPS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed:  {error_count}")
    logger.info(f"Total:   {success_count + error_count}")
    logger.info("")
    logger.info("Outputs per study:")
    logger.info(f"  • {seg_dir}/*_seg-vert_msk.nii.gz")
    logger.info(f"  • {seg_dir}/*_seg-spine_msk.nii.gz")
    logger.info(f"  • {seg_dir}/*_ctd.json (ALL structures)")
    logger.info(f"  • {seg_dir}/*_unc.nii.gz")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
