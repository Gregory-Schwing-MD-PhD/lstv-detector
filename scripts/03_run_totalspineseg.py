#!/usr/bin/env python3
"""
TotalSpineSeg Wrapper

Runs TotalSegmentator on sagittal and axial T2 images.
Uses the 'total' task which includes vertebrae segmentation.

Usage:
    python 03_run_totalspineseg.py \
        --nifti_dir results/nifti \
        --output_dir results/totalspineseg \
        --mode prod
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_totalseg(nifti_path: Path, output_dir: Path, study_id: str, 
                 series_type: str) -> dict:
    """
    Run TotalSegmentator on a NIfTI file.
    
    Args:
        series_type: 'sagittal' or 'axial'
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # TotalSegmentator command
        # Using 'total' task which includes vertebrae
        cmd = [
            'TotalSegmentator',
            '-i', str(nifti_path),
            '-o', str(output_dir),
            '--task', 'total',
            '--fast',  # Use fast version (slightly less accurate but faster)
            '--ml',    # Multi-label output (single file with all labels)
        ]
        
        logger.info(f"  Running TotalSegmentator ({series_type})...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.error(f"  TotalSegmentator failed:\n{result.stderr}")
            return None
        
        # TotalSegmentator creates segmentations.nii.gz in output_dir
        seg_file = output_dir / 'segmentations.nii.gz'
        
        if not seg_file.exists():
            logger.error(f"  Output file not found: {seg_file}")
            return None
        
        # Rename to include study_id and series type
        final_name = f"{study_id}_{series_type}_vertebrae.nii.gz"
        final_path = output_dir.parent / final_name
        
        import shutil
        shutil.move(str(seg_file), str(final_path))
        
        logger.info(f"  ✓ {series_type.capitalize()} segmentation complete")
        
        return {'segmentation': str(final_path)}
    
    except subprocess.TimeoutExpired:
        logger.error(f"  TotalSegmentator timed out (>600s)")
        return None
    except Exception as e:
        logger.error(f"  Error: {e}")
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
        description='TotalSpineSeg Segmentation Pipeline'
    )
    parser.add_argument('--nifti_dir', required=True,
                       help='Directory with NIfTI files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for segmentations')
    parser.add_argument('--series', choices=['sagittal', 'axial', 'both'],
                       default='both',
                       help='Which series to process')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--mode', choices=['trial', 'debug', 'prod'],
                       default='prod')
    
    args = parser.parse_args()
    
    nifti_dir = Path(args.nifti_dir)
    output_dir = Path(args.output_dir)
    progress_file = output_dir / 'progress.json'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    progress = load_progress(progress_file)
    already_processed = set(progress['processed'])
    
    # Find NIfTI files based on series selection
    study_files = []
    
    if args.series in ['sagittal', 'both']:
        sag_files = sorted(nifti_dir.glob("*_sag_t2.nii.gz"))
        for f in sag_files:
            study_id = f.stem.replace('_sag_t2', '')
            key = f"{study_id}_sagittal"
            if key not in already_processed:
                study_files.append((study_id, f, 'sagittal'))
    
    if args.series in ['axial', 'both']:
        axial_files = sorted(nifti_dir.glob("*_axial_t2.nii.gz"))
        for f in axial_files:
            study_id = f.stem.replace('_axial_t2', '')
            key = f"{study_id}_axial"
            if key not in already_processed:
                study_files.append((study_id, f, 'axial'))
    
    if args.mode == 'debug':
        study_files = study_files[:1]
    elif args.mode == 'trial':
        study_files = study_files[:3]
    elif args.limit:
        study_files = study_files[:args.limit]
    
    logger.info("=" * 70)
    logger.info("TOTALSPINESEG SEGMENTATION")
    logger.info("=" * 70)
    logger.info(f"Mode:       {args.mode}")
    logger.info(f"Series:     {args.series}")
    logger.info(f"To process: {len(study_files)}")
    logger.info(f"Output:     {output_dir}")
    logger.info("=" * 70)
    
    success_count = len(progress['success'])
    error_count = len(progress['failed'])
    
    for study_id, nifti_path, series_type in tqdm(study_files, desc="Segmenting"):
        logger.info(f"\n[{study_id} - {series_type}]")
        
        try:
            # Create series-specific output directory for temp files
            temp_dir = output_dir / f"temp_{study_id}_{series_type}"
            
            result = run_totalseg(nifti_path, temp_dir, study_id, series_type)
            
            # Clean up temp directory
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
            
            if result:
                key = f"{study_id}_{series_type}"
                progress['processed'].append(key)
                progress['success'].append(key)
                save_progress(progress_file, progress)
                success_count += 1
                logger.info(f"  ✓ Done")
            else:
                key = f"{study_id}_{series_type}"
                logger.warning(f"  ✗ Failed")
                progress['processed'].append(key)
                progress['failed'].append(key)
                save_progress(progress_file, progress)
                error_count += 1
        
        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted - progress saved")
            break
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            key = f"{study_id}_{series_type}"
            progress['processed'].append(key)
            progress['failed'].append(key)
            save_progress(progress_file, progress)
            error_count += 1
    
    logger.info("\n" + "=" * 70)
    logger.info("TOTALSPINESEG COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed:  {error_count}")
    logger.info(f"Total:   {success_count + error_count}")
    logger.info("")
    logger.info("Outputs:")
    logger.info(f"  • {output_dir}/*_sagittal_vertebrae.nii.gz")
    logger.info(f"  • {output_dir}/*_axial_vertebrae.nii.gz")
    logger.info("")
    logger.info("Next: python 04_detect_lstv.py")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
