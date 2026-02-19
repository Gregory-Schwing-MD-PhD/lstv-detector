#!/usr/bin/env python3
"""
TotalSpineSeg Wrapper - FULL VERSION

Runs TotalSpineSeg (not raw TotalSegmentator) on sagittal T2w and axial T2 NIfTI files.
Uses series CSV to find the correct series per study.

TotalSpineSeg outputs individual vertebra labels (L1-L5/L6, Sacrum) and disc labels,
which are essential for LSTV detection.

NIfTI layout expected (from 01_dicom_to_nifti.py):
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-sag_T2w.nii.gz
  results/nifti/{study_id}/{series_id}/sub-{study_id}_acq-ax_T2w.nii.gz

Usage:
    python 03_run_totalspineseg.py \
        --nifti_dir  results/nifti \
        --series_csv data/raw/train_series_descriptions.csv \
        --output_dir results/totalspineseg \
        --mode trial
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAGITTAL_T2_PATTERNS = [
    'Sagittal T2/STIR',
    'Sagittal T2',
    'SAG T2',
    'Sag T2',
]

AXIAL_T2_PATTERNS = [
    'Axial T2',
    'AXIAL T2',
    'Ax T2',
    'AX T2',
]


# ============================================================================
# SERIES SELECTION VIA CSV
# ============================================================================

def load_series_csv(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from series CSV")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def get_series_id(series_df: pd.DataFrame, study_id: str, patterns: list) -> str | None:
    """Return series_id (str) matching one of the patterns, or None."""
    if series_df is None:
        return None
    try:
        study_rows = series_df[series_df['study_id'] == int(study_id)]
        for pattern in patterns:
            match = study_rows[
                study_rows['series_description'].str.contains(pattern, case=False, na=False)
            ]
            if not match.empty:
                return str(match.iloc[0]['series_id'])
    except Exception as e:
        logger.warning(f"  Series CSV lookup failed for {study_id}: {e}")
    return None


# ============================================================================
# TOTALSPINESEG (PROPER)
# ============================================================================

def run_totalspineseg(nifti_path: Path, study_output_dir: Path, study_id: str, acq: str) -> bool:
    """
    Run TotalSpineSeg (full pipeline) on one NIfTI.
    
    acq: 'sagittal' or 'axial'
    
    TotalSpineSeg outputs:
    - step1_output/: Individual vertebra labels (L1-L5, Sacrum, etc.)
    - step1_levels/: Single voxel markers at each disc level
    - step1_cord/: Spinal cord segmentation
    - step1_canal/: Spinal canal segmentation
    
    Returns True if successful, False otherwise.
    """
    temp_output = study_output_dir / f"temp_{acq}"
    final_dir = study_output_dir / acq
    final_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use full totalspineseg command
        cmd = [
            'totalspineseg',
            str(nifti_path),
            str(temp_output),
            '--step1',  # Step 1 gives vertebra labels + landmarks
        ]

        logger.info(f"  Running TotalSpineSeg ({acq})...")
        sys.stdout.flush()
        
        result = subprocess.run(
            cmd,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,  # 15 min timeout
        )
        sys.stdout.flush()

        if result.returncode != 0:
            logger.error(f"  TotalSpineSeg failed:\n{result.stderr}")
            return False

        # Check for step1_output (has individual labels)
        step1_output_dir = temp_output / 'step1_output'
        if not step1_output_dir.exists():
            logger.error(f"  step1_output not found: {step1_output_dir}")
            return False
        
        # Find the output file
        output_files = list(step1_output_dir.glob("*.nii.gz"))
        if not output_files:
            logger.error(f"  No output files in {step1_output_dir}")
            return False
        
        # Copy step1_output (individual vertebra labels)
        labeled_output = final_dir / f"{study_id}_{acq}_labeled.nii.gz"
        shutil.copy(output_files[0], labeled_output)
        logger.info(f"  ✓ Labeled vertebrae: {labeled_output.name}")
        
        # Copy step1_levels (disc level markers) if exists
        step1_levels_dir = temp_output / 'step1_levels'
        if step1_levels_dir.exists():
            level_files = list(step1_levels_dir.glob("*.nii.gz"))
            if level_files:
                levels_output = final_dir / f"{study_id}_{acq}_levels.nii.gz"
                shutil.copy(level_files[0], levels_output)
                logger.info(f"  ✓ Disc levels: {levels_output.name}")
        
        # Copy step1_cord (spinal cord segmentation) if exists
        step1_cord_dir = temp_output / 'step1_cord'
        if step1_cord_dir.exists():
            cord_files = list(step1_cord_dir.glob("*.nii.gz"))
            if cord_files:
                cord_output = final_dir / f"{study_id}_{acq}_cord.nii.gz"
                shutil.copy(cord_files[0], cord_output)
                logger.info(f"  ✓ Spinal cord: {cord_output.name}")
        
        # Copy step1_canal (spinal canal segmentation) if exists
        step1_canal_dir = temp_output / 'step1_canal'
        if step1_canal_dir.exists():
            canal_files = list(step1_canal_dir.glob("*.nii.gz"))
            if canal_files:
                canal_output = final_dir / f"{study_id}_{acq}_canal.nii.gz"
                shutil.copy(canal_files[0], canal_output)
                logger.info(f"  ✓ Spinal canal: {canal_output.name}")
        
        # Clean up temp directory
        if temp_output.exists():
            shutil.rmtree(temp_output)
        
        return True

    except subprocess.TimeoutExpired:
        logger.error("  TotalSpineSeg timed out (>15min)")
        sys.stdout.flush()
        return False
    except Exception as e:
        logger.error(f"  Error: {e}")
        logger.debug(traceback.format_exc())
        sys.stdout.flush()
        return False
    finally:
        # Cleanup temp directory if it exists
        if temp_output.exists():
            try:
                shutil.rmtree(temp_output)
            except:
                pass


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress(progress_file):
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                p = json.load(f)
            logger.info(f"Resuming: {len(p['success'])} done, {len(p['failed'])} failed")
            return p
        except Exception as e:
            logger.warning(f"Could not load progress: {e} — starting fresh")
    return {'processed': [], 'success': [], 'failed': []}


def save_progress(progress_file, progress):
    try:
        tmp = progress_file.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(progress, f, indent=2)
        tmp.replace(progress_file)
    except Exception as e:
        logger.warning(f"Could not save progress: {e}")


def mark_failed(progress, study_id):
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress['failed']:
        progress['failed'].append(study_id)


def mark_success(progress, study_id):
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress['success']:
        progress['success'].append(study_id)
    if study_id in progress['failed']:
        progress['failed'].remove(study_id)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TotalSpineSeg Segmentation Pipeline')
    parser.add_argument('--nifti_dir',  required=True,
                        help='Root NIfTI directory ({study_id}/{series_id}/sub-*_T2w.nii.gz)')
    parser.add_argument('--series_csv', required=True,
                        help='CSV with study_id, series_id, series_description')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for segmentations')
    parser.add_argument('--valid_ids',  default=None,
                        help='Optional .npy file of study IDs to process')
    parser.add_argument('--limit',      type=int, default=None)
    parser.add_argument('--mode',       choices=['trial', 'debug', 'prod'], default='prod')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()

    nifti_dir     = Path(args.nifti_dir)
    output_dir    = Path(args.output_dir)
    progress_file = output_dir / 'progress.json'

    output_dir.mkdir(parents=True, exist_ok=True)

    series_df = load_series_csv(Path(args.series_csv))
    if series_df is None:
        logger.error("Cannot proceed without series CSV")
        return 1

    progress = load_progress(progress_file)
    skip_ids = (
        set(progress.get('success', []))
        if args.retry_failed
        else set(progress['processed'])
    )

    # Study dirs are the top-level subdirs of nifti_dir
    study_dirs = sorted([d for d in nifti_dir.iterdir() if d.is_dir() and d.name != 'metadata'])

    if args.valid_ids:
        try:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            study_dirs = [d for d in study_dirs if d.name in valid_ids]
            logger.info(f"Filtered to {len(study_dirs)} studies from valid_ids")
        except Exception as e:
            logger.error(f"Failed to load valid_ids: {e}")
            return 1

    if args.mode == 'debug':
        study_dirs = study_dirs[:1]
    elif args.mode == 'trial':
        study_dirs = study_dirs[:3]
    elif args.limit:
        study_dirs = study_dirs[:args.limit]

    study_dirs = [d for d in study_dirs if d.name not in skip_ids]

    logger.info("=" * 70)
    logger.info("TOTALSPINESEG SEGMENTATION (Full Pipeline)")
    logger.info("=" * 70)
    logger.info(f"Mode:        {args.mode}")
    logger.info(f"To process:  {len(study_dirs)}")
    logger.info(f"NIfTI dir:   {nifti_dir}")
    logger.info(f"Output:      {output_dir}")
    logger.info("")
    logger.info("TotalSpineSeg outputs per study:")
    logger.info("  • {acq}_labeled.nii.gz  - Individual vertebra labels")
    logger.info("  • {acq}_levels.nii.gz   - Disc level markers")
    logger.info("  • {acq}_cord.nii.gz     - Spinal cord segmentation")
    logger.info("  • {acq}_canal.nii.gz    - Spinal canal segmentation")
    logger.info("=" * 70)
    sys.stdout.flush()

    success_count = len(progress['success'])
    error_count   = len(progress['failed'])

    for study_dir in tqdm(study_dirs, desc="Studies"):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")
        sys.stdout.flush()
        
        study_output_dir = output_dir / study_id
        study_output_dir.mkdir(parents=True, exist_ok=True)
        any_success = False

        try:
            # ── Sagittal T2w ─────────────────────────────────────────────────
            sag_series_id = get_series_id(series_df, study_id, SAGITTAL_T2_PATTERNS)
            if sag_series_id is None:
                logger.warning("  ⚠ No sagittal T2w series in CSV")
            else:
                # Look for sagittal NIfTI - may have _Eq_1 suffix from dcm2niix
                nifti_path = study_dir / sag_series_id / f"sub-{study_id}_acq-sag_T2w.nii.gz"
                if not nifti_path.exists():
                    # Try with _Eq_1 suffix
                    nifti_path_eq = study_dir / sag_series_id / f"sub-{study_id}_acq-sag_T2w_Eq_1.nii.gz"
                    if nifti_path_eq.exists():
                        nifti_path = nifti_path_eq
                    else:
                        logger.warning(f"  ✗ Sagittal NIfTI not found: {nifti_path}")
                        nifti_path = None
                
                if nifti_path:
                    logger.info(f"  Series (sag): {sag_series_id}")
                    result = run_totalspineseg(nifti_path, study_output_dir, study_id, 'sagittal')
                    if result:
                        any_success = True

            # ── Axial T2 ─────────────────────────────────────────────────────
            ax_series_id = get_series_id(series_df, study_id, AXIAL_T2_PATTERNS)
            if ax_series_id is None:
                logger.warning("  ⚠ No axial T2 series in CSV")
            else:
                # Look for axial NIfTI - may have _Eq_1 suffix from dcm2niix
                nifti_path = study_dir / ax_series_id / f"sub-{study_id}_acq-ax_T2w.nii.gz"
                if not nifti_path.exists():
                    # Try with _Eq_1 suffix
                    nifti_path_eq = study_dir / ax_series_id / f"sub-{study_id}_acq-ax_T2w_Eq_1.nii.gz"
                    if nifti_path_eq.exists():
                        nifti_path = nifti_path_eq
                    else:
                        logger.warning(f"  ✗ Axial NIfTI not found: {nifti_path}")
                        nifti_path = None
                
                if nifti_path:
                    logger.info(f"  Series (ax):  {ax_series_id}")
                    result = run_totalspineseg(nifti_path, study_output_dir, study_id, 'axial')
                    if result:
                        any_success = True

            # ── Progress ──────────────────────────────────────────────────────
            if any_success:
                mark_success(progress, study_id)
                save_progress(progress_file, progress)
                success_count += 1
                logger.info("  ✓ Done")
                sys.stdout.flush()
            else:
                logger.warning("  ✗ No series segmented successfully")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                sys.stdout.flush()

        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved")
            save_progress(progress_file, progress)
            sys.stdout.flush()
            break
        except Exception as e:
            logger.error(f"  ✗ Unexpected error: {e}")
            logger.debug(traceback.format_exc())
            mark_failed(progress, study_id)
            save_progress(progress_file, progress)
            error_count += 1
            sys.stdout.flush()

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info(f"Success:  {success_count}")
    logger.info(f"Failed:   {error_count}")
    logger.info(f"Total:    {success_count + error_count}")
    if progress['failed']:
        logger.info(f"Failed IDs: {progress['failed']}")
    logger.info(f"Progress: {progress_file}")
    logger.info("")
    logger.info("Output structure per study:")
    logger.info(f"  {output_dir}/{{study_id}}/sagittal/")
    logger.info("    ├── {study_id}_sagittal_labeled.nii.gz  (Individual vertebra labels)")
    logger.info("    ├── {study_id}_sagittal_levels.nii.gz   (Disc level markers)")
    logger.info("    ├── {study_id}_sagittal_cord.nii.gz     (Spinal cord)")
    logger.info("    └── {study_id}_sagittal_canal.nii.gz    (Spinal canal)")
    logger.info("")
    logger.info("Next: sbatch slurm_scripts/04_lstv_detection.sh")
    sys.stdout.flush()

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
