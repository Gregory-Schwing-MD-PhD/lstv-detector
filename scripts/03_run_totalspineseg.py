#!/usr/bin/env python3
"""
TotalSpineSeg Wrapper

Runs TotalSegmentator on sagittal T2w and axial T2 NIfTI files.
Uses series CSV to find the correct series per study.

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

def load_series_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from series CSV")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def get_series_id(series_df: pd.DataFrame, study_id: str, patterns: list) -> str | None:
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
# TOTALSEGMENTATOR
# ============================================================================

def run_totalseg(nifti_path: Path, output_dir: Path, study_id: str, acq: str) -> Path | None:
    """
    Run TotalSegmentator on one NIfTI.
    acq: 'sag' or 'ax'
    Returns path to final segmentation file, or None on failure.
    """
    temp_dir = output_dir / f"temp_{study_id}_{acq}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            'TotalSegmentator',
            '-i', str(nifti_path),
            '-o', str(temp_dir),
            '--task', 'total',
            '--fast',
            '--ml',
        ]

        logger.info(f"  Running TotalSegmentator (acq-{acq})...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"  TotalSegmentator failed:\n{result.stderr}")
            return None

        seg_file = temp_dir / 'segmentations.nii.gz'
        if not seg_file.exists():
            logger.error(f"  Output file not found: {seg_file}")
            return None

        final_path = output_dir / f"{study_id}_acq-{acq}_seg-total_msk.nii.gz"
        shutil.move(str(seg_file), str(final_path))
        logger.info(f"  ✓ Saved -> {final_path.name}")
        return final_path

    except subprocess.TimeoutExpired:
        logger.error("  TotalSegmentator timed out (>600s)")
        return None
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                p = json.load(f)
            logger.info(f"Resuming: {len(p['success'])} done, {len(p['failed'])} failed")
            return p
        except Exception as e:
            logger.warning(f"Could not load progress: {e} — starting fresh")
    return {'processed': [], 'success': [], 'failed': []}


def save_progress(progress_file: Path, progress: dict):
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

    study_dirs = sorted([d for d in nifti_dir.iterdir() if d.is_dir()])

    if args.valid_ids:
        try:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            study_dirs = [d for d in study_dirs if d.name in valid_ids]
            logger.info(f"Filtered to {len(study_dirs)} studies from valid_ids")
        except Exception as e:
            logger.error(f"Failed to load valid_ids: {e}")
            return 1

    # Filter already-processed BEFORE applying mode limits
    study_dirs = [d for d in study_dirs if d.name not in skip_ids]

    if args.mode == 'debug':
        study_dirs = study_dirs[:1]
    elif args.mode == 'trial':
        study_dirs = study_dirs[:3]
    elif args.limit:
        study_dirs = study_dirs[:args.limit]

    logger.info("=" * 70)
    logger.info("TOTALSPINESEG SEGMENTATION")
    logger.info("=" * 70)
    logger.info(f"Mode:        {args.mode}")
    logger.info(f"To process:  {len(study_dirs)}")
    logger.info(f"NIfTI dir:   {nifti_dir}")
    logger.info(f"Output:      {output_dir}")
    logger.info("=" * 70)

    success_count = len(progress['success'])
    error_count   = len(progress['failed'])

    for study_dir in tqdm(study_dirs, desc="Segmenting"):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")
        study_output_dir = output_dir / study_id
        study_output_dir.mkdir(parents=True, exist_ok=True)
        any_success = False

        try:
            # ── Sagittal T2w ─────────────────────────────────────────────────
            sag_series_id = get_series_id(series_df, study_id, SAGITTAL_T2_PATTERNS)
            if sag_series_id is None:
                logger.warning("  ⚠ No sagittal T2w series in CSV")
            else:
                nifti_path = study_dir / sag_series_id / f"sub-{study_id}_acq-sag_T2w.nii.gz"
                if not nifti_path.exists():
                    logger.warning(f"  ⚠ Sagittal NIfTI not found: {nifti_path}")
                else:
                    result = run_totalseg(nifti_path, study_output_dir, study_id, 'sag')
                    if result:
                        any_success = True

            # ── Axial T2 ─────────────────────────────────────────────────────
            ax_series_id = get_series_id(series_df, study_id, AXIAL_T2_PATTERNS)
            if ax_series_id is None:
                logger.warning("  ⚠ No axial T2 series in CSV")
            else:
                nifti_path = study_dir / ax_series_id / f"sub-{study_id}_acq-ax_T2w.nii.gz"
                if not nifti_path.exists():
                    logger.warning(f"  ⚠ Axial NIfTI not found: {nifti_path}")
                else:
                    result = run_totalseg(nifti_path, study_output_dir, study_id, 'ax')
                    if result:
                        any_success = True

            # ── Progress ──────────────────────────────────────────────────────
            if any_success:
                mark_success(progress, study_id)
                save_progress(progress_file, progress)
                success_count += 1
                logger.info("  ✓ Done")
            else:
                logger.warning("  ✗ No series segmented successfully")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1

        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved")
            save_progress(progress_file, progress)
            break
        except Exception as e:
            logger.error(f"  ✗ Unexpected error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            mark_failed(progress, study_id)
            save_progress(progress_file, progress)
            error_count += 1

    logger.info("\n" + "=" * 70)
    logger.info("TOTALSPINESEG COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed:  {error_count}")
    logger.info(f"Total:   {success_count + error_count}")
    if progress['failed']:
        logger.info(f"Failed IDs: {progress['failed']}")
    logger.info(f"\nOutputs: {output_dir}/{{study_id}}/{{study_id}}_acq-[sag|ax]_seg-total_msk.nii.gz")
    logger.info("Next: sbatch slurm_scripts/04_detect_lstv.sh")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
