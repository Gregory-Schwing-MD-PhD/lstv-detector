#!/usr/bin/env python3
"""
02b_spineps_selective.py — Run SPINEPS on top-N / bottom-N uncertainty studies
===============================================================================

Reads results/epistemic_uncertainty/lstv_uncertainty_metrics.csv produced by
inference_dicom.py, ranks studies by mean_lstv_prob, selects the top N
(highest LSTV probability) and bottom N (lowest — likely normal controls),
then runs SPINEPS on those studies — skipping any that already have a
completed segmentation in results/spineps/segmentations/.

This lets you start small (N=10) and incrementally increase N without
re-running studies that are already done.

Usage:
    python 02b_spineps_selective.py \\
        --uncertainty_csv results/epistemic_uncertainty/lstv_uncertainty_metrics.csv \\
        --nifti_dir       results/nifti \\
        --spineps_dir     results/spineps \\
        --series_csv      data/raw/train_series_descriptions.csv \\
        --top_n           20 \\
        [--rank_by        mean_lstv_prob]   # column to rank on
        [--dry_run]                         # print selected studies, don't run
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse helpers from 02_run_spineps.py
sys.path.insert(0, str(Path(__file__).parent))
from run_spineps import (   # type: ignore  (adjust import name to match your filename)
    load_series_csv,
    get_sagittal_t2_series,
    run_spineps,
    load_progress,
    save_progress,
    mark_success,
    mark_failed,
    save_metadata,
    compute_all_centroids,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# STUDY SELECTION
# ============================================================================

def select_studies(csv_path: Path,
                   top_n: int,
                   rank_by: str = 'mean_lstv_prob',
                   valid_ids: set | None = None) -> tuple[list[str], pd.DataFrame]:
    """
    Load uncertainty CSV, rank by `rank_by`, return:
      - list of study_ids: top_n highest + top_n lowest (deduplicated)
      - the full ranked DataFrame for logging
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Uncertainty CSV not found: {csv_path}\n"
                                f"Run inference_dicom.py first.")

    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)

    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered CSV to {len(df)} studies via valid_ids ({before - len(df)} excluded)")

    if rank_by not in df.columns:
        available = ', '.join(df.columns.tolist())
        raise ValueError(f"Column '{rank_by}' not in CSV. Available: {available}")

    df_sorted = df.sort_values(rank_by, ascending=False).reset_index(drop=True)

    top_ids    = df_sorted.head(top_n)['study_id'].tolist()
    bottom_ids = df_sorted.tail(top_n)['study_id'].tolist()

    # Deduplicate while preserving order: top first, then bottom
    seen = set()
    selected = []
    for sid in top_ids + bottom_ids:
        if sid not in seen:
            selected.append(sid)
            seen.add(sid)

    logger.info(f"Ranked by:   {rank_by}")
    logger.info(f"Top {top_n}:     {top_ids[:5]}{'...' if top_n > 5 else ''}")
    logger.info(f"Bottom {top_n}: {bottom_ids[:5]}{'...' if top_n > 5 else ''}")
    logger.info(f"Total selected (deduped): {len(selected)}")

    return selected, df_sorted


def already_segmented(study_id: str, spineps_dir: Path) -> bool:
    """
    Return True if a completed SPINEPS segmentation exists for this study.
    A segmentation is considered complete if the instance mask is present.
    """
    instance_mask = (spineps_dir / 'segmentations' / study_id /
                     f"{study_id}_seg-vert_msk.nii.gz")
    return instance_mask.exists()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run SPINEPS on top-N / bottom-N uncertainty studies'
    )
    parser.add_argument('--uncertainty_csv', required=True)
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--spineps_dir',      required=True)
    parser.add_argument('--series_csv',       required=True)
    parser.add_argument('--valid_ids',        default=None,
                        help='.npy file of valid study IDs (e.g. models/valid_id.npy)')
    parser.add_argument('--top_n',            type=int, required=True,
                        help='Number of studies from each end of the ranking')
    parser.add_argument('--rank_by',          default='mean_lstv_prob',
                        help='CSV column to rank studies by (default: mean_lstv_prob)')
    parser.add_argument('--dry_run',          action='store_true',
                        help='Print selected studies without running SPINEPS')
    args = parser.parse_args()

    uncertainty_csv = Path(args.uncertainty_csv)
    nifti_dir       = Path(args.nifti_dir)
    spineps_dir     = Path(args.spineps_dir)
    series_csv      = Path(args.series_csv)
    progress_file   = spineps_dir / 'progress_selective.json'

    seg_dir      = spineps_dir / 'segmentations'
    metadata_dir = spineps_dir / 'metadata'
    seg_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Select studies
    logger.info("=" * 70)
    logger.info(f"SPINEPS SELECTIVE — top/bottom {args.top_n}")
    logger.info("=" * 70)

    valid_ids = None
    if args.valid_ids:
        try:
            import numpy as _np
            valid_ids = set(str(x) for x in _np.load(args.valid_ids))
            logger.info(f"Loaded {len(valid_ids)} valid study IDs from {args.valid_ids}")
        except Exception as e:
            logger.error(f"Failed to load valid_ids: {e}")
            return 1

    selected_ids, df_ranked = select_studies(uncertainty_csv, args.top_n, args.rank_by, valid_ids)

    # Classify each selected study
    skip_already_done = []
    to_run            = []
    for sid in selected_ids:
        if already_segmented(sid, spineps_dir):
            skip_already_done.append(sid)
        else:
            to_run.append(sid)

    logger.info(f"\nSelected:          {len(selected_ids)}")
    logger.info(f"Already done:      {len(skip_already_done)} (skipping)")
    logger.info(f"To run SPINEPS:    {len(to_run)}")

    if skip_already_done:
        logger.info(f"  Skipping: {skip_already_done}")

    if args.dry_run:
        logger.info("\n--- DRY RUN — would process ---")
        for sid in to_run:
            row = df_ranked[df_ranked['study_id'] == sid]
            score = float(row[args.rank_by].iloc[0]) if not row.empty else float('nan')
            logger.info(f"  {sid}  {args.rank_by}={score:.4f}")
        logger.info("--- DRY RUN complete — no SPINEPS was run ---")
        return 0

    if not to_run:
        logger.info("\nAll selected studies already segmented. Nothing to do.")
        logger.info(f"To segment more studies, increase --top_n above {args.top_n}.")
        return 0

    # Load series CSV and progress
    series_df = load_series_csv(series_csv)
    if series_df is None:
        logger.error("Cannot load series CSV — aborting")
        return 1

    progress = load_progress(progress_file)

    success_count = 0
    error_count   = 0

    for study_id in tqdm(to_run, desc='SPINEPS'):
        logger.info(f"\n[{study_id}]")

        # Double-check — another process may have completed it
        if already_segmented(study_id, spineps_dir):
            logger.info(f"  [{study_id}] Segmentation appeared since last check — skipping")
            mark_success(progress, study_id)
            save_progress(progress_file, progress)
            continue

        try:
            # Locate NIfTI (must exist from 01_dicom_to_nifti.py)
            series_id = get_sagittal_t2_series(series_df, study_id)
            if series_id is None:
                logger.warning(f"  [{study_id}] No sagittal T2w series in CSV")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            nifti_path = (nifti_dir / study_id / series_id /
                          f"sub-{study_id}_acq-sag_T2w.nii.gz")
            if not nifti_path.exists():
                logger.warning(f"  [{study_id}] NIfTI not found: {nifti_path}")
                logger.warning(f"  Run 01_dicom_to_nifti.sh first for this study")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            logger.info(f"  Series: {series_id}")

            # Run SPINEPS
            study_seg_dir = seg_dir / study_id
            outputs = run_spineps(nifti_path, study_seg_dir, study_id)

            if outputs is None:
                logger.warning(f"  [{study_id}] SPINEPS failed")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            save_metadata(study_id, outputs, metadata_dir)
            mark_success(progress, study_id)
            save_progress(progress_file, progress)
            success_count += 1

            # Log score for context
            row = df_ranked[df_ranked['study_id'] == study_id]
            if not row.empty:
                score = float(row[args.rank_by].iloc[0])
                logger.info(f"  [{study_id}] ✓  {args.rank_by}={score:.4f}")

        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved")
            save_progress(progress_file, progress)
            break
        except Exception as e:
            logger.error(f"  [{study_id}] ✗ Unexpected: {e}")
            logger.debug(traceback.format_exc())
            mark_failed(progress, study_id)
            save_progress(progress_file, progress)
            error_count += 1

    logger.info("\n" + "=" * 70)
    logger.info("SELECTIVE SPINEPS DONE")
    logger.info(f"Success:       {success_count}")
    logger.info(f"Failed:        {error_count}")
    logger.info(f"Already done:  {len(skip_already_done)}")
    if progress.get('failed'):
        logger.info(f"Failed IDs:    {progress['failed']}")
    logger.info(f"\nTo add more studies, re-run with a larger --top_n")
    logger.info(f"Already-done studies will be skipped automatically.")
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
