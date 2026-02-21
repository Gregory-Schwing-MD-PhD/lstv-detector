#!/usr/bin/env python3
"""
02b_spineps_selective.py — Run SPINEPS on top-N / bottom-N uncertainty studies
===============================================================================

Reads results/epistemic_uncertainty/lstv_uncertainty_metrics.csv, ranks studies
by the given column, selects the top N (highest) and bottom N (lowest), then
runs SPINEPS on those studies — skipping any already segmented.

Fully self-contained — no imports from other local scripts.

Usage:
    python scripts/02b_spineps_selective.py \
        --uncertainty_csv results/epistemic_uncertainty/lstv_uncertainty_metrics.csv \
        --nifti_dir       results/nifti \
        --spineps_dir     results/spineps \
        --series_csv      data/raw/train_series_descriptions.csv \
        --valid_ids       models/valid_id.npy \
        --top_n           1 \
        --rank_by         l5_s1_confidence \
        [--dry_run]
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from tqdm import tqdm

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

SAGITTAL_T2_PATTERNS = [
    'Sagittal T2/STIR',
    'Sagittal T2',
    'SAG T2',
    'Sag T2',
]


# ============================================================================
# SERIES CSV
# ============================================================================

def load_series_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from series CSV")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def get_sagittal_t2_series(series_df: pd.DataFrame, study_id: str) -> str | None:
    if series_df is None:
        return None
    try:
        study_rows = series_df[series_df['study_id'] == int(study_id)]
        for pattern in SAGITTAL_T2_PATTERNS:
            match = study_rows[
                study_rows['series_description'].str.contains(pattern, case=False, na=False)
            ]
            if not match.empty:
                return str(match.iloc[0]['series_id'])
    except Exception as e:
        logger.warning(f"  Series CSV lookup failed for {study_id}: {e}")
    return None


# ============================================================================
# PROGRESS
# ============================================================================

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                p = json.load(f)
            logger.info(f"Resuming: {len(p.get('success', []))} done, "
                        f"{len(p.get('failed', []))} failed")
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


def mark_success(progress: dict, study_id: str):
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress['success']:
        progress['success'].append(study_id)
    if study_id in progress.get('failed', []):
        progress['failed'].remove(study_id)


def mark_failed(progress: dict, study_id: str):
    if study_id not in progress['processed']:
        progress['processed'].append(study_id)
    if study_id not in progress.get('failed', []):
        progress.setdefault('failed', []).append(study_id)


# ============================================================================
# METADATA
# ============================================================================

def save_metadata(study_id: str, outputs: dict, metadata_dir: Path):
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'study_id':  study_id,
        'outputs':   {k: str(v) for k, v in outputs.items()},
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    with open(metadata_dir / f"{study_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


# ============================================================================
# CENTROIDS
# ============================================================================

def compute_all_centroids(instance_mask_path, semantic_mask_path, ctd_path):
    if not HAS_NIBABEL:
        return {}
    try:
        instance_data = nib.load(instance_mask_path).get_fdata().astype(int)
        semantic_data = nib.load(semantic_mask_path).get_fdata().astype(int)

        with open(ctd_path) as f:
            ctd_data = json.load(f)

        if len(ctd_data) < 2:
            return {}

        counts = {'vertebrae': 0, 'discs': 0, 'endplates': 0, 'subregions': 0}

        for label in np.unique(instance_data)[1:]:
            ls = str(label)
            if ls in ctd_data[1]:
                continue
            mask = instance_data == label
            ctd_data[1][ls] = {'50': list(center_of_mass(mask))}
            if label <= 28:
                counts['vertebrae'] += 1
            elif 119 <= label <= 126:
                counts['discs'] += 1
            elif label >= 200:
                counts['endplates'] += 1

        for label in np.unique(semantic_data)[1:]:
            ls = str(label)
            if ls in ctd_data[1]:
                continue
            mask = semantic_data == label
            ctd_data[1][ls] = {'50': list(center_of_mass(mask))}
            counts['subregions'] += 1

        with open(ctd_path, 'w') as f:
            json.dump(ctd_data, f, indent=2)

        return counts

    except Exception as e:
        logger.warning(f"  Centroid computation error: {e}")
        return {}


# ============================================================================
# UNCERTAINTY MAP
# ============================================================================

def compute_uncertainty_from_softmax(derivatives_dir: Path, study_id: str,
                                      seg_dir: Path) -> bool:
    if not HAS_NIBABEL:
        return False
    try:
        logits_files = list(derivatives_dir.glob(f"**/*{study_id}*logit*.npz"))
        if not logits_files:
            return False

        softmax     = np.load(logits_files[0])['arr_0']  # (H, W, D, C)
        epsilon     = 1e-8
        entropy     = -np.sum(softmax * np.log(softmax + epsilon), axis=-1)
        num_classes = softmax.shape[-1]
        uncertainty = (entropy / np.log(num_classes)).astype(np.float32)

        semantic_mask = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
        if not semantic_mask.exists():
            return False

        ref = nib.load(semantic_mask)
        nib.save(
            nib.Nifti1Image(uncertainty, ref.affine, ref.header),
            seg_dir / f"{study_id}_unc.nii.gz",
        )
        logger.info("  ✓ Uncertainty map saved")
        return True

    except Exception as e:
        logger.warning(f"  Uncertainty map failed: {e}")
        return False


# ============================================================================
# SPINEPS
# ============================================================================

def run_spineps(nifti_path: Path, seg_dir: Path, study_id: str) -> dict | None:
    seg_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['SPINEPS_SEGMENTOR_MODELS'] = '/app/models'
    env['SPINEPS_ENVIRONMENT_DIR']  = '/app/models'

    cmd = [
        'python', '-m', 'spineps.entrypoint', 'sample',
        '-i', str(nifti_path),
        '-model_semantic',  't2w',
        '-model_instance',  'instance',
        '-model_labeling',  't2w_labeling',
        '-save_softmax_logits',
        '-override_semantic',
        '-override_instance',
        '-override_ctd',
    ]

    logger.info("  Running SPINEPS...")
    try:
        result = subprocess.run(
            cmd, stderr=subprocess.PIPE, text=True, timeout=600, env=env
        )
    except subprocess.TimeoutExpired:
        logger.error("  SPINEPS timed out (>600s)")
        return None
    except Exception as e:
        logger.error(f"  SPINEPS error: {e}")
        return None

    if result.returncode != 0:
        logger.error(f"  SPINEPS non-zero exit:\n{result.stderr}")
        return None

    derivatives_base = nifti_path.parent / "derivatives_seg"
    if not derivatives_base.exists():
        logger.error(f"  derivatives_seg not found at: {derivatives_base}")
        return None

    def find_file(exact, glob_pat):
        f = derivatives_base / exact
        if f.exists():
            return f
        hits = list(derivatives_base.glob(glob_pat))
        return hits[0] if hits else None

    outputs = {}

    f = find_file(f"sub-{study_id}_acq-sag_mod-T2w_seg-vert_msk.nii.gz",
                  "**/*_seg-vert_msk.nii.gz")
    if f:
        dest = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
        shutil.copy(f, dest)
        outputs['instance_mask'] = dest
        logger.info("  ✓ Instance mask")
    else:
        logger.warning("  ⚠ Instance mask not found")

    f = find_file(f"sub-{study_id}_acq-sag_mod-T2w_seg-spine_msk.nii.gz",
                  "**/*_seg-spine_msk.nii.gz")
    if f:
        dest = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
        shutil.copy(f, dest)
        outputs['semantic_mask'] = dest
        logger.info("  ✓ Semantic mask")

    f = find_file(f"sub-{study_id}_acq-sag_mod-T2w_seg-subreg_msk.nii.gz",
                  "**/*_seg-subreg_msk.nii.gz")
    if f:
        dest = seg_dir / f"{study_id}_seg-subreg_msk.nii.gz"
        shutil.copy(f, dest)
        outputs['subreg_mask'] = dest
        logger.info("  ✓ Sub-region mask")

    f = find_file(f"sub-{study_id}_acq-sag_mod-T2w_ctd.json", "**/*_ctd.json")
    if f:
        dest = seg_dir / f"{study_id}_ctd.json"
        shutil.copy(f, dest)
        outputs['centroid_json'] = dest
        logger.info("  ✓ Centroids JSON")

        if 'instance_mask' in outputs and 'semantic_mask' in outputs:
            counts = compute_all_centroids(
                outputs['instance_mask'], outputs['semantic_mask'], dest
            )
            if counts:
                total = sum(counts.values())
                logger.info(f"  ✓ Added {total} centroids: "
                            f"{counts['discs']} discs, "
                            f"{counts['endplates']} endplates, "
                            f"{counts['subregions']} subregions")

    if 'semantic_mask' in outputs:
        if compute_uncertainty_from_softmax(derivatives_base, study_id, seg_dir):
            outputs['uncertainty_map'] = seg_dir / f"{study_id}_unc.nii.gz"

    if 'instance_mask' not in outputs:
        logger.error("  Instance mask missing — treating as failure")
        return None

    return outputs


# ============================================================================
# STUDY SELECTION
# ============================================================================

def select_studies(csv_path: Path, top_n: int, rank_by: str,
                   valid_ids: set | None) -> tuple[list[str], pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Uncertainty CSV not found: {csv_path}\n"
            f"Run inference_dicom.py (00_ian_pan_inference.sh) first."
        )

    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)

    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered CSV to {len(df)} studies via valid_ids "
                    f"({before - len(df)} excluded)")

    if rank_by not in df.columns:
        raise ValueError(
            f"Column '{rank_by}' not in CSV.\nAvailable: {', '.join(df.columns)}"
        )

    df_sorted  = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top_ids    = df_sorted.head(top_n)['study_id'].tolist()
    bottom_ids = df_sorted.tail(top_n)['study_id'].tolist()

    seen, selected = set(), []
    for sid in top_ids + bottom_ids:
        if sid not in seen:
            selected.append(sid)
            seen.add(sid)

    logger.info(f"Rank by:          {rank_by}")
    logger.info(f"Top {top_n}:           {top_ids}")
    logger.info(f"Bottom {top_n}:        {bottom_ids}")
    logger.info(f"Total (deduped):  {len(selected)}")

    return selected, df_sorted


def already_segmented(study_id: str, spineps_dir: Path) -> bool:
    return (spineps_dir / 'segmentations' / study_id /
            f"{study_id}_seg-vert_msk.nii.gz").exists()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run SPINEPS on top-N / bottom-N uncertainty studies'
    )
    parser.add_argument('--uncertainty_csv', required=True,
                        help='results/epistemic_uncertainty/lstv_uncertainty_metrics.csv')
    parser.add_argument('--nifti_dir',       required=True,
                        help='results/nifti')
    parser.add_argument('--spineps_dir',     required=True,
                        help='results/spineps')
    parser.add_argument('--series_csv',      required=True,
                        help='data/raw/train_series_descriptions.csv')
    parser.add_argument('--valid_ids',       default=None,
                        help='models/valid_id.npy')
    parser.add_argument('--top_n',           type=int, required=True,
                        help='Studies from each end of the ranking')
    parser.add_argument('--rank_by',         default='l5_s1_confidence',
                        help='CSV column to rank by (default: l5_s1_confidence)')
    parser.add_argument('--dry_run',         action='store_true',
                        help='Print selected studies without running SPINEPS')
    args = parser.parse_args()

    uncertainty_csv = Path(args.uncertainty_csv)
    nifti_dir       = Path(args.nifti_dir)
    spineps_dir     = Path(args.spineps_dir)
    series_csv      = Path(args.series_csv)
    progress_file   = spineps_dir / 'progress_selective.json'
    seg_dir         = spineps_dir / 'segmentations'
    metadata_dir    = spineps_dir / 'metadata'

    seg_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"SPINEPS SELECTIVE — top/bottom {args.top_n} by {args.rank_by}")
    logger.info("=" * 70)

    # Load valid IDs
    valid_ids = None
    if args.valid_ids:
        try:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            logger.info(f"Loaded {len(valid_ids)} valid study IDs")
        except Exception as e:
            logger.error(f"Failed to load valid_ids: {e}")
            return 1

    # Select studies
    selected_ids, df_ranked = select_studies(
        uncertainty_csv, args.top_n, args.rank_by, valid_ids
    )

    skip_already_done = [s for s in selected_ids if already_segmented(s, spineps_dir)]
    to_run            = [s for s in selected_ids if not already_segmented(s, spineps_dir)]

    logger.info(f"\nSelected:       {len(selected_ids)}")
    logger.info(f"Already done:   {len(skip_already_done)} (skipping: {skip_already_done})")
    logger.info(f"To run SPINEPS: {len(to_run)}")

    if args.dry_run:
        logger.info("\n--- DRY RUN ---")
        for sid in to_run:
            row   = df_ranked[df_ranked['study_id'] == sid]
            score = float(row[args.rank_by].iloc[0]) if not row.empty else float('nan')
            logger.info(f"  {sid}  {args.rank_by}={score:.4f}")
        logger.info("--- DRY RUN complete — nothing was run ---")
        return 0

    if not to_run:
        logger.info("\nAll selected studies already segmented.")
        logger.info(f"Edit TOP_N in the slurm script and resubmit to segment more.")
        return 0

    series_df = load_series_csv(series_csv)
    if series_df is None:
        logger.error("Cannot load series CSV — aborting")
        return 1

    progress      = load_progress(progress_file)
    success_count = 0
    error_count   = 0

    for study_id in tqdm(to_run, desc='SPINEPS'):
        logger.info(f"\n[{study_id}]")

        # Re-check in case a parallel process finished it
        if already_segmented(study_id, spineps_dir):
            logger.info(f"  Already segmented since last check — skipping")
            mark_success(progress, study_id)
            save_progress(progress_file, progress)
            continue

        try:
            series_id = get_sagittal_t2_series(series_df, study_id)
            if series_id is None:
                logger.warning(f"  No sagittal T2w series in CSV")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            nifti_path = (nifti_dir / study_id / series_id /
                          f"sub-{study_id}_acq-sag_T2w.nii.gz")
            if not nifti_path.exists():
                logger.warning(f"  NIfTI not found: {nifti_path}")
                logger.warning(f"  Run 01_dicom_to_nifti.sh first for this study")
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            logger.info(f"  NIfTI: {nifti_path}")

            outputs = run_spineps(nifti_path, seg_dir / study_id, study_id)
            if outputs is None:
                mark_failed(progress, study_id)
                save_progress(progress_file, progress)
                error_count += 1
                continue

            save_metadata(study_id, outputs, metadata_dir)
            mark_success(progress, study_id)
            save_progress(progress_file, progress)
            success_count += 1

            row = df_ranked[df_ranked['study_id'] == study_id]
            if not row.empty:
                score = float(row[args.rank_by].iloc[0])
                logger.info(f"  ✓  {args.rank_by}={score:.4f}")

        except KeyboardInterrupt:
            logger.warning("\nInterrupted — progress saved")
            save_progress(progress_file, progress)
            break
        except Exception as e:
            logger.error(f"  Unexpected: {e}")
            logger.debug(traceback.format_exc())
            mark_failed(progress, study_id)
            save_progress(progress_file, progress)
            error_count += 1

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info(f"Success:      {success_count}")
    logger.info(f"Failed:       {error_count}")
    logger.info(f"Already done: {len(skip_already_done)}")
    if progress.get('failed'):
        logger.info(f"Failed IDs:   {progress['failed']}")
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
