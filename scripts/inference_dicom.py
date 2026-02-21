#!/usr/bin/env python3
"""
LSTV Detection via Epistemic Uncertainty — DICOM version
=========================================================

Uses pydicom to load raw DICOM series directly, bypassing NIfTI conversion.
Ian Pan's model was trained on DICOM pixel values — loading via NIfTI destroys
the signal (confidence drops from 0.97 → 0.003). Always use this script for
Ian Pan inference.

Output: results/epistemic_uncertainty/
  ├── lstv_uncertainty_metrics.csv   ← per-study scores (append-safe)
  ├── progress.json                  ← resume support
  └── debug_visualizations/          ← mid-slice PNGs (always saved)

Usage:
    python inference_dicom.py \\
        --input_dir  data/raw/train_images \\
        --series_csv data/raw/train_series_descriptions.csv \\
        --output_dir results/epistemic_uncertainty \\
        --mode       prod          # trial | prod
        [--trial_size 3]           # only used when mode=trial
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from natsort import natsorted
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not available — inference will fail")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

SAGITTAL_T2_PATTERNS = [
    'Sagittal T2/STIR',
    'Sagittal T2',
    'SAG T2',
    'Sag T2',
    'sag_t2',
]

DISC_LEVELS = ['L1L2', 'L2L3', 'L3L4', 'L4L5', 'L5S1']

# ============================================================================
# SERIES / DICOM LOADING
# ============================================================================

def load_series_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from series CSV")
        return df
    except Exception as e:
        logger.error(f"Failed to load series CSV: {e}")
        return None


def find_sagittal_series_dir(study_dir: Path,
                              series_df: pd.DataFrame | None,
                              study_id: str) -> Path | None:
    """
    Find the sagittal T2w series directory.

    Priority:
      1. Match series_id from CSV against subdirectory names
      2. Fallback: pick the subdirectory with the most DICOM files
    """
    # --- CSV lookup ---
    if series_df is not None:
        try:
            study_rows = series_df[series_df['study_id'] == int(study_id)]
            for pattern in SAGITTAL_T2_PATTERNS:
                match = study_rows[
                    study_rows['series_description'].str.contains(
                        pattern, case=False, na=False
                    )
                ]
                if not match.empty:
                    sid = str(match.iloc[0]['series_id'])
                    candidate = study_dir / sid
                    if candidate.exists():
                        return candidate
                    # Series dir may not exist by that exact name; fall through
                    logger.debug(f"  CSV series_id {sid} not found on disk — trying fallback")
        except Exception as e:
            logger.debug(f"  CSV lookup failed: {e}")

    # --- Fallback: most DICOMs ---
    subdirs = [d for d in study_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    best = max(subdirs, key=lambda d: len(list(d.glob('*.dcm'))))
    if list(best.glob('*.dcm')):
        logger.debug(f"  Fallback: using {best.name} ({len(list(best.glob('*.dcm')))} DICOMs)")
        return best
    return None


def load_dicom_volume(series_dir: Path) -> np.ndarray | None:
    """
    Load all .dcm files in series_dir, sort by InstanceNumber / filename,
    and stack into a (H, W, N_slices) float32 array normalised to [0, 255].
    """
    dcm_files = natsorted(series_dir.glob('*.dcm'))
    if not dcm_files:
        return None

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return None

    # Sort by InstanceNumber if available
    try:
        slices.sort(key=lambda s: int(s.InstanceNumber))
    except Exception:
        pass  # natsorted filename order is fine

    pixel_arrays = []
    for ds in slices:
        try:
            arr = ds.pixel_array.astype(np.float32)
            # Apply RescaleSlope / RescaleIntercept if present
            slope     = float(getattr(ds, 'RescaleSlope',     1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            arr = arr * slope + intercept
            pixel_arrays.append(arr)
        except Exception:
            continue

    if not pixel_arrays:
        return None

    volume = np.stack(pixel_arrays, axis=-1)  # (H, W, N)

    # Normalise to [0, 255]
    lo, hi = volume.min(), volume.max()
    if hi > lo:
        volume = (volume - lo) / (hi - lo) * 255.0
    return volume.astype(np.float32)


# ============================================================================
# IAN PAN MODEL
# ============================================================================

def load_ian_pan_model(model_path: Path, device: str = 'cuda'):
    """Load Ian Pan's LSTM uncertainty model from checkpoint."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required")

    # Import model architecture — adjust import path as needed for your env
    try:
        from models.ian_pan_model import LSTVUncertaintyModel  # type: ignore
        model = LSTVUncertaintyModel()
    except ImportError:
        # Fallback: try loading as a full checkpoint
        model = torch.load(model_path, map_location=device)
        if isinstance(model, dict):
            raise RuntimeError(
                "Checkpoint is a state_dict but model architecture import failed. "
                "Ensure models/ian_pan_model.py is on PYTHONPATH."
            )
        model.eval()
        return model.to(device)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def run_mc_dropout(model, input_tensor, n_passes: int = 20, device: str = 'cuda'):
    """
    Monte Carlo dropout inference.
    Returns (mean_probs, epistemic_uncertainty) both shape (n_disc_levels,).
    """
    model.train()  # enable dropout
    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(input_tensor.to(device))
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.append(probs)
    model.eval()

    all_probs = np.stack(all_probs, axis=0)     # (n_passes, batch, n_levels)
    mean_probs  = all_probs.mean(axis=0)[0]     # (n_levels,)
    epistemic   = all_probs.var(axis=0)[0]      # (n_levels,)
    return mean_probs, epistemic


# ============================================================================
# INFERENCE — single study
# ============================================================================

def preprocess_volume_for_model(volume: np.ndarray) -> 'torch.Tensor':
    """Convert (H, W, N) numpy array to model input tensor."""
    import torch
    # Ian Pan expects (1, N, H, W) — batch=1, slices as channels/seq
    # Adjust axis order based on actual model input spec
    arr = volume.transpose(2, 0, 1)            # (N, H, W)
    arr = arr / 255.0                          # already normalised but ensure [0,1]
    tensor = torch.from_numpy(arr).unsqueeze(0).float()  # (1, N, H, W)
    return tensor


def run_inference_study(study_id: str,
                         study_dir: Path,
                         series_df: pd.DataFrame | None,
                         model,
                         device: str,
                         debug_dir: Path,
                         n_mc_passes: int = 20) -> dict | None:
    """
    Run full Ian Pan inference on one study.
    Returns a flat dict of metrics, or None on failure.
    """
    series_dir = find_sagittal_series_dir(study_dir, series_df, study_id)
    if series_dir is None:
        logger.warning(f"  [{study_id}] No sagittal series found")
        return None

    volume = load_dicom_volume(series_dir)
    if volume is None:
        logger.warning(f"  [{study_id}] Could not load DICOM volume")
        return None

    n_slices = volume.shape[2]
    logger.info(f"  [{study_id}] Volume shape: {volume.shape}  series: {series_dir.name}")

    # Save mid-slice debug PNG (always, not just debug mode)
    _save_debug_slice(volume, study_id, debug_dir)

    # Inference
    try:
        import torch
        input_tensor = preprocess_volume_for_model(volume)
        mean_probs, epistemic = run_mc_dropout(model, input_tensor, n_mc_passes, device)
    except Exception as e:
        logger.error(f"  [{study_id}] Inference failed: {e}")
        logger.debug(traceback.format_exc())
        return None

    # Build result row
    row = {
        'study_id':          study_id,
        'series_dir':        str(series_dir),
        'n_slices':          n_slices,
        'n_mc_passes':       n_mc_passes,
        # Aggregate scores across all disc levels
        'mean_lstv_prob':    float(mean_probs.mean()),
        'max_lstv_prob':     float(mean_probs.max()),
        'mean_epistemic_unc': float(epistemic.mean()),
        'max_epistemic_unc': float(epistemic.max()),
        # Per-level scores
        **{f'prob_{lvl}':  float(mean_probs[i]) for i, lvl in enumerate(DISC_LEVELS)},
        **{f'unc_{lvl}':   float(epistemic[i])  for i, lvl in enumerate(DISC_LEVELS)},
    }
    return row


def _save_debug_slice(volume: np.ndarray, study_id: str, debug_dir: Path):
    debug_dir.mkdir(parents=True, exist_ok=True)
    mid = volume.shape[2] // 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.imshow(volume[:, :, mid].T, cmap='gray', origin='lower')
    ax.set_title(f'{study_id} — mid-slice {mid}/{volume.shape[2]}')
    ax.axis('off')
    fig.savefig(debug_dir / f'{study_id}_mid_slice.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_progress(output_dir: Path) -> dict:
    pf = output_dir / 'progress.json'
    if pf.exists():
        try:
            with open(pf) as f:
                p = json.load(f)
            logger.info(f"Resuming: {len(p.get('success',[]))} done, "
                        f"{len(p.get('failed',[]))} failed")
            return p
        except Exception:
            pass
    return {'success': [], 'failed': [], 'processed': []}


def save_progress(output_dir: Path, progress: dict):
    pf  = output_dir / 'progress.json'
    tmp = pf.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(progress, f, indent=2)
    tmp.replace(pf)


def append_result_csv(output_dir: Path, row: dict):
    """Append one row to the CSV; write header only if file is new."""
    csv_path = output_dir / 'lstv_uncertainty_metrics.csv'
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV Ian Pan Inference — DICOM mode')
    parser.add_argument('--input_dir',  required=True,
                        help='Root DICOM directory ({study_id}/{series_id}/*.dcm)')
    parser.add_argument('--series_csv', required=True,
                        help='CSV with study_id, series_id, series_description')
    parser.add_argument('--output_dir', default='results/epistemic_uncertainty')
    parser.add_argument('--model_path', default='models/ian_pan_lstv.pth')
    parser.add_argument('--valid_ids',  default=None,
                        help='.npy file of valid study IDs to process (e.g. models/valid_id.npy)')
    parser.add_argument('--mode',       choices=['trial', 'prod'], default='prod')
    parser.add_argument('--trial_size', type=int, default=3)
    parser.add_argument('--n_mc_passes', type=int, default=20,
                        help='Monte Carlo dropout passes for epistemic uncertainty')
    parser.add_argument('--retry_failed', action='store_true')
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir  = output_dir / 'debug_visualizations'

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Device
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Series CSV
    series_df = load_series_csv(Path(args.series_csv))

    # Model
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    logger.info(f"Loading model from {model_path}")
    model = load_ian_pan_model(model_path, device)

    # Study list — scan input dir directly
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(study_dirs)} study directories")

    # Filter to valid IDs if provided
    if args.valid_ids:
        try:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
            before = len(study_dirs)
            study_dirs = [d for d in study_dirs if d.name in valid_ids]
            logger.info(f"Filtered to {len(study_dirs)} studies via valid_ids "
                        f"({before - len(study_dirs)} excluded)")
        except Exception as e:
            logger.error(f"Failed to load valid_ids from {args.valid_ids}: {e}")
            return 1

    if args.mode == 'trial':
        study_dirs = study_dirs[:args.trial_size]
        logger.info(f"Trial mode: processing first {args.trial_size}")

    # Progress / resume
    progress = load_progress(output_dir)
    skip_ids = (
        set(progress.get('success', []))
        if not args.retry_failed
        else set()
    )
    study_dirs = [d for d in study_dirs if d.name not in skip_ids]
    logger.info(f"Studies remaining: {len(study_dirs)}")

    logger.info("=" * 70)
    logger.info("IAN PAN EPISTEMIC UNCERTAINTY INFERENCE")
    logger.info("=" * 70)
    logger.info(f"Mode:        {args.mode}")
    logger.info(f"Input:       {input_dir}")
    logger.info(f"Output:      {output_dir}")
    logger.info(f"MC passes:   {args.n_mc_passes}")
    logger.info(f"Total:       {len(study_dirs)} to process")
    logger.info("=" * 70)

    success = failed = 0

    for study_dir in tqdm(study_dirs, desc='Studies'):
        study_id = study_dir.name
        logger.info(f"\n[{study_id}]")

        try:
            row = run_inference_study(
                study_id, study_dir, series_df, model, device,
                debug_dir, args.n_mc_passes
            )
            if row is None:
                raise RuntimeError("run_inference_study returned None")

            append_result_csv(output_dir, row)
            progress['success'].append(study_id)
            if study_id not in progress['processed']:
                progress['processed'].append(study_id)
            save_progress(output_dir, progress)
            success += 1
            logger.info(f"  [{study_id}] ✓  mean_lstv_prob={row['mean_lstv_prob']:.4f}  "
                        f"mean_unc={row['mean_epistemic_unc']:.4f}")

        except KeyboardInterrupt:
            logger.warning("\n⚠ Interrupted — progress saved")
            save_progress(output_dir, progress)
            break
        except Exception as e:
            logger.error(f"  [{study_id}] ✗ {e}")
            logger.debug(traceback.format_exc())
            progress.setdefault('failed', []).append(study_id)
            if study_id not in progress['processed']:
                progress['processed'].append(study_id)
            save_progress(output_dir, progress)
            failed += 1

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info(f"Success: {success}  |  Failed: {failed}")
    logger.info(f"CSV:     {output_dir}/lstv_uncertainty_metrics.csv")
    logger.info(f"Debug:   {debug_dir}/")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("  sbatch slurm_scripts/02b_spineps_selective.sh  # set TOP_N= env var")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
