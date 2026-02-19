#!/usr/bin/env python3
"""
LSTV Detector - Morphological Castellvi Classification

Combines SPINEPS and TotalSpineSeg outputs to detect and classify
lumbosacral transitional vertebrae (LSTV) using purely morphological
criteria, assessed independently for each side (Left / Right).

Input files per study_id
------------------------
SPINEPS:
  spineps_dir/segmentations/{study_id}/{study_id}_seg-spine_msk.nii.gz
    - Label 43: Left transverse / costal process
    - Label 44: Right transverse / costal process
  spineps_dir/segmentations/{study_id}/{study_id}_unc.nii.gz
    - Normalised Shannon entropy [0, 1], float32

TotalSpineSeg:
  totalspine_dir/{study_id}/sagittal/{study_id}_sagittal_labeled.nii.gz
    - Labels 41-45: L1-L5; 46: L6 (only in lumbarisation); 50: Sacrum
    - Labels 91-100: intervertebral discs
  totalspine_dir/{study_id}/sagittal/{study_id}_sagittal_unc.nii.gz
    - Normalised Shannon entropy [0, 1], float32

Classification logic (per side)
---------------------------------
1.  Identify Target Vertebra (TV): L6 if label 46 present, else L5 (45).
2.  Isolate TV-level TP voxels from SPINEPS using the TV's S-I z-range in
    the TotalSpineSeg mask (both images brought to canonical RAS orientation
    before any voxel-coordinate arithmetic).
3.  Measure S-I height of the isolated TP mask (mm).
4.  Compute minimum distance between isolated TP mask and Sacrum mask using
    scipy distance_transform_edt.
5.  Decision tree:
      distance > 2.0 mm  →  height > 19 mm  →  Type I
                         →  height ≤ 19 mm  →  Normal
      distance ≤ 2.0 mm  →  sample mean uncertainty in Contact Zone
                         →  uncertainty > 0.35  →  Type II (pseudo-arthrosis)
                         →  uncertainty ≤ 0.35  →  Type III (solid fusion)

Final assembly (Left + Right)
------------------------------
  Normal  + Normal  → None (no LSTV)
  I  (either/both)  → Type I
  II (either/both)  → Type II
  III(either/both)  → Type III
  II on one side + III on other → Type IV

Usage:
    python 04_detect_lstv.py \
        --spineps_dir    results/spineps \
        --totalspine_dir results/totalspineseg \
        --output_dir     results/lstv_detection \
        [--uncertainty_threshold 0.35] \
        [--debug]
"""

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, distance_transform_edt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# TotalSpineSeg label map
LUMBAR_LABELS   = {41: 'L1', 42: 'L2', 43: 'L3', 44: 'L4', 45: 'L5', 46: 'L6'}
DISC_LABELS     = {91: 'T12-L1', 92: 'L1-L2', 93: 'L2-L3', 94: 'L3-L4',
                   95: 'L4-L5', 100: 'L5-S'}
SACRUM_LABEL    = 50
L5_LABEL        = 45
L6_LABEL        = 46

# SPINEPS costal / transverse process labels
TP_LEFT_LABEL   = 43
TP_RIGHT_LABEL  = 44

# Morphometric thresholds
TP_HEIGHT_MM        = 19.0   # S-I height above which a TP is "enlarged" (Type I)
CONTACT_DIST_MM     = 2.0    # Minimum TP-to-Sacrum distance; ≤ this = contact
UNC_THRESHOLD       = 0.35   # Mean uncertainty in contact zone; > this = Type II
CONTACT_DILATION_MM = 3.0    # Dilation radius (mm) used to build the contact zone


# ============================================================================
# NIfTI HELPERS
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI file and reorient to closest canonical (RAS) orientation.

    After this call:
      axis 0  →  Left–Right  (R+)
      axis 1  →  Posterior–Anterior  (A+)
      axis 2  →  Inferior–Superior  (S+)

    Returns (data_array, canonical_nii).
    """
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata(), nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    """Return voxel dimensions in mm as a length-3 array (dx, dy, dz)."""
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


# ============================================================================
# PER-SIDE MORPHOMETRICS
# ============================================================================

def get_tv_z_range(tss_data: np.ndarray, tv_label: int) -> Optional[Tuple[int, int]]:
    """
    Return the (z_min, z_max) slice indices of the Target Vertebra in
    canonical space (axis 2 = S-I).  Returns None if the label is absent.
    """
    tv_mask = tss_data == tv_label
    if not tv_mask.any():
        return None
    z_coords = np.where(tv_mask)[2]
    return int(z_coords.min()), int(z_coords.max())


def isolate_tp_at_tv(
    spineps_data: np.ndarray,
    tp_label: int,
    z_min: int,
    z_max: int,
) -> np.ndarray:
    """
    Extract TP voxels (tp_label) from SPINEPS that fall within the S-I
    z-range [z_min, z_max] of the Target Vertebra.

    Both images must already be in canonical orientation so that axis 2
    is comparable between the two volumes.  If the voxel grids differ in
    size (they will, since SPINEPS and TotalSpineSeg produce different FOVs
    and resolutions) we clip the z-range to the SPINEPS volume's extent
    rather than resampling, which is sufficient for coarse z-overlap.
    """
    tp_full = spineps_data == tp_label
    z_max_safe = min(z_max, spineps_data.shape[2] - 1)
    z_min_safe = max(z_min, 0)
    isolated = np.zeros_like(tp_full)
    isolated[:, :, z_min_safe:z_max_safe + 1] = tp_full[:, :, z_min_safe:z_max_safe + 1]
    return isolated


def measure_si_height_mm(mask: np.ndarray, vox_mm: np.ndarray) -> float:
    """
    Measure the superior-inferior extent (axis 2) of a binary mask in mm.
    Returns 0.0 if the mask is empty.
    """
    if not mask.any():
        return 0.0
    z_coords = np.where(mask)[2]
    return float((z_coords.max() - z_coords.min()) * vox_mm[2])


def min_distance_mm(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    vox_mm: np.ndarray,
) -> float:
    """
    Minimum surface-to-surface distance in mm between two binary masks,
    using a distance transform on the complement of mask_b.

    If the masks overlap, returns 0.0.
    If either mask is empty, returns infinity.
    """
    if not mask_a.any() or not mask_b.any():
        return float('inf')

    # distance_transform_edt measures distance from every False voxel
    # to the nearest True voxel, using anisotropic sampling.
    dist_from_b = distance_transform_edt(~mask_b, sampling=vox_mm)
    distances_at_a = dist_from_b[mask_a]
    return float(distances_at_a.min())


def build_contact_zone(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    vox_mm: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """
    Build a 'Contact Zone' mask as the intersection of dilated mask_a and
    dilated mask_b.  Dilation radius is approximated per-axis in voxels.
    """
    radius_vox = np.maximum(np.round(radius_mm / vox_mm).astype(int), 1)
    struct_a = np.ones(2 * radius_vox + 1, dtype=bool)
    struct_b = np.ones(2 * radius_vox + 1, dtype=bool)
    dilated_a = binary_dilation(mask_a, structure=struct_a)
    dilated_b = binary_dilation(mask_b, structure=struct_b)
    return dilated_a & dilated_b


def sample_uncertainty_in_zone(
    unc_data: np.ndarray,
    zone_mask: np.ndarray,
) -> float:
    """
    Return the mean uncertainty value within zone_mask.
    Returns NaN if the zone is empty.
    """
    vals = unc_data[zone_mask]
    return float(vals.mean()) if vals.size > 0 else float('nan')


# ============================================================================
# PER-SIDE CLASSIFICATION
# ============================================================================

def classify_side(
    tp_isolated: np.ndarray,
    sacrum_mask: np.ndarray,
    spineps_unc: np.ndarray,
    tss_unc: np.ndarray,
    vox_mm_spineps: np.ndarray,
    vox_mm_tss: np.ndarray,
    unc_threshold: float,
    debug: bool,
) -> Dict:
    """
    Apply the full decision tree for one side (Left or Right).

    Returns a dict with keys:
      classification : 'Normal' | 'Type I' | 'Type II' | 'Type III'
      tp_height_mm   : float
      tp_present     : bool
      contact        : bool
      dist_mm        : float
      mean_unc_spineps : float | None
      mean_unc_tss     : float | None
      contact_zone_voxels : int  (only when debug=True)
    """
    result = {
        'tp_present':        tp_isolated.any(),
        'tp_height_mm':      0.0,
        'contact':           False,
        'dist_mm':           float('inf'),
        'mean_unc_spineps':  None,
        'mean_unc_tss':      None,
        'classification':    'Normal',
    }

    if not tp_isolated.any():
        return result

    # ── Height ───────────────────────────────────────────────────────────────
    height_mm = measure_si_height_mm(tp_isolated, vox_mm_spineps)
    result['tp_height_mm'] = round(height_mm, 2)

    # ── Distance to Sacrum ───────────────────────────────────────────────────
    # The SPINEPS TP mask and TotalSpineSeg sacrum mask may live on different
    # grids; we use the TSS voxel size for the sacrum distance transform since
    # the sacrum label comes from TotalSpineSeg.
    if not sacrum_mask.any():
        # No sacrum segmented — can only assess TP height
        result['dist_mm'] = float('inf')
        result['classification'] = 'Type I' if height_mm > TP_HEIGHT_MM else 'Normal'
        return result

    dist_mm = min_distance_mm(tp_isolated, sacrum_mask, vox_mm_tss)
    result['dist_mm'] = round(dist_mm, 2)
    result['contact'] = dist_mm <= CONTACT_DIST_MM

    if not result['contact']:
        # No bony contact — height alone determines Type I vs Normal
        result['classification'] = 'Type I' if height_mm > TP_HEIGHT_MM else 'Normal'
        return result

    # ── Contact Zone Uncertainty ─────────────────────────────────────────────
    contact_zone = build_contact_zone(
        tp_isolated, sacrum_mask, vox_mm_tss, CONTACT_DILATION_MM
    )

    if debug:
        result['contact_zone_voxels'] = int(contact_zone.sum())

    # TotalSpineSeg uncertainty (primary — this model directly segmented the joint)
    mean_unc_tss = sample_uncertainty_in_zone(tss_unc, contact_zone)
    result['mean_unc_tss'] = round(mean_unc_tss, 4) if not np.isnan(mean_unc_tss) else None

    # SPINEPS uncertainty (secondary — different model space, used as corroboration)
    mean_unc_spineps = sample_uncertainty_in_zone(spineps_unc, contact_zone)
    result['mean_unc_spineps'] = round(mean_unc_spineps, 4) if not np.isnan(mean_unc_spineps) else None

    # Use TSS uncertainty as the primary decision signal; fall back to SPINEPS
    # if TSS map is unavailable.
    primary_unc = mean_unc_tss if result['mean_unc_tss'] is not None else mean_unc_spineps

    if primary_unc is None or np.isnan(primary_unc):
        # Cannot compute uncertainty — flag as indeterminate, default to Type II
        result['classification'] = 'Type II'
        result['unc_indeterminate'] = True
    elif primary_unc > unc_threshold:
        # High uncertainty → model confused by pseudo-arthrosis → incomplete fusion
        result['classification'] = 'Type II'
    else:
        # Low uncertainty → model sees solid bone → complete fusion
        result['classification'] = 'Type III'

    return result


# ============================================================================
# FINAL CASTELLVI ASSEMBLY
# ============================================================================

CASTELLVI_TABLE = {
    # (left_type, right_type) → final Castellvi type (None = no LSTV)
    ('Normal',  'Normal'):  None,
    ('Type I',  'Normal'):  'Type I',
    ('Normal',  'Type I'):  'Type I',
    ('Type I',  'Type I'):  'Type I',
    ('Type II', 'Normal'):  'Type II',
    ('Normal',  'Type II'): 'Type II',
    ('Type II', 'Type II'): 'Type II',
    ('Type III','Normal'):  'Type III',
    ('Normal',  'Type III'):'Type III',
    ('Type III','Type III'):'Type III',
    ('Type II', 'Type III'):'Type IV',
    ('Type III','Type II'): 'Type IV',
    # Edge cases involving Type I on one side
    ('Type I',  'Type II'): 'Type II',
    ('Type II', 'Type I'):  'Type II',
    ('Type I',  'Type III'):'Type III',
    ('Type III','Type I'):  'Type III',
}


def assemble_castellvi(left_cls: str, right_cls: str) -> Optional[str]:
    return CASTELLVI_TABLE.get((left_cls, right_cls), None)


# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class LSTVDetector:
    def __init__(
        self,
        spineps_dir: Path,
        totalspine_dir: Path,
        unc_threshold: float = UNC_THRESHOLD,
        debug: bool = False,
    ):
        self.spineps_dir    = spineps_dir
        self.totalspine_dir = totalspine_dir
        self.unc_threshold  = unc_threshold
        self.debug          = debug

    # ── File resolution ──────────────────────────────────────────────────────

    def _spineps_mask(self, study_id: str) -> Path:
        return self.spineps_dir / 'segmentations' / study_id / f"{study_id}_seg-spine_msk.nii.gz"

    def _spineps_unc(self, study_id: str) -> Path:
        return self.spineps_dir / 'segmentations' / study_id / f"{study_id}_unc.nii.gz"

    def _tss_mask(self, study_id: str) -> Path:
        return self.totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"

    def _tss_unc(self, study_id: str) -> Path:
        return self.totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_unc.nii.gz"

    # ── Per-study classification ─────────────────────────────────────────────

    def classify_study(self, study_id: str) -> Dict:
        result = {
            'study_id':       study_id,
            'lstv_detected':  False,
            'castellvi_type': None,
            'left':           {},
            'right':          {},
            'details':        {},
            'errors':         [],
        }

        # ── Load TotalSpineSeg mask (required) ───────────────────────────────
        tss_mask_path = self._tss_mask(study_id)
        if not tss_mask_path.exists():
            result['errors'].append(f"TotalSpineSeg mask not found: {tss_mask_path}")
            logger.warning(f"  [{study_id}] TSS mask missing — skipping")
            return result

        try:
            tss_data, tss_nii = load_canonical(tss_mask_path)
            tss_data = tss_data.astype(int)
        except Exception as e:
            result['errors'].append(f"Failed to load TSS mask: {e}")
            return result

        vox_tss = voxel_size_mm(tss_nii)

        # ── Load TotalSpineSeg uncertainty ───────────────────────────────────
        tss_unc_path = self._tss_unc(study_id)
        if tss_unc_path.exists():
            try:
                tss_unc_data, _ = load_canonical(tss_unc_path)
                tss_unc_data = tss_unc_data.astype(np.float32)
            except Exception as e:
                logger.warning(f"  [{study_id}] Could not load TSS uncertainty: {e}")
                tss_unc_data = np.zeros_like(tss_data, dtype=np.float32)
        else:
            logger.warning(f"  [{study_id}] TSS uncertainty map not found — using zeros")
            tss_unc_data = np.zeros_like(tss_data, dtype=np.float32)

        # ── Load SPINEPS mask (required) ─────────────────────────────────────
        spineps_mask_path = self._spineps_mask(study_id)
        if not spineps_mask_path.exists():
            result['errors'].append(f"SPINEPS mask not found: {spineps_mask_path}")
            logger.warning(f"  [{study_id}] SPINEPS mask missing — skipping")
            return result

        try:
            spineps_data, spineps_nii = load_canonical(spineps_mask_path)
            spineps_data = spineps_data.astype(int)
        except Exception as e:
            result['errors'].append(f"Failed to load SPINEPS mask: {e}")
            return result

        vox_spineps = voxel_size_mm(spineps_nii)

        # ── Load SPINEPS uncertainty ─────────────────────────────────────────
        spineps_unc_path = self._spineps_unc(study_id)
        if spineps_unc_path.exists():
            try:
                spineps_unc_data, _ = load_canonical(spineps_unc_path)
                spineps_unc_data = spineps_unc_data.astype(np.float32)
            except Exception as e:
                logger.warning(f"  [{study_id}] Could not load SPINEPS uncertainty: {e}")
                spineps_unc_data = np.zeros_like(spineps_data, dtype=np.float32)
        else:
            logger.warning(f"  [{study_id}] SPINEPS uncertainty map not found — using zeros")
            spineps_unc_data = np.zeros_like(spineps_data, dtype=np.float32)

        # ── Identify Target Vertebra ─────────────────────────────────────────
        unique_labels = set(np.unique(tss_data).tolist())
        tv_label = L6_LABEL if L6_LABEL in unique_labels else L5_LABEL
        result['details']['tv_label']      = tv_label
        result['details']['tv_name']       = LUMBAR_LABELS.get(tv_label, '?')
        result['details']['has_l6']        = L6_LABEL in unique_labels
        result['details']['sacrum_present'] = SACRUM_LABEL in unique_labels

        # ── TV z-range in canonical TSS space ────────────────────────────────
        tv_z_range = get_tv_z_range(tss_data, tv_label)
        if tv_z_range is None:
            result['errors'].append(f"Target vertebra (label {tv_label}) not found in TSS mask")
            logger.warning(f"  [{study_id}] TV label {tv_label} absent")
            return result

        z_min, z_max = tv_z_range
        result['details']['tv_z_range'] = [z_min, z_max]

        # ── Sacrum mask (from TSS, canonical space) ───────────────────────────
        sacrum_mask = (tss_data == SACRUM_LABEL)

        # ── Per-side classification ───────────────────────────────────────────
        for side, tp_label in [('left', TP_LEFT_LABEL), ('right', TP_RIGHT_LABEL)]:
            tp_isolated = isolate_tp_at_tv(spineps_data, tp_label, z_min, z_max)

            side_result = classify_side(
                tp_isolated     = tp_isolated,
                sacrum_mask     = sacrum_mask,
                spineps_unc     = spineps_unc_data,
                tss_unc         = tss_unc_data,
                vox_mm_spineps  = vox_spineps,
                vox_mm_tss      = vox_tss,
                unc_threshold   = self.unc_threshold,
                debug           = self.debug,
            )
            result[side] = side_result
            logger.info(
                f"  [{study_id}] {side:5s}: {side_result['classification']:8s}  "
                f"h={side_result['tp_height_mm']:.1f}mm  "
                f"d={side_result['dist_mm']:.1f}mm  "
                + (f"unc_tss={side_result['mean_unc_tss']}"
                   if side_result['mean_unc_tss'] is not None else "")
            )

        # ── Assemble final Castellvi type ─────────────────────────────────────
        left_cls  = result['left'].get('classification', 'Normal')
        right_cls = result['right'].get('classification', 'Normal')
        final     = assemble_castellvi(left_cls, right_cls)

        result['castellvi_type'] = final
        result['lstv_detected']  = final is not None

        return result

    # ── Batch ─────────────────────────────────────────────────────────────────

    def detect_all_studies(self, output_dir: Path) -> List[Dict]:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Discover studies: any study_id that has a SPINEPS segmentation folder
        spineps_seg_dir = self.spineps_dir / 'segmentations'
        if not spineps_seg_dir.exists():
            logger.error(f"SPINEPS segmentation dir not found: {spineps_seg_dir}")
            return []

        study_dirs = sorted([d for d in spineps_seg_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(study_dirs)} studies to process")

        results     = []
        lstv_count  = 0
        error_count = 0

        for study_dir in study_dirs:
            study_id = study_dir.name
            logger.info(f"\n[{study_id}]")

            try:
                result = self.classify_study(study_id)
                results.append(result)

                if result['errors']:
                    error_count += 1
                    for err in result['errors']:
                        logger.warning(f"  ⚠  {err}")

                if result['lstv_detected']:
                    lstv_count += 1
                    logger.info(f"  ✓ LSTV: {result['castellvi_type']}")
                else:
                    logger.info(f"  ✗ No LSTV detected")

            except Exception as e:
                logger.error(f"  Unhandled error for {study_id}: {e}")
                logger.debug(traceback.format_exc())
                results.append({
                    'study_id':      study_id,
                    'lstv_detected': False,
                    'castellvi_type': None,
                    'errors':        [str(e)],
                })
                error_count += 1

        # ── Save per-study results ────────────────────────────────────────────
        results_file = output_dir / 'lstv_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # ── Summary ───────────────────────────────────────────────────────────
        castellvi_breakdown = {t: 0 for t in ('Type I', 'Type II', 'Type III', 'Type IV')}
        for r in results:
            if r['castellvi_type'] in castellvi_breakdown:
                castellvi_breakdown[r['castellvi_type']] += 1

        summary = {
            'total_studies':      len(results),
            'lstv_detected':      lstv_count,
            'lstv_rate':          lstv_count / len(results) if results else 0.0,
            'error_count':        error_count,
            'uncertainty_threshold_used': self.unc_threshold,
            'castellvi_breakdown': castellvi_breakdown,
        }

        summary_file = output_dir / 'lstv_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("LSTV DETECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total studies:         {summary['total_studies']}")
        logger.info(f"LSTV detected:         {summary['lstv_detected']} ({summary['lstv_rate']:.1%})")
        logger.info(f"Errors / incomplete:   {summary['error_count']}")
        logger.info(f"Uncertainty threshold: {summary['uncertainty_threshold_used']}")
        logger.info("")
        logger.info("Castellvi Breakdown:")
        for type_name, count in summary['castellvi_breakdown'].items():
            logger.info(f"  {type_name}: {count}")
        logger.info("")
        logger.info(f"Results → {results_file}")
        logger.info(f"Summary → {summary_file}")

        return results


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Morphological LSTV Detection and Castellvi Classification'
    )
    parser.add_argument('--spineps_dir',    required=True,
                        help='SPINEPS output directory')
    parser.add_argument('--totalspine_dir', required=True,
                        help='TotalSpineSeg output directory')
    parser.add_argument('--output_dir',     required=True,
                        help='Output directory for detection results')
    parser.add_argument('--uncertainty_threshold', type=float, default=UNC_THRESHOLD,
                        help=f'Mean uncertainty threshold in contact zone for Type II vs III '
                             f'(default: {UNC_THRESHOLD}). Higher = more cases classified as '
                             f'Type III. Tune with --debug output.')
    parser.add_argument('--debug', action='store_true',
                        help='Write extra fields to JSON (contact zone sizes, raw '
                             'uncertainty values) to support threshold tuning.')
    args = parser.parse_args()

    detector = LSTVDetector(
        spineps_dir    = Path(args.spineps_dir),
        totalspine_dir = Path(args.totalspine_dir),
        unc_threshold  = args.uncertainty_threshold,
        debug          = args.debug,
    )

    detector.detect_all_studies(Path(args.output_dir))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
