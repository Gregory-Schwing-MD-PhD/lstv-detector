#!/usr/bin/env python3
"""
04_detect_lstv.py — Hybrid Two-Phase LSTV Castellvi Classifier (v5)
======================================================================
Classifies Lumbosacral Transitional Vertebrae (LSTV) using the Castellvi
system, then calls lstv_engine.py for radiologically-grounded phenotype
classification (lumbarization / sacralization / transitional_indeterminate).

WHAT'S NEW IN v5
-----------------
Integrates lumbosacral vertebral angle measurements from:
  Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280.
  doi: 10.22038/ABJS.2025.83244.3790

Five sagittal angles are now computed from segmentation masks:

  A-angle  : sacral superior surface vs scan vertical (Chalian 2012)
  B-angle  : L3 superior endplate vs sacral superior surface (Chalian 2012)
  C-angle  : largest posterior-body-line angle across TV±1 levels (novel)
  D-angle  : superior surfaces of most cranial sacrum vs most caudal lumbar
  delta    : D - D1  (D-angle minus angle of TV to TV-1)

KEY THRESHOLDS (Seilanian Toosi 2025):
  delta ≤ 8.5°  → Type 2 LSTV   Sens 92.3%, Spec 87.9%, NPV 99.5%
  C    ≤ 35.5°  → any LSTV      Sens 72.2%, Spec 57.6%, NPV 91.4%
  L4-L5 dehydrated + L5-S1 preserved → OR 19.9 for LSTV (p<0.001)
  Increased A-angle + decreased D-angle = independent predictors (p<0.05)

v5.1 FIXES
----------
- validate_tp_concordance now uses TSS L4-L5 disc (label 95) as an UPPER
  boundary for TP position. A valid L5 TP must sit BELOW the L4-L5 disc
  and ABOVE the L5-S1 disc / sacrum top. Disc labels are used as propagation
  anchors (ground-truth).
- All vertebral angle values are logged for every study (not just flagged ones).

CASTELLVI CLASSIFICATION (Castellvi et al. 1984, Spine 9:31-35)
-----------------------------------------------------------------
Type I   : Dysplastic TP ≥ 19 mm craniocaudal height, no sacral contact
           Ia = unilateral; Ib = bilateral
Type II  : Pseudo-articulation (diarthrodial joint) between TP and sacrum
           IIa = unilateral; IIb = bilateral
Type III : Complete osseous fusion of TP with sacrum
           IIIa = unilateral; IIIb = bilateral
Type IV  : Mixed — one side Type II, other side Type III

LABEL REFERENCE — SOURCE DISAMBIGUATION
----------------------------------------
SPINEPS seg-spine_msk.nii.gz  (subregion / semantic labels):
  43 = Costal_Process_Left   ← TP SOURCE
  44 = Costal_Process_Right  ← TP SOURCE
  26 = Sacrum
  ⚠ TSS labels 43/44 = L3/L4 vertebral bodies — NEVER used as TP source

SPINEPS seg-vert_msk.nii.gz  (VERIDAH per-vertebra instance labels):
  20=L1  21=L2  22=L3  23=L4  24=L5  25=L6  26=Sacrum

TotalSpineSeg sagittal_labeled.nii.gz:
  41=L1  42=L2  43=L3  44=L4  45=L5  50=Sacrum
  91-100 = disc labels
  95  = L4-L5 disc  ← NEW: used as upper boundary for L5 TP
  100 = L5-S1 disc  ← existing: used as lower boundary / floor
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lstv_engine import (
    load_lstv_masks, analyze_lstv, compute_lstv_pathology_score,
    TP_HEIGHT_MM, CONTACT_DIST_MM,
    TSS_SACRUM, TSS_LUMBAR, SP_TP_L, SP_TP_R, SP_SACRUM,
    VD_L1, VD_L2, VD_L3, VD_L4, VD_L5, VD_L6, VD_SAC,
    VERIDAH_NAMES, VERIDAH_TV_SEARCH, EXPECTED_LUMBAR,
)

# ── Vertebral angle thresholds (Seilanian Toosi et al. 2025) ──────────────────
DELTA_ANGLE_TYPE2_THRESHOLD = 8.5
C_ANGLE_LSTV_THRESHOLD      = 35.5
A_ANGLE_NORMAL_MEDIAN       = 41.0
D_ANGLE_NORMAL_MEDIAN       = 13.5

# ── TSS disc labels used as propagation anchors ───────────────────────────────
TSS_DISC_L4L5 = 95    # L4-L5 disc — upper boundary for L5 TP
TSS_DISC_L5S1 = 100   # L5-S1 disc — lower boundary / floor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-7s  %(message)s')
logger = logging.getLogger(__name__)

# ── Phase 2 signal thresholds ─────────────────────────────────────────────────
BBOX_HALF          = 16
P2_DARK_CLEFT_FRAC = 0.55
P2_MIN_STD_RATIO   = 0.12

# ── Cross-validation thresholds ───────────────────────────────────────────────
XVAL_MIN_DICE      = 0.30
XVAL_MAX_CENTROID  = 20.0   # mm

# ── TP sacrum-overlap threshold ───────────────────────────────────────────────
TP_SACRUM_OVERLAP_FRAC = 0.10


# ══════════════════════════════════════════════════════════════════════════════
# NIfTI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Cannot reduce {path.name} to 3D: shape={data.shape}")
    return data, nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a, b  = a.astype(bool), b.astype(bool)
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom else float('nan')


def centroid_mm(mask: np.ndarray, vox_mm: np.ndarray) -> Optional[np.ndarray]:
    coords = np.array(np.where(mask))
    return coords.mean(axis=1) * vox_mm if coords.size else None


def run_cross_validation(sag_spineps: np.ndarray,
                          sag_vert:    np.ndarray,
                          sag_tss:     np.ndarray,
                          vox_mm:      np.ndarray,
                          study_id:    str) -> dict:
    xval: dict = {'warnings': []}

    sp_sac  = (sag_spineps == SP_SACRUM)
    tss_sac = (sag_tss     == TSS_SACRUM)
    if sp_sac.any() and tss_sac.any():
        d = dice_coefficient(sp_sac, tss_sac)
        xval['sacrum_dice'] = round(d, 4)
        if d < XVAL_MIN_DICE:
            msg = f"Sacrum Dice={d:.3f} < {XVAL_MIN_DICE} — SPINEPS/TSS sacrum mismatch"
            logger.warning(f"  [{study_id}] {msg}")
            xval['warnings'].append(msg)
        else:
            logger.info(f"  [{study_id}] Sacrum Dice={d:.3f} ✓")

    vd_l5  = (sag_vert == VD_L5)
    tss_l5 = (sag_tss  == 45)
    if vd_l5.any() and tss_l5.any():
        c_v = centroid_mm(vd_l5,  vox_mm)
        c_t = centroid_mm(tss_l5, vox_mm)
        if c_v is not None and c_t is not None:
            dist = float(np.linalg.norm(c_v - c_t))
            xval['l5_centroid_dist_mm'] = round(dist, 2)
            if dist > XVAL_MAX_CENTROID:
                msg = f"L5 centroid dist={dist:.1f}mm > {XVAL_MAX_CENTROID}mm"
                logger.warning(f"  [{study_id}] {msg}")
                xval['warnings'].append(msg)
            else:
                logger.info(f"  [{study_id}] L5 centroid dist={dist:.1f}mm ✓")

    return xval


# ══════════════════════════════════════════════════════════════════════════════
# MASK OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_tv_z_range(vert_data: np.ndarray, tv_label: int) -> Optional[Tuple[int, int]]:
    mask = (vert_data == tv_label)
    if not mask.any(): return None
    zc = np.where(mask)[2]
    return int(zc.min()), int(zc.max())


def isolate_tp_at_tv(sp_data: np.ndarray, tp_label: int,
                     z_min: int, z_max: int) -> np.ndarray:
    tp    = (sp_data == tp_label)
    out   = np.zeros_like(tp)
    z_lo  = max(0, z_min - 3)
    z_hi  = min(sp_data.shape[2] - 1, z_max + 3)
    out[:, :, z_lo:z_hi + 1] = tp[:, :, z_lo:z_hi + 1]
    return out


def inferiormost_tp_cc(tp_mask: np.ndarray,
                        sacrum_mask: Optional[np.ndarray] = None) -> np.ndarray:
    if not tp_mask.any(): return np.zeros_like(tp_mask, bool)
    labeled, n = cc_label(tp_mask)
    if n == 1: return tp_mask.astype(bool)

    sac_z_min = None
    if sacrum_mask is not None and sacrum_mask.any():
        sac_z_min = int(np.where(sacrum_mask)[2].min())

    comps = []
    for i in range(1, n + 1):
        comp = (labeled == i)
        zc   = np.where(comp)[2]
        comps.append((float(zc.mean()), int(zc.max()), comp))
    comps.sort(key=lambda t: t[0])

    if sac_z_min is not None:
        cands = [c for _, zm, c in comps if zm < sac_z_min]
        if cands: return cands[0].astype(bool)

    return comps[0][2].astype(bool)


def measure_tp_height_mm(tp_mask: np.ndarray, vox_mm: np.ndarray) -> float:
    if not tp_mask.any(): return 0.0
    zc = np.where(tp_mask)[2]
    return float((int(zc.max()) - int(zc.min()) + 1) * vox_mm[2])


def tp_centroid_z_mm(sp_data: np.ndarray, tp_label: int,
                     z_range: Tuple[int, int], vox_mm: np.ndarray,
                     sac_mask: Optional[np.ndarray] = None) -> Optional[float]:
    isolated = isolate_tp_at_tv(sp_data, tp_label, *z_range)
    tp       = inferiormost_tp_cc(isolated, sac_mask)
    if not tp.any(): return None
    return float(np.mean(np.where(tp)[2])) * vox_mm[2]


def min_dist_3d(mask_a: np.ndarray, mask_b: np.ndarray,
                vox_mm: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    if not mask_a.any() or not mask_b.any():
        return float('inf'), None, None
    dt       = distance_transform_edt(~mask_b, sampling=vox_mm)
    dist_at  = np.where(mask_a, dt, np.inf)
    flat_idx = int(np.argmin(dist_at))
    vox_a    = np.array(np.unravel_index(flat_idx, mask_a.shape))
    dist_mm  = float(dt[tuple(vox_a)])

    z_lo  = max(0, int(vox_a[2]) - 20)
    z_hi  = min(mask_b.shape[2], int(vox_a[2]) + 20)
    sub   = mask_b[:, :, z_lo:z_hi]
    if sub.any():
        coords       = np.array(np.where(sub))
        coords[2, :] += z_lo
    else:
        coords = np.array(np.where(mask_b))
    d2    = ((coords.T * vox_mm - vox_a * vox_mm) ** 2).sum(axis=1)
    vox_b = coords[:, int(np.argmin(d2))]
    return dist_mm, vox_a, vox_b


# ══════════════════════════════════════════════════════════════════════════════
# TP SACRUM-DISPLACEMENT CHECK (v5.1: uses L4-L5 disc as upper boundary)
# ══════════════════════════════════════════════════════════════════════════════

def validate_tp_concordance(sag_sp:    np.ndarray,
                             sag_vert:  np.ndarray,
                             sag_tss:   Optional[np.ndarray],
                             vox_mm:    np.ndarray,
                             tv_z:      Tuple[int, int],
                             sac_mask:  np.ndarray,
                             study_id:  str) -> Tuple[bool, Tuple[int, int]]:
    """
    Verify that TP masks are correctly positioned at the TV level, using TSS
    disc labels as propagation anchors.

    v5.1 Logic:
    -----------
    A valid L5 TP should sit BETWEEN the L4-L5 disc and the L5-S1 disc:
      - UPPER boundary: bottom of L4-L5 disc (TSS label 95)
        → TP centroid must be BELOW this (i.e. more caudal / smaller z)
      - LOWER boundary: top of L5-S1 disc or sacrum superior edge
        → TP centroid must be ABOVE this (i.e. more cranial / larger z)

    A TP is flagged as displaced if:
      1. Its centroid is below (caudal to) the L5-S1 disc floor AND
      2. It overlaps the sacrum mask by >= 10%

    OR if its centroid is ABOVE the L4-L5 disc (would mean it belongs to L4,
    not L5 — a different kind of concordance error).

    Uses TSS disc labels 95 and 100 as ground-truth propagation anchors.
    """
    if sag_tss is None:
        return False, tv_z

    tss_sacrum_mask = (sag_tss == TSS_SACRUM)
    if not tss_sacrum_mask.any():
        return False, tv_z

    # ── Establish disc boundaries ─────────────────────────────────────────────
    # Upper boundary: bottom (most caudal z) of L4-L5 disc (TSS 95)
    l4l5_disc_mask = (sag_tss == TSS_DISC_L4L5)
    l4l5_disc_present = l4l5_disc_mask.any()
    if l4l5_disc_present:
        # The BOTTOM (most caudal = lowest z in RAS) of the L4-L5 disc
        l4l5_disc_bottom_z = int(np.where(l4l5_disc_mask)[2].min())
        l4l5_disc_top_z    = int(np.where(l4l5_disc_mask)[2].max())
        logger.info(
            f"  [{study_id}] L4-L5 disc (TSS 95): z=[{l4l5_disc_bottom_z},{l4l5_disc_top_z}]"
        )
    else:
        logger.info(f"  [{study_id}] L4-L5 disc (TSS 95): absent — skipping upper-bound check")
        l4l5_disc_bottom_z = None

    # Lower boundary: floor of L5-S1 disc or sacrum top
    l5s1_disc_mask = (sag_tss == TSS_DISC_L5S1)
    if l5s1_disc_mask.any():
        disc_floor_z   = int(np.where(l5s1_disc_mask)[2].max())
        disc_floor_src = 'TSS L5-S1 disc (label 100)'
    elif tss_sacrum_mask.any():
        disc_floor_z   = int(np.where(tss_sacrum_mask)[2].max())
        disc_floor_src = 'TSS sacrum superior edge (fallback)'
    else:
        return False, tv_z

    logger.info(
        f"  [{study_id}] TP boundaries: "
        f"upper=L4-L5-disc-bottom(z={l4l5_disc_bottom_z}) "
        f"lower={disc_floor_src}(z={disc_floor_z})"
    )

    def _is_displaced(tp_lbl: int, side: str) -> bool:
        isolated = isolate_tp_at_tv(sag_sp, tp_lbl, *tv_z)
        tp = inferiormost_tp_cc(isolated, tss_sacrum_mask)
        if not tp.any():
            return False

        n_tp             = int(tp.sum())
        overlap_frac     = int((tp & tss_sacrum_mask).sum()) / n_tp
        centroid_z       = float(np.mean(np.where(tp)[2]))

        # Primary displacement check: centroid below L5-S1 disc floor AND overlaps sacrum
        below_disc       = centroid_z < disc_floor_z
        sacrum_displaced = (overlap_frac >= TP_SACRUM_OVERLAP_FRAC and below_disc)

        # Secondary check: centroid above the L4-L5 disc bottom (wrong level — belongs to L4)
        above_l4l5 = (l4l5_disc_bottom_z is not None and centroid_z > l4l5_disc_bottom_z)

        logger.info(
            f"  [{study_id}] {side:5s} TP  "
            f"sacrum_overlap={overlap_frac:.1%}  "
            f"centroid_z={centroid_z:.1f}vox  "
            f"disc_floor_z={disc_floor_z}vox ({disc_floor_src})  "
            f"l4l5_disc_bottom_z={l4l5_disc_bottom_z}  "
            f"below_disc={below_disc}  "
            f"above_l4l5={above_l4l5}  "
            f"sacrum_displaced={sacrum_displaced}"
        )

        if sacrum_displaced:
            logger.warning(
                f"  [{study_id}] {side} TP DISPLACED IN SACRUM — "
                f"{overlap_frac:.0%} overlap with TSS sacrum "
                f"AND centroid ({centroid_z:.0f}) below disc floor ({disc_floor_z})"
            )
            return True

        if above_l4l5:
            logger.warning(
                f"  [{study_id}] {side} TP ABOVE L4-L5 DISC — "
                f"centroid z={centroid_z:.0f} > L4-L5 disc bottom z={l4l5_disc_bottom_z} "
                f"— TP instance belongs to L4 level, not L5"
            )
            return True

        return False

    displaced_L = _is_displaced(SP_TP_L, 'left')
    displaced_R = _is_displaced(SP_TP_R, 'right')

    if not displaced_L and not displaced_R:
        return False, tv_z

    # ── Find corrected TV Z reference ─────────────────────────────────────────
    # Prefer the region between L4-L5 disc and L5-S1 disc as the "true TV zone"
    ref_z: Optional[Tuple[int, int]] = None

    # First choice: use the inter-disc zone (between L4-L5 bottom and L5-S1 top)
    if l4l5_disc_present and l5s1_disc_mask.any():
        l5s1_top_z = int(np.where(l5s1_disc_mask)[2].max())
        l4l5_bot_z = int(np.where(l4l5_disc_mask)[2].min())
        if l4l5_bot_z > l5s1_top_z + 4:
            ref_z = (l5s1_top_z + 1, l4l5_bot_z - 1)
            logger.info(
                f"  [{study_id}] Using inter-disc zone as TV Z reference: {ref_z} "
                f"(between L5-S1 top z={l5s1_top_z} and L4-L5 bottom z={l4l5_bot_z})"
            )

    # Second choice: TSS L5 mask
    if ref_z is None:
        tss_l5 = (sag_tss == 45)
        if tss_l5.any():
            zc    = np.where(tss_l5)[2]
            ref_z = (int(zc.min()), int(zc.max()))
            logger.info(f"  [{study_id}] Using TSS L5 Z reference: {ref_z}")

    # Third choice: VERIDAH L5
    if ref_z is None:
        from lstv_engine import VD_L5
        vd_l5 = (sag_vert == VD_L5)
        if vd_l5.any():
            zc    = np.where(vd_l5)[2]
            ref_z = (int(zc.min()), int(zc.max()))
            logger.info(f"  [{study_id}] Using VERIDAH L5 Z reference: {ref_z}")

    if ref_z is None:
        logger.warning(f"  [{study_id}] Cannot correct — no L5 reference available")
        return False, tv_z

    sides = ('left ' if displaced_L else '') + ('right' if displaced_R else '')
    logger.info(
        f"  [{study_id}] TP correction: {sides.strip()} displaced → "
        f"re-isolating both from corrected z={ref_z}"
    )
    return True, ref_z


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — AXIAL T2w SIGNAL CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_bbox(axial_t2w: np.ndarray, midpoint: np.ndarray,
                  half: int = BBOX_HALF) -> Optional[np.ndarray]:
    x0, y0, z0 = int(midpoint[0]), int(midpoint[1]), int(midpoint[2])
    nx, ny, nz  = axial_t2w.shape
    if not (0 <= z0 < nz): return None
    patch = axial_t2w[max(0, x0 - half):min(nx, x0 + half),
                      max(0, y0 - half):min(ny, y0 + half), z0].copy()
    return patch if patch.size > 0 else None


def _classify_signal(patch: np.ndarray, axial_t2w: np.ndarray) -> Tuple[str, dict]:
    vals = patch.astype(float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 'Type II', {'reason': 'empty patch', 'valid': False}

    p_mean    = float(np.mean(vals))
    p_std     = float(np.std(vals))
    cv        = p_std / (p_mean + 1e-6)
    global_fg = axial_t2w[axial_t2w > 0]
    p95       = float(np.percentile(global_fg, 95)) if global_fg.size else 1.0
    dark_thr  = P2_DARK_CLEFT_FRAC * p95

    feats = {
        'patch_mean': round(p_mean, 2), 'patch_std': round(p_std, 2),
        'coeff_var':  round(cv, 4),     'global_p95': round(p95, 2),
        'dark_thresh':round(dark_thr, 2), 'valid': True,
    }

    if p_mean < dark_thr:
        feats['reason'] = (f"mean={p_mean:.1f} < dark_thr={dark_thr:.1f} "
                           f"— dark/intermediate signal → fibrocartilage cleft → Type II")
        return 'Type II', feats
    elif cv < P2_MIN_STD_RATIO:
        feats['reason'] = (f"CV={cv:.3f} < {P2_MIN_STD_RATIO} "
                           f"— uniform bright signal → osseous marrow bridge → Type III")
        return 'Type III', feats
    else:
        feats['reason'] = "Bright but heterogeneous — ambiguous; Type II (conservative)"
        return 'Type II', feats


def phase2_axial(side: str, tp_label: int,
                 ax_spineps: np.ndarray, ax_tss: np.ndarray,
                 ax_t2w: np.ndarray, ax_vox_mm: np.ndarray) -> dict:
    out: dict = {'phase2_attempted': True, 'classification': 'Type II',
                 'midpoint_vox': None, 'p2_features': None, 'p2_valid': False}

    tp_ax  = (ax_spineps == tp_label)
    sac_ax = (ax_tss     == TSS_SACRUM)

    if not tp_ax.any():
        out['p2_note'] = f"TP label {tp_label} absent in registered SPINEPS mask"
        return out
    if not sac_ax.any():
        out['p2_note'] = f"Sacrum label {TSS_SACRUM} absent in axial TSS"
        return out

    dist_mm, vox_a, vox_b = min_dist_3d(tp_ax, sac_ax, ax_vox_mm)
    if vox_a is None or vox_b is None:
        out['p2_note'] = 'min_dist_3d returned None'
        return out

    midpoint = ((vox_a + vox_b) / 2.0).astype(int)
    out['midpoint_vox']  = midpoint.tolist()
    out['axial_dist_mm'] = round(float(dist_mm), 3)

    patch = _extract_bbox(ax_t2w, midpoint, BBOX_HALF)
    if patch is None:
        out['p2_note'] = 'Bounding box outside axial volume'
        return out

    cls, feats              = _classify_signal(patch, ax_t2w)
    out['classification']   = cls
    out['p2_features']      = feats
    out['p2_valid']         = feats.get('valid', False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SAGITTAL GEOMETRIC CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def phase1_sagittal(side: str, tp_label: int,
                    sag_sp: np.ndarray, sag_tss: Optional[np.ndarray],
                    sag_vox_mm: np.ndarray,
                    tv_z_range: Tuple[int, int]) -> dict:
    out = {
        'tp_present': False, 'tp_height_mm': 0.0, 'contact': False,
        'dist_mm': float('inf'), 'tp_vox': None, 'sacrum_vox': None,
        'classification': 'Normal', 'phase1_done': False, 'sacrum_source': None,
    }

    tss_sac = (sag_tss == TSS_SACRUM) if sag_tss is not None else None
    if tss_sac is not None and tss_sac.any():
        sac_mask             = tss_sac
        out['sacrum_source'] = f'TSS label {TSS_SACRUM}'
    else:
        sac_mask = (sag_sp == SP_SACRUM)
        out['sacrum_source'] = 'SPINEPS label 26 (fallback)'

    tp_at_tv = isolate_tp_at_tv(sag_sp, tp_label, *tv_z_range)
    tp_mask  = inferiormost_tp_cc(tp_at_tv, sac_mask if sac_mask.any() else None)

    if not tp_mask.any():
        out['note'] = f"TP label {tp_label} absent at TV level"
        return out

    out['tp_present']   = True
    out['tp_height_mm'] = measure_tp_height_mm(tp_mask, sag_vox_mm)
    out['tp_z_min_vox'] = int(np.where(tp_mask)[2].min())
    out['tp_z_max_vox'] = int(np.where(tp_mask)[2].max())
    out['tp_centroid_z_mm'] = round(
        float(np.mean(np.where(tp_mask)[2])) * sag_vox_mm[2], 2)

    if not sac_mask.any():
        return out

    dist_mm, tp_vox, sac_vox = min_dist_3d(tp_mask, sac_mask, sag_vox_mm)
    out['dist_mm']    = round(float(dist_mm), 3)
    out['phase1_done']= True
    if tp_vox  is not None: out['tp_vox']     = tp_vox.tolist()
    if sac_vox is not None: out['sacrum_vox'] = sac_vox.tolist()

    if dist_mm > CONTACT_DIST_MM:
        out['contact']        = False
        if out['tp_height_mm'] >= TP_HEIGHT_MM:
            out['classification'] = 'Type I'
    else:
        out['contact']        = True
        out['classification'] = 'CONTACT_PENDING_P2'

    return out


# ══════════════════════════════════════════════════════════════════════════════
# L6 VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

TSS_DISC_LABELS = set(range(91, 101))


def _verify_l6(sag_vert:  np.ndarray,
               sag_tss:   Optional[np.ndarray],
               vox_mm:    np.ndarray,
               tv_z:      Tuple[int, int],
               study_id:  str) -> Tuple[bool, str]:
    if sag_tss is None:
        return False, "no TSS available — cannot verify L6"

    l6_z_min, l6_z_max = tv_z
    l6_centroid_z = (l6_z_min + l6_z_max) / 2.0

    tss_l5   = (sag_tss == 45)
    tss_sac  = (sag_tss == TSS_SACRUM)

    if not tss_sac.any():
        return False, "TSS sacrum (label 50) absent — cannot verify L6 position"

    sac_z_sup = float(np.where(tss_sac)[2].max())

    if l6_centroid_z <= sac_z_sup:
        return False, (
            f"L6 centroid z={l6_centroid_z:.0f} ≤ TSS sacrum superior z={sac_z_sup:.0f} "
            f"— VERIDAH L6 is inside the sacrum (mislabeled sacrum)"
        )

    if tss_l5.any():
        l5_z_min = float(np.where(tss_l5)[2].min())
        if l6_centroid_z >= l5_z_min:
            return False, (
                f"L6 centroid z={l6_centroid_z:.0f} ≥ TSS L5 inferior z={l5_z_min:.0f} "
                f"— VERIDAH L6 overlaps TSS L5 (not a distinct inferior segment)"
            )

    disc_above_z: Optional[float] = None
    disc_below_z: Optional[float] = None

    for disc_lbl in TSS_DISC_LABELS:
        disc_mask = (sag_tss == disc_lbl)
        if not disc_mask.any():
            continue
        disc_zc = float(np.mean(np.where(disc_mask)[2]))

        if disc_zc > l6_z_max + 2:
            if disc_above_z is None or disc_zc < disc_above_z:
                disc_above_z = disc_zc

        if disc_zc < l6_z_min - 2 and disc_zc > sac_z_sup:
            if disc_below_z is None or disc_zc > disc_below_z:
                disc_below_z = disc_zc

    if disc_above_z is None:
        return False, (
            f"no TSS disc found above VERIDAH L6 (L6 z=[{l6_z_min},{l6_z_max}]) — "
            f"no L5-L6 disc space present"
        )

    if disc_below_z is None:
        return False, (
            f"no TSS disc found between VERIDAH L6 inferior (z={l6_z_min}) and "
            f"TSS sacrum superior (z={sac_z_sup:.0f}) — no L6-S1 disc space present"
        )

    return True, (
        f"positional OK (centroid z={l6_centroid_z:.0f}, "
        f"sac_sup={sac_z_sup:.0f}), "
        f"disc above z={disc_above_z:.0f}, "
        f"disc below z={disc_below_z:.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PER-STUDY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_angle(v) -> str:
    """Format an angle value for logging, returning 'N/A' if None."""
    return f'{v:.1f}°' if v is not None else 'N/A'


def classify_study(study_id:       str,
                   spineps_dir:    Path,
                   totalspine_dir: Path,
                   registered_dir: Path,
                   nifti_dir:      Path,
                   run_morpho:     bool = True) -> dict:
    out: dict = {
        'study_id':           study_id,
        'lstv_detected':      False,
        'lstv_reason':        [],
        'castellvi_type':     None,
        'confidence':         'high',
        'left':               {},
        'right':              {},
        'details':            {},
        'cross_validation':   {},
        'lstv_morphometrics': None,
        'pathology_score':    None,
        'errors':             [],
    }

    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_path = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path  = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_sag    = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    tss_ax     = totalspine_dir / study_id / 'axial'    / f"{study_id}_axial_labeled.nii.gz"
    sp_ax      = registered_dir / study_id / f"{study_id}_spineps_reg.nii.gz"

    def _load(path: Path, tag: str):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}")
            return None, None
        try:
            return load_canonical(path)
        except Exception as exc:
            logger.warning(f"  {tag}: {exc}")
            return None, None

    def _find_t2w(acq: str) -> Optional[Path]:
        sd = nifti_dir / study_id
        if not sd.exists(): return None
        for sub in sorted(sd.iterdir()):
            p = sub / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
            if p.exists(): return p
        return None

    sag_sp,  sp_nii  = _load(spine_path, 'seg-spine_msk')
    sag_vert, _      = _load(vert_path,  'seg-vert_msk')
    sag_tss,  _      = _load(tss_sag,    'TSS sagittal')

    if sag_sp   is None: out['errors'].append('Missing SPINEPS seg-spine_msk'); return out
    if sag_vert is None: out['errors'].append('Missing SPINEPS seg-vert_msk');  return out
    if sag_tss  is None: out['errors'].append('Missing TotalSpineSeg sagittal'); return out

    sag_sp   = sag_sp.astype(int)
    sag_vert = sag_vert.astype(int)
    sag_tss  = sag_tss.astype(int)
    vox_mm   = voxel_size_mm(sp_nii)

    # TSS lumbar label inventory
    tss_unique         = sorted(int(v) for v in np.unique(sag_tss) if v > 0)
    tss_lumbar_present = {lbl: name for lbl, name in TSS_LUMBAR.items()
                          if lbl in tss_unique}
    tss_lumbar_missing = {lbl: name for lbl, name in TSS_LUMBAR.items()
                          if lbl not in tss_unique}
    logger.info(
        f"  [{study_id}] TSS lumbar labels present: "
        f"{list(tss_lumbar_present.values()) or 'none'}"
        + (f"  MISSING: {list(tss_lumbar_missing.values())}" if tss_lumbar_missing else "")
    )

    # Log TSS disc labels present (for TP boundary verification)
    tss_discs_present = [lbl for lbl in [TSS_DISC_L4L5, TSS_DISC_L5S1] if lbl in tss_unique]
    logger.info(
        f"  [{study_id}] TSS disc anchors present: "
        f"{['L4-L5(95)' if l == TSS_DISC_L4L5 else 'L5-S1(100)' for l in tss_discs_present] or 'none'}"
    )

    xval = run_cross_validation(sag_sp, sag_vert, sag_tss, vox_mm, study_id)
    out['cross_validation'] = xval
    for w in xval.get('warnings', []): out['errors'].append(f'XVAL: {w}')

    # Axial data (Phase 2)
    ax_tss, ax_tss_nii = _load(tss_ax, 'TSS axial')
    ax_sp,  _          = _load(sp_ax,  'registered SPINEPS axial')
    ax_t2w, ax_vox_mm  = None, None
    t2w_path = _find_t2w('ax')
    if t2w_path:
        arr, t2w_nii = _load(t2w_path, 'axial T2w')
        if arr is not None:
            ax_t2w    = arr
            ax_vox_mm = voxel_size_mm(t2w_nii)

    p2_available = (ax_tss is not None and ax_sp is not None and ax_t2w is not None)
    if not p2_available:
        logger.warning(f"  Phase 2 unavailable — contact cases → Type II (conservative)")

    if ax_tss is not None: ax_tss = ax_tss.astype(int)
    if ax_sp  is not None: ax_sp  = ax_sp.astype(int)

    # TV identification
    vert_unique = sorted(int(v) for v in np.unique(sag_vert) if v > 0)
    named       = [VERIDAH_NAMES[l] for l in vert_unique if l in VERIDAH_NAMES]
    logger.info(f"  [{study_id}] VERIDAH labels: {named}")

    tv_label, tv_name = None, None
    for cand in VERIDAH_TV_SEARCH:
        if cand in vert_unique:
            tv_label = cand; tv_name = VERIDAH_NAMES[cand]; break

    if tv_label is None:
        out['errors'].append('No lumbar VERIDAH labels found'); return out

    tv_z = get_tv_z_range(sag_vert, tv_label)
    if tv_z is None:
        out['errors'].append(f'TV label {tv_name} empty in VERIDAH mask'); return out

    # L6 verification
    l6_verified = False
    if tv_label == VD_L6:
        l6_ok, l6_reason = _verify_l6(sag_vert, sag_tss, vox_mm, tv_z, study_id)
        l6_verified = l6_ok
        if not l6_ok:
            logger.warning(
                f"  [{study_id}] VERIDAH L6 FAILED verification: {l6_reason} "
                f"— demoting TV to L5"
            )
            if VD_L5 in vert_unique:
                tv_label = VD_L5
                tv_name  = VERIDAH_NAMES[VD_L5]
                tv_z     = get_tv_z_range(sag_vert, VD_L5) or tv_z
            else:
                out['errors'].append('L6 verification failed and no VERIDAH L5 fallback')
                return out
        else:
            logger.info(f"  [{study_id}] VERIDAH L6 verified ✓ — {l6_reason}")

    # TP concordance validation (v5.1: uses L4-L5 disc as upper boundary)
    tss_sac_mask = (sag_tss == TSS_SACRUM) if sag_tss is not None else None
    sac_mask_p1  = (tss_sac_mask
                    if tss_sac_mask is not None and tss_sac_mask.any()
                    else (sag_sp == SP_SACRUM))

    tp_corrected, tv_z_final = validate_tp_concordance(
        sag_sp, sag_vert, sag_tss, vox_mm, tv_z, sac_mask_p1, study_id)

    out['details'] = {
        'tv_label':                   tv_label,
        'tv_name':                    tv_name,
        'has_l6':                     tv_label == VD_L6,
        'l6_verified':                l6_verified if tv_label == VD_L6 else None,
        'tv_z_range':                 list(tv_z),
        'tp_concordance_corrected':   tp_corrected,
        'corrected_tv_z_range':       list(tv_z_final) if tp_corrected else None,
        'sag_vox_mm':                 vox_mm.tolist(),
        'phase2_available':           p2_available,
        'tp_source':                  'seg-spine_msk labels 43 (L) / 44 (R)',
        'sacrum_source':              'TSS label 50 (preferred) / SPINEPS 26 (fallback)',
        'label_note':                 'TSS 43/44 = L3/L4 vertebrae — TP always from seg-spine_msk',
        'tss_lumbar_labels':          tss_lumbar_present,
        'tss_disc_anchors_used':      {
            'l4l5_disc_label': TSS_DISC_L4L5,
            'l5s1_disc_label': TSS_DISC_L5S1,
        },
    }
    logger.info(
        f"  [{study_id}] TV={tv_name}  z=[{tv_z[0]},{tv_z[1]}]"
        + (f"  → corrected z=[{tv_z_final[0]},{tv_z_final[1]}]" if tp_corrected else "")
    )

    # Phase 1 + 2 per side
    for side, tp_lbl in (('left', SP_TP_L), ('right', SP_TP_R)):
        try:
            p1 = phase1_sagittal(side, tp_lbl, sag_sp, sag_tss, vox_mm, tv_z_final)
            logger.info(
                f"  {side:5s} P1: {p1['classification']:22s} "
                f"h={p1['tp_height_mm']:.1f}mm  "
                f"d={p1['dist_mm']:.1f}mm  "
                f"z=[{p1.get('tp_z_min_vox','?')},{p1.get('tp_z_max_vox','?')}]  "
                f"cz={p1.get('tp_centroid_z_mm','?')}mm  "
                f"sac={p1.get('sacrum_source','?')}"
            )

            if p1['contact'] and p2_available:
                p2 = phase2_axial(side, tp_lbl, ax_sp, ax_tss, ax_t2w, ax_vox_mm)
                p1['phase2']        = p2
                p1['classification']= p2['classification']
                logger.info(f"  {side:5s} P2: {p2['classification']}  "
                            f"valid={p2.get('p2_valid')}  "
                            f"reason={p2.get('p2_features', {}).get('reason','?')}")
            elif p1['contact'] and not p2_available:
                p1['classification'] = 'Type II'
                p1['phase2']         = {'phase2_attempted': False,
                                        'p2_note': 'Axial data unavailable — Type II (conservative)'}
                out['confidence']    = 'low'

            out[side] = p1

        except Exception as exc:
            out['errors'].append(f'{side}: {exc}')
            logger.error(f"  {side} failed: {exc}")
            logger.debug(traceback.format_exc())

    # Final Castellvi type
    l_cls = out['left'].get('classification',  'Normal')
    r_cls = out['right'].get('classification', 'Normal')
    valid = {l_cls, r_cls} - {'Normal', 'CONTACT_PENDING_P2'}

    if valid:
        RANK = {'Type I': 1, 'Type II': 2, 'Type III': 3, 'Type IV': 4}
        if (l_cls not in ('Normal', 'CONTACT_PENDING_P2') and
                r_cls not in ('Normal', 'CONTACT_PENDING_P2')):
            out['castellvi_type'] = (l_cls + 'b' if l_cls == r_cls else 'Type IV')
        else:
            dominant = max(valid, key=lambda t: RANK.get(t, 0))
            out['castellvi_type'] = dominant + 'a'

        out['lstv_detected'] = True
        out['lstv_reason'].append(f"Castellvi {out['castellvi_type']} — TP morphology")
        logger.info(f"  ✓ [{study_id}] Castellvi: {out['castellvi_type']}")
    else:
        logger.info(f"  ✗ [{study_id}] No Castellvi finding")

    # Extended LSTV morphometrics + angle analysis
    if run_morpho:
        try:
            masks  = load_lstv_masks(study_id, spineps_dir, totalspine_dir)
            morpho = analyze_lstv(masks, castellvi_result=out)
            out['lstv_morphometrics'] = morpho.to_dict()

            consensus = morpho.lumbar_count_consensus
            phenotype = morpho.lstv_phenotype or 'normal'

            if consensus is not None and consensus != EXPECTED_LUMBAR:
                out['lstv_detected'] = True
                direction = 'LUMBARIZATION' if consensus > EXPECTED_LUMBAR else 'SACRALIZATION'
                reason = (f"Lumbar count = {consensus} (expected {EXPECTED_LUMBAR}) — "
                          f"{direction} by vertebral counting "
                          f"(TSS={morpho.lumbar_count_tss}, "
                          f"VERIDAH={morpho.lumbar_count_veridah})")
                out['lstv_reason'].append(reason)
                logger.info(f"  ✓ [{study_id}] LSTV: {reason}")

            if phenotype in ('sacralization', 'lumbarization'):
                out['lstv_detected'] = True
                primary = morpho.primary_criteria_met or []
                reason  = (f"Phenotype: {phenotype.upper()} "
                           f"({morpho.phenotype_confidence} confidence) — "
                           f"criteria: {'; '.join(primary)}")
                if not any('Phenotype' in r for r in out['lstv_reason']):
                    out['lstv_reason'].append(reason)
                logger.info(f"  ✓ [{study_id}] LSTV: {reason}")

            # v5: Angle-based LSTV detection (Seilanian Toosi 2025)
            morpho_dict   = out['lstv_morphometrics'] or {}
            va_dict       = morpho_dict.get('vertebral_angles') or {}
            va_delta      = va_dict.get('delta_angle')
            va_delta_flag = va_dict.get('delta_le8p5', False)
            va_c_flag     = va_dict.get('c_le35p5', False)
            va_disc_pat   = va_dict.get('disc_pattern_lstv', False)
            va_a_increased= va_dict.get('a_increased', False)
            va_d_decreased= va_dict.get('d_decreased', False)
            va_c_angle    = va_dict.get('c_angle')
            va_a_angle    = va_dict.get('a_angle')
            va_b_angle    = va_dict.get('b_angle')
            va_d_angle    = va_dict.get('d_angle')
            va_d1_angle   = va_dict.get('d1_angle')
            va_summary    = va_dict.get('summary', '')

            # ── Always log all angle values (v5.1 fix) ────────────────────────
            logger.info(
                f"  [{study_id}] Angles (Seilanian Toosi 2025): "
                f"A={_fmt_angle(va_a_angle)}  "
                f"B={_fmt_angle(va_b_angle)}  "
                f"C={_fmt_angle(va_c_angle)}  "
                f"D={_fmt_angle(va_d_angle)}  "
                f"D1={_fmt_angle(va_d1_angle)}  "
                f"delta={_fmt_angle(va_delta)}  "
                f"[delta_flag={va_delta_flag}  c_flag={va_c_flag}  "
                f"disc_pattern={va_disc_pat}  "
                f"A_elevated={va_a_increased}  D_decreased={va_d_decreased}]"
            )
            if va_summary:
                logger.info(f"  [{study_id}] Angles summary: {va_summary}")

            if va_delta_flag:
                out['lstv_detected'] = True
                reason = (f"Angle: delta={va_delta:.1f}° ≤ {DELTA_ANGLE_TYPE2_THRESHOLD}° — "
                          f"predicts Castellvi Type 2 LSTV (sens 92.3%, spec 87.9%, "
                          f"NPV 99.5%; Seilanian Toosi 2025)")
                if not any('delta' in r.lower() and 'Angle' in r for r in out['lstv_reason']):
                    out['lstv_reason'].append(reason)
                logger.info(f"  ✓ [{study_id}] LSTV (angle/delta): {reason}")

            elif va_c_flag and (va_disc_pat or va_a_increased or va_d_decreased):
                out['lstv_detected'] = True
                corroboration = (
                    'disc pattern' if va_disc_pat
                    else 'A↑' if va_a_increased else 'D↓'
                )
                reason = (f"Angle: C={va_c_angle:.1f}° ≤ {C_ANGLE_LSTV_THRESHOLD}° "
                          f"+ {corroboration} — supports LSTV "
                          f"(sens 72.2%, spec 57.6%; Seilanian Toosi 2025)")
                if not any('C=' in r and 'Angle' in r for r in out['lstv_reason']):
                    out['lstv_reason'].append(reason)
                logger.info(f"  ✓ [{study_id}] LSTV (angle/C+): {reason}")

            elif va_disc_pat and not out['lstv_detected']:
                out['lstv_detected'] = True
                reason = ("Disc pattern: L4-L5 dehydrated + L5-S1 preserved "
                          "(OR 19.9 in multivariate; Seilanian Toosi 2025)")
                if not any('Disc pattern' in r for r in out['lstv_reason']):
                    out['lstv_reason'].append(reason)
                logger.info(f"  ✓ [{study_id}] LSTV (disc pattern): {reason}")

            logger.info(
                f"  [{study_id}] Morphometrics: "
                f"TV={morpho.tv_name}, count={consensus}, "
                f"phenotype={phenotype} ({morpho.phenotype_confidence}), "
                f"primary={morpho.primary_criteria_met}"
            )

        except Exception as exc:
            logger.error(f"  [{study_id}] lstv_engine error: {exc}")
            out['errors'].append(f'lstv_engine: {exc}')

    # ── Final summary log — always includes all angle values (v5.1) ──────────
    morpho_dict = out.get('lstv_morphometrics') or {}
    probs_dict  = morpho_dict.get('probabilities') or {}
    sr_dict     = morpho_dict.get('surgical_relevance') or {}
    va_dict     = morpho_dict.get('vertebral_angles') or {}
    p_sac       = probs_dict.get('p_sacralization', 0)
    p_lumb      = probs_dict.get('p_lumbarization', 0)
    p_norm      = probs_dict.get('p_normal', 0)
    wl_risk     = sr_dict.get('wrong_level_risk', '?')
    bert_prob   = sr_dict.get('bertolotti_probability', 0)
    rdr         = morpho_dict.get('relative_disc_ratio')

    # Build full angle string — always, regardless of LSTV status
    delta_val = va_dict.get('delta_angle')
    c_val     = va_dict.get('c_angle')
    a_val     = va_dict.get('a_angle')
    d_val     = va_dict.get('d_angle')
    d1_val    = va_dict.get('d1_angle')
    b_val     = va_dict.get('b_angle')
    angle_str = (
        f"  A={_fmt_angle(a_val)}"
        f"  B={_fmt_angle(b_val)}"
        f"  C={_fmt_angle(c_val)}{'⚠' if va_dict.get('c_le35p5') else ''}"
        f"  D={_fmt_angle(d_val)}{'↓' if va_dict.get('d_decreased') else ''}"
        f"  D1={_fmt_angle(d1_val)}"
        f"  delta={_fmt_angle(delta_val)}{'⚠' if va_dict.get('delta_le8p5') else ''}"
    ) if (delta_val is not None or c_val is not None) else "  [angles not computed]"

    if out['lstv_detected']:
        ph_str = ''
        if morpho_dict.get('lstv_phenotype'):
            ph_str = (f"phenotype={morpho_dict['lstv_phenotype']} "
                      f"({morpho_dict.get('phenotype_confidence','')})")
        logger.info(
            f"  ✓✓ [{study_id}] LSTV DETECTED  "
            f"Castellvi={out.get('castellvi_type','None')}  "
            f"{ph_str}  "
            f"P(sac)={p_sac:.0%}  P(lumb)={p_lumb:.0%}  P(norm)={p_norm:.0%}  "
            f"surgical_risk={wl_risk}  bertolotti={bert_prob:.0%}"
            + (f"  disc_ratio={rdr:.2f}" if rdr is not None else "")
            + angle_str
            + (f"  [TP-FIXED]" if tp_corrected else "")
        )
    else:
        logger.info(
            f"  ✗✗ [{study_id}] No LSTV  "
            f"P(sac)={p_sac:.0%}  P(lumb)={p_lumb:.0%}  P(norm)={p_norm:.0%}"
            + angle_str
        )

    out['pathology_score'] = compute_lstv_pathology_score(
        out, out.get('lstv_morphometrics'))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STUDY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_studies_csv(csv_path: Path, top_n: int, rank_by: str,
                        valid_ids: Optional[set]) -> List[str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids: df = df[df['study_id'].isin(valid_ids)]
    df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    ids = df.head(top_n)['study_id'].tolist() + df.tail(top_n)['study_id'].tolist()
    seen, result = set(), []
    for sid in ids:
        if sid not in seen: result.append(sid); seen.add(sid)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Hybrid Two-Phase LSTV Castellvi Classifier + Morphometrics (v5.1)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spineps_dir',      required=True)
    parser.add_argument('--totalspine_dir',   required=True)
    parser.add_argument('--registered_dir',   required=True)
    parser.add_argument('--nifti_dir',        required=True)
    parser.add_argument('--output_dir',       required=True)
    parser.add_argument('--study_id',         default=None)
    parser.add_argument('--all',              action='store_true')
    parser.add_argument('--uncertainty_csv',  default=None)
    parser.add_argument('--valid_ids',        default=None)
    parser.add_argument('--top_n',            type=int, default=None)
    parser.add_argument('--rank_by',          default='l5_s1_confidence')
    parser.add_argument('--no_morpho',        action='store_true')
    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    registered_dir = Path(args.registered_dir)
    nifti_dir      = Path(args.nifti_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    if args.study_id:
        study_ids = [args.study_id]
    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")
    else:
        if not args.uncertainty_csv or not args.top_n:
            parser.error("--uncertainty_csv and --top_n required unless --all or --study_id")
        valid = (set(str(x) for x in np.load(args.valid_ids)) if args.valid_ids else None)
        study_ids = select_studies_csv(Path(args.uncertainty_csv), args.top_n,
                                        args.rank_by, valid)
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]

    logger.info(f"Processing {len(study_ids)} studies")

    results: List[dict] = []
    errors              = 0
    castellvi_counts    = {k: 0 for k in
                           ['Type Ia','Type Ib','Type IIa','Type IIb',
                            'Type IIIa','Type IIIb','Type IV']}
    phenotype_counts: Dict[str, int] = {}
    tp_correction_count = 0
    delta_le8p5_count   = 0
    c_le35p5_count      = 0
    disc_pattern_count  = 0
    angle_only_lstv     = 0

    for sid in study_ids:
        logger.info(f"\n{'='*60}\n[{sid}]")
        try:
            r = classify_study(
                sid, spineps_dir, totalspine_dir, registered_dir, nifti_dir,
                run_morpho=not args.no_morpho,
            )
            results.append(r)
            if r.get('errors'): errors += 1
            if r.get('details', {}).get('tp_concordance_corrected'): tp_correction_count += 1
            ct = r.get('castellvi_type') or ''
            for k in castellvi_counts:
                if ct.replace(' ', '') == k.replace(' ', ''):
                    castellvi_counts[k] += 1
            morpho = r.get('lstv_morphometrics') or {}
            ph = morpho.get('lstv_phenotype', 'normal')
            phenotype_counts[ph] = phenotype_counts.get(ph, 0) + 1

            va = morpho.get('vertebral_angles') or {}
            if va.get('delta_le8p5'):      delta_le8p5_count += 1
            if va.get('c_le35p5'):         c_le35p5_count += 1
            if va.get('disc_pattern_lstv'):disc_pattern_count += 1
            has_ct   = bool(ct and ct not in ('None', None))
            va_flags = va.get('delta_le8p5') or va.get('c_le35p5') or va.get('disc_pattern_lstv')
            if va_flags and not has_ct:
                angle_only_lstv += 1

        except Exception as exc:
            logger.error(f"  Unhandled: {exc}")
            logger.debug(traceback.format_exc())
            errors += 1

    lstv_n = sum(1 for r in results if r.get('lstv_detected'))
    n      = max(len(results), 1)
    scores = sorted(
        ((r['study_id'], r.get('pathology_score') or 0) for r in results),
        key=lambda t: t[1], reverse=True,
    )

    p_sac_vals  = []
    p_lumb_vals = []
    wl_risk_counts: Dict[str, int] = {}
    bertolotti_ge50 = 0
    high_cert_sac  = 0
    high_cert_lumb = 0
    nerve_ambig    = 0
    rel_disc_low   = 0

    for r in results:
        morpho = r.get('lstv_morphometrics') or {}
        probs  = morpho.get('probabilities') or {}
        ps = probs.get('p_sacralization', 0)
        pl = probs.get('p_lumbarization', 0)
        p_sac_vals.append(ps)
        p_lumb_vals.append(pl)
        if ps > 0.80:  high_cert_sac  += 1
        if pl > 0.80:  high_cert_lumb += 1

        sr  = morpho.get('surgical_relevance') or {}
        wlr = sr.get('wrong_level_risk', 'low')
        wl_risk_counts[wlr] = wl_risk_counts.get(wlr, 0) + 1
        if sr.get('nerve_root_ambiguity'): nerve_ambig += 1
        bp = sr.get('bertolotti_probability', 0)
        if bp >= 0.50: bertolotti_ge50 += 1

        rdr = morpho.get('relative_disc_ratio')
        if rdr is not None and rdr < 0.65: rel_disc_low += 1

    sep = '=' * 60
    logger.info(f"\n{sep}")
    logger.info(f"{'LSTV DETECTION SUMMARY':^60}")
    logger.info(f"{sep}")
    logger.info(f"Studies processed:        {len(results)}")
    logger.info(f"LSTV detected:            {lstv_n}  ({100*lstv_n/n:.1f}%)")
    logger.info(f"  Sacralization:          {phenotype_counts.get('sacralization',0)}")
    logger.info(f"  Lumbarization:          {phenotype_counts.get('lumbarization',0)}")
    logger.info(f"  Transitional:           {phenotype_counts.get('transitional_indeterminate',0)}")
    logger.info(f"  Normal:                 {phenotype_counts.get('normal',0)}")
    logger.info(f"Errors:                   {errors}")
    logger.info(f"TP concordance fixes:     {tp_correction_count}")

    logger.info(f"\n── Castellvi Type Breakdown ──────────────────────────────")
    for t, cnt in castellvi_counts.items():
        if cnt: logger.info(f"  {t:12s}: {cnt}")
    total_ct = sum(castellvi_counts.values())
    logger.info(f"  {'TOTAL':12s}: {total_ct}  ({100*total_ct/n:.1f}%)")

    logger.info(f"\n── Vertebral Angle Statistics (Seilanian Toosi 2025) ─────")
    logger.info(f"  delta ≤ {DELTA_ANGLE_TYPE2_THRESHOLD}° (Type 2 predictor): {delta_le8p5_count} ({100*delta_le8p5_count/n:.1f}%)")
    logger.info(f"  C ≤ {C_ANGLE_LSTV_THRESHOLD}° (any LSTV):         {c_le35p5_count} ({100*c_le35p5_count/n:.1f}%)")
    logger.info(f"  L4-L5 dehyd + L5-S1 preserved:      {disc_pattern_count} ({100*disc_pattern_count/n:.1f}%)")
    logger.info(f"  Angle-only LSTV (no Castellvi):      {angle_only_lstv}")

    logger.info(f"\n── Probability Model Statistics ──────────────────────────")
    if p_sac_vals:
        logger.info(f"  P(sacralization): mean={float(np.mean(p_sac_vals)):.2%}  >80%: {high_cert_sac}")
        logger.info(f"  P(lumbarization): mean={float(np.mean(p_lumb_vals)):.2%}  >80%: {high_cert_lumb}")
        logger.info(f"  Relative disc ratio < 0.65: {rel_disc_low}")

    logger.info(f"\n── Surgical Risk Distribution ────────────────────────────")
    for risk_lvl in ('critical', 'high', 'moderate', 'low-moderate', 'low'):
        cnt = wl_risk_counts.get(risk_lvl, 0)
        if cnt: logger.info(f"  {risk_lvl:14s}: {cnt} ({100*cnt/n:.1f}%)")
    logger.info(f"  Nerve root ambiguity: {nerve_ambig}")
    logger.info(f"  Bertolotti P≥50%:     {bertolotti_ge50}")

    logger.info(f"\n── Top-10 Pathology Scores ───────────────────────────────")
    for sid, sc in scores[:10]:
        r_match = next((r for r in results if r['study_id'] == sid), {})
        morpho  = r_match.get('lstv_morphometrics') or {}
        probs   = morpho.get('probabilities') or {}
        sr      = morpho.get('surgical_relevance') or {}
        va      = morpho.get('vertebral_angles') or {}
        ph      = morpho.get('lstv_phenotype', '?')
        ct      = r_match.get('castellvi_type', 'None')
        fix     = ' [TP-FIXED]' if r_match.get('details', {}).get('tp_concordance_corrected') else ''
        ps      = probs.get('p_sacralization', 0)
        pl      = probs.get('p_lumbarization', 0)
        wl      = sr.get('wrong_level_risk', '?')
        bp      = sr.get('bertolotti_probability', 0)
        rdr     = morpho.get('relative_disc_ratio')
        rdr_str = f'  disc_ratio={rdr:.2f}' if rdr is not None else ''
        # Always include all angles in top-10 listing
        delta_v = va.get('delta_angle')
        c_v     = va.get('c_angle')
        a_v     = va.get('a_angle')
        d_v     = va.get('d_angle')
        angle_line = (
            f"  A={_fmt_angle(a_v)}"
            f"  C={_fmt_angle(c_v)}{'⚠' if va.get('c_le35p5') else ''}"
            f"  D={_fmt_angle(d_v)}{'↓' if va.get('d_decreased') else ''}"
            f"  delta={_fmt_angle(delta_v)}{'⚠' if va.get('delta_le8p5') else ''}"
        ) if (delta_v is not None or c_v is not None) else ''
        logger.info(
            f"  {sid}: score={sc:.1f}  {ph}  castellvi={ct}  "
            f"P(sac)={ps:.0%}  P(lumb)={pl:.0%}  "
            f"surgical_risk={wl}  bertolotti={bp:.0%}"
            f"{rdr_str}{angle_line}{fix}"
        )

    logger.info(f"\n{sep}")

    out_json = output_dir / 'lstv_results.json'
    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)

    summary = {
        'total':                       len(results),
        'lstv_detected':               lstv_n,
        'lstv_rate':                   round(lstv_n / n, 4),
        'errors':                      errors,
        'tp_concordance_fixes':        tp_correction_count,
        'castellvi_breakdown':         castellvi_counts,
        'phenotype_breakdown':         phenotype_counts,
        'probability_stats': {
            'mean_p_sacralization':    round(float(np.mean(p_sac_vals)), 4) if p_sac_vals else None,
            'mean_p_lumbarization':    round(float(np.mean(p_lumb_vals)), 4) if p_lumb_vals else None,
            'high_confidence_sac':     high_cert_sac,
            'high_confidence_lumb':    high_cert_lumb,
            'relative_disc_ratio_low': rel_disc_low,
        },
        'surgical_risk_breakdown':     wl_risk_counts,
        'nerve_root_ambiguity_count':  nerve_ambig,
        'bertolotti_probability_ge50': bertolotti_ge50,
        'angle_stats': {
            'delta_le8p5_count':       delta_le8p5_count,
            'c_le35p5_count':          c_le35p5_count,
            'disc_pattern_count':      disc_pattern_count,
            'angle_only_lstv':         angle_only_lstv,
            'reference':               'Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280',
        },
        'top_scores': scores[:20],
    }
    with open(output_dir / 'lstv_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info(f"Results → {out_json}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
