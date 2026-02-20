#!/usr/bin/env python3
"""
05_visualize_overlay.py — LSTV Overlay Visualizer v7
=====================================================
Layout purpose-built for stepwise visual Castellvi grading.

  3 × 3 panel grid
  ─────────────────────────────────────────────────────────────────────────────
  ROW 0  — Castellvi Type I check  (TP craniocaudal height ≥ 19 mm?)
  ─────────────────────────────────────────────────────────────────────────────
  [0,0]  Left  TP sagittal — x maximising left  TP craniocaudal span
         Yellow bracket shows measured height.
         Cyan dashed line = TV mid-level (z_tv).
         Orange dashed line = min-dist level (z_min_dist).

  [0,1]  Right TP sagittal — x maximising right TP craniocaudal span
         Same annotations.

  [0,2]  TSS Labels — Axial at z_ref = mean(z_tv, z_min_dist)
         Single representative axial slice; both rows' sagittal panels
         draw lines referencing z_tv and z_min_dist, this slice sits
         between them to give anatomy context for both.

  ─────────────────────────────────────────────────────────────────────────────
  ROW 1  — Castellvi Type II / III check  (pseudarthrosis or fusion?)
  ─────────────────────────────────────────────────────────────────────────────
  [1,0]  Left  TP sagittal — same x as [0,0], but now annotated at
         z_min_dist_L (orange dashed line = closest-approach level).
         SPINEPS TP + sacrum masks overlaid so you can see contact/gap.

  [1,1]  Right TP sagittal — same x as [0,1], annotated at z_min_dist_R.

  [1,2]  TSS Labels — Axial at z_min_dist (side with smaller gap)
         + undilated TP masks + sacrum, showing the actual contact plane.

  ─────────────────────────────────────────────────────────────────────────────
  ROW 2  — Context + summary
  ─────────────────────────────────────────────────────────────────────────────
  [2,0]  Sagittal TSS level confirmation (midline x)
         Confirm TV=L5/L6 vs sacrum; both z_tv and z_min_dist lines drawn.

  [2,1]  Axial T2w plain at TV mid — anatomy orientation / sanity check.

  [2,2]  Classification summary
         Includes: Castellvi type, LSTV flag, TP heights L/R (measured
         craniocaudal span from sagittal masks), TP-sacrum min distances
         L/R (from registered axial distance transform), per-side class.
  ─────────────────────────────────────────────────────────────────────────────

Indicator lines on every sagittal panel
────────────────────────────────────────
  Cyan  dashed  = z_tv      (TV mid-level, used for Type I height measurement)
  Orange dashed = z_min_dist_L or z_min_dist_R (closest approach for that side)

All sagittal panels use the ORIGINAL sagittal T2w + pre-registration SPINEPS /
TSS labels to avoid reslice artifacts.
"""

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

TP_HEIGHT_MM     = 19.0
CONTACT_DIST_MM  = 2.0
TP_LEFT_LABEL    = 43
TP_RIGHT_LABEL   = 44
SACRUM_LABEL     = 50
L5_LABEL         = 45
L6_LABEL         = 46

TSS_LABEL_COLORS = {
    41:  ([0.20, 0.40, 0.80], 'L1'),
    42:  ([0.25, 0.50, 0.85], 'L2'),
    43:  ([0.30, 0.60, 0.90], 'L3'),
    44:  ([0.35, 0.70, 0.95], 'L4'),
    45:  ([0.40, 0.80, 1.00], 'L5'),
    46:  ([0.00, 0.30, 0.70], 'L6'),
    50:  ([1.00, 0.55, 0.00], 'Sacrum'),
    95:  ([0.90, 0.20, 0.90], 'L4-L5'),
    100: ([1.00, 0.00, 0.60], 'L5-S'),
}

DISPLAY_DILATION_VOXELS = 2

# Line colours for indicator lines on sagittal panels
LINE_TV       = 'cyan'
LINE_MINDIST  = '#FF8C00'   # dark-orange


# ============================================================================
# NIfTI HELPERS
# ============================================================================

def load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return data, nii


def voxel_size_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def get_tv_z_range(tss_data: np.ndarray,
                   tv_label: int) -> Optional[Tuple[int, int]]:
    mask = tss_data == tv_label
    if not mask.any():
        return None
    z = np.where(mask)[2]
    return int(z.min()), int(z.max())


def isolate_tp_at_tv(data: np.ndarray, tp_label: int,
                     z_min: int, z_max: int) -> np.ndarray:
    tp_full = data == tp_label
    iso     = np.zeros_like(tp_full)
    z_lo    = max(z_min, 0)
    z_hi    = min(z_max, data.shape[2] - 1)
    iso[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
    return iso


def dilate_for_display(mask: np.ndarray, voxels: int = 2) -> np.ndarray:
    if not mask.any() or voxels < 1:
        return mask
    struct = np.ones((voxels * 2 + 1,) * 3, dtype=bool)
    return binary_dilation(mask, structure=struct)


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_t2w(nifti_dir: Path, study_id: str, acq: str) -> Optional[Path]:
    study_dir = nifti_dir / study_id
    if not study_dir.exists():
        return None
    for series_dir in sorted(study_dir.iterdir()):
        p = series_dir / f"sub-{study_id}_acq-{acq}_T2w.nii.gz"
        if p.exists():
            return p
    return None


def find_original_spineps_seg(spineps_dir: Path, study_id: str) -> Optional[Path]:
    p = spineps_dir / 'segmentations' / study_id / f"{study_id}_seg-spine_msk.nii.gz"
    return p if p.exists() else None


def find_original_tss_sag(totalspine_dir: Path, study_id: str) -> Optional[Path]:
    p = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"
    return p if p.exists() else None


def find_native_axial_tss(totalspine_dir: Path, study_id: str) -> Optional[Path]:
    p = totalspine_dir / study_id / 'axial' / f"{study_id}_axial_labeled.nii.gz"
    return p if p.exists() else None


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def overlay_mask(ax, mask2d: np.ndarray, color_rgb, alpha: float = 0.65):
    if not mask2d.any():
        return
    rgba = np.zeros((*mask2d.shape, 4), dtype=float)
    rgba[mask2d] = [*color_rgb, alpha]
    ax.imshow(rgba.transpose(1, 0, 2), origin='lower')


def _ax_sl(vol: Optional[np.ndarray], z: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[:, :, min(z, vol.shape[2] - 1)]


def _sag_sl(vol: Optional[np.ndarray], x: int) -> np.ndarray:
    if vol is None:
        return np.zeros((1, 1))
    return vol[min(x, vol.shape[0] - 1), :, :]


def _unavailable(ax, label: str):
    ax.set_facecolor('#0d0d1a')
    ax.text(0.5, 0.5, f'{label}\nnot available',
            ha='center', va='center', color='#888888', fontsize=10,
            transform=ax.transAxes)
    ax.axis('off')


def _hline(ax, z_sag: Optional[float], color: str, label: str = ''):
    """Draw a horizontal dashed line at voxel row z_sag on a sagittal panel."""
    if z_sag is None:
        return
    y = float(z_sag)
    ax.axhline(y=y, color=color, linewidth=1.3, linestyle='--', alpha=0.90)
    if label:
        ax.text(3, y + 1, label, color=color, fontsize=7, va='bottom',
                fontweight='bold')


# ============================================================================
# AFFINE CONVERSION: axial z → sagittal z
# ============================================================================

def ax_z_to_sag_z(ax_nii:  Optional[nib.Nifti1Image],
                   sag_nii: Optional[nib.Nifti1Image],
                   z_ax: int) -> Optional[float]:
    """Map axial voxel z-index → sagittal voxel z-index via world RAS."""
    if ax_nii is None or sag_nii is None:
        return None
    try:
        cx    = ax_nii.shape[0] / 2.0
        cy    = ax_nii.shape[1] / 2.0
        world = nib.affines.apply_affine(ax_nii.affine,
                                          np.array([[cx, cy, z_ax]]))[0]
        vx    = nib.affines.apply_affine(np.linalg.inv(sag_nii.affine),
                                          world[np.newaxis])[0]
        return float(vx[2])
    except Exception as e:
        logger.debug(f"ax_z_to_sag_z failed: {e}")
        return None


# ============================================================================
# SLICE SELECTION: max craniocaudal TP height  (Type I)
# ============================================================================

def best_x_for_tp_height(orig_spineps: Optional[np.ndarray],
                          orig_tss_sag:  Optional[np.ndarray],
                          tp_label: int,
                          tv_label: int,
                          vox_z_mm: float) -> Tuple[int, float]:
    """
    Sweep sagittal x-slices of orig_spineps.  Within each slice restrict
    the TP mask to the TV z-range (from orig_tss_sag).  Measure craniocaudal
    span = (z_max - z_min) × vox_z_mm.  Return (best_x, max_span_mm).
    """
    if orig_spineps is None:
        return 0, 0.0

    tp_3d = (orig_spineps == tp_label)

    if orig_tss_sag is not None:
        zr = get_tv_z_range(orig_tss_sag, tv_label)
        if zr is not None:
            z_lo, z_hi = zr
            tmp = np.zeros_like(tp_3d)
            tmp[:, :, z_lo:z_hi + 1] = tp_3d[:, :, z_lo:z_hi + 1]
            if tmp.any():
                tp_3d = tmp

    if not tp_3d.any():
        return orig_spineps.shape[0] // 2, 0.0

    best_x, best_span = orig_spineps.shape[0] // 2, 0.0
    for x in range(orig_spineps.shape[0]):
        col = tp_3d[x]                  # (Y, Z)
        if not col.any():
            continue
        zc = np.where(col.any(axis=0))[0]
        if zc.size < 2:
            continue
        span = (zc.max() - zc.min()) * vox_z_mm
        if span > best_span:
            best_span = span
            best_x    = x

    return best_x, best_span


# ============================================================================
# SLICE SELECTION: min distance TP ↔ sacrum  (Type II/III)
# ============================================================================

def min_dist_z_for_tp_sacrum(tp_mask:     np.ndarray,
                               sacrum_mask: np.ndarray,
                               vox_mm:      np.ndarray) -> Tuple[int, float]:
    """
    Distance-transform approach in registered axial space.
    Returns (z_best, min_dist_mm).  Returns (0, inf) if masks are empty.
    """
    if not tp_mask.any() or not sacrum_mask.any():
        return 0, float('inf')

    dist = distance_transform_edt(~sacrum_mask, sampling=vox_mm)
    nz   = tp_mask.shape[2]
    per_z = np.full(nz, np.inf)
    for z in range(nz):
        sl = tp_mask[:, :, z]
        if sl.any():
            per_z[z] = dist[:, :, z][sl].min()

    z_best = int(np.argmin(per_z))
    return z_best, float(per_z[z_best])


# ============================================================================
# TSS AXIAL SLICE HELPER
# ============================================================================

def _tss_axial_slice(ax_bg:       np.ndarray,
                      tss_display: Optional[np.ndarray],
                      z: int) -> Optional[np.ndarray]:
    """Return TSS label slice at z, shape-matched to ax_bg slice."""
    if tss_display is None:
        return None
    z   = min(z, tss_display.shape[2] - 1)
    sl  = tss_display[:, :, z]
    bg  = ax_bg[:, :, min(z, ax_bg.shape[2] - 1)]
    if sl.shape != bg.shape:
        out = np.zeros(bg.shape, dtype=sl.dtype)
        sy  = min(sl.shape[0], bg.shape[0])
        sx  = min(sl.shape[1], bg.shape[1])
        out[:sy, :sx] = sl[:sy, :sx]
        return out
    return sl


# ============================================================================
# PANEL FUNCTIONS
# ============================================================================

# ── Sagittal TP height panel  [row 0, col 0/1] ──────────────────────────────

def _panel_sag_tp_height(ax,
                          sag_img:     Optional[np.ndarray],
                          sag_spineps: Optional[np.ndarray],
                          sag_tss:     Optional[np.ndarray],
                          tp_label:    int,
                          side_name:   str,
                          x_idx:       int,
                          span_mm:     float,
                          z_tv_sag:    Optional[float],
                          z_md_sag:    Optional[float],
                          tv_name:     str):
    """
    Row 0 sagittal panel.
    Shows the TP mask that maximises craniocaudal height (Type I check).
    Yellow bracket annotates the measured span.
    Cyan line  = TV mid-level.
    Orange line = min-dist level (for context; graded in row 1).
    """
    color_this  = [1.00, 0.10, 0.10] if tp_label == TP_LEFT_LABEL else [0.00, 0.80, 1.00]
    color_other = [0.00, 0.80, 1.00] if tp_label == TP_LEFT_LABEL else [1.00, 0.10, 0.10]
    other_label = TP_RIGHT_LABEL     if tp_label == TP_LEFT_LABEL else TP_LEFT_LABEL

    img_sl = _sag_sl(sag_img, x_idx)
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.80)

    if sag_spineps is not None:
        # Other side, faint
        overlay_mask(ax, _sag_sl(sag_spineps == other_label, x_idx),
                     color_other, 0.22)
        # This side, bright
        this_sl = _sag_sl(sag_spineps == tp_label, x_idx)
        overlay_mask(ax, this_sl, color_this, 0.85)

        # Yellow height bracket
        if this_sl.any():
            zc = np.where(this_sl.any(axis=0))[0]
            if zc.size >= 2:
                z_lo, z_hi = int(zc.min()), int(zc.max())
                mid_z  = zc[len(zc) // 2]
                col_at = this_sl[:, mid_z]
                x_mid  = int(np.where(col_at)[0].mean()) if col_at.any() else this_sl.shape[0] // 2
                tick   = max(3, int(this_sl.shape[0] * 0.02))
                ax.plot([x_mid, x_mid], [z_lo, z_hi],
                        color='yellow', lw=1.8, alpha=0.95)
                for z_end in (z_lo, z_hi):
                    ax.plot([x_mid - tick, x_mid + tick], [z_end, z_end],
                            color='yellow', lw=1.8, alpha=0.95)
                ax.text(x_mid + tick + 2, (z_lo + z_hi) / 2,
                        f'{span_mm:.1f} mm',
                        color='yellow', fontsize=8, va='center', fontweight='bold')

    if sag_tss is not None:
        overlay_mask(ax, _sag_sl(sag_tss == SACRUM_LABEL, x_idx),
                     [1.00, 0.55, 0.00], 0.45)

    # Indicator lines
    _hline(ax, z_tv_sag, LINE_TV,      label='TV mid')
    _hline(ax, z_md_sag, LINE_MINDIST, label='min-dist')

    ax.legend(handles=[
        mpatches.Patch(color=color_this,          label=f'{side_name} TP'),
        mpatches.Patch(color=color_other,         label=f'{"R" if side_name=="Left" else "L"} TP (faint)'),
        mpatches.Patch(color=[1.00, 0.55, 0.00],  label='Sacrum'),
        mpatches.Patch(color=LINE_TV,             label='TV mid'),
        mpatches.Patch(color=LINE_MINDIST,        label='min-dist z'),
    ], loc='lower right', fontsize=6, framealpha=0.55)

    thresh_flag = '✓' if span_mm < TP_HEIGHT_MM else '✗ ≥19 mm → Type I'
    ax.set_title(
        f'Type I check — {side_name} TP  (x={x_idx})\n'
        f'Height: {span_mm:.1f} mm  {thresh_flag}',
        fontsize=10, color='white'
    )
    ax.axis('off')


# ── TSS Labels axial  [row 0, col 2  AND  row 1, col 2] ─────────────────────

def _panel_tss_axial(ax,
                      img_sl:     np.ndarray,
                      tss_sl:     Optional[np.ndarray],
                      tp_left_sl: Optional[np.ndarray],
                      tp_right_sl: Optional[np.ndarray],
                      sacrum_sl:  Optional[np.ndarray],
                      native:     bool,
                      z_idx:      int,
                      subtitle:   str = ''):
    """
    TSS label colours on axial T2w.  Optionally also overlays TP masks
    (undilated) and sacrum for the min-dist panel.
    """
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.78)
    patches = []
    if tss_sl is not None:
        for label, (color, name) in TSS_LABEL_COLORS.items():
            m = tss_sl == label
            if m.any():
                overlay_mask(ax, m, color, 0.45)
                patches.append(mpatches.Patch(color=color, label=name))

    # Extra TP overlays for min-dist panel (undilated, accurate)
    if tp_left_sl is not None and tp_left_sl.any():
        overlay_mask(ax, tp_left_sl,  [1.00, 0.10, 0.10], 0.75)
        if not any(p.get_label() == 'Left TP' for p in patches):
            patches.append(mpatches.Patch(color=[1.00, 0.10, 0.10], label='Left TP'))
    if tp_right_sl is not None and tp_right_sl.any():
        overlay_mask(ax, tp_right_sl, [0.00, 0.80, 1.00], 0.75)
        if not any(p.get_label() == 'Right TP' for p in patches):
            patches.append(mpatches.Patch(color=[0.00, 0.80, 1.00], label='Right TP'))

    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.55)

    src   = 'native axial' if native else 'resampled'
    title = f'TSS Labels — Axial  z={z_idx}  ({src})'
    if subtitle:
        title += f'\n{subtitle}'
    ax.set_title(title, fontsize=10, color='white')
    ax.axis('off')


# ── Sagittal TP proximity panel  [row 1, col 0/1] ────────────────────────────

def _panel_sag_tp_proximity(ax,
                              sag_img:     Optional[np.ndarray],
                              sag_spineps: Optional[np.ndarray],
                              sag_tss:     Optional[np.ndarray],
                              tp_label:    int,
                              side_name:   str,
                              x_idx:       int,
                              span_mm:     float,
                              dist_mm:     float,
                              z_tv_sag:    Optional[float],
                              z_md_sag:    Optional[float],
                              tv_name:     str):
    """
    Row 1 sagittal panel.
    Same x-slice as the height panel above (maximises TP visibility),
    but now the orange min-dist line is the primary annotation.
    Both TP and sacrum shown brightly so you can see contact/gap.
    """
    color_this  = [1.00, 0.10, 0.10] if tp_label == TP_LEFT_LABEL else [0.00, 0.80, 1.00]
    other_label = TP_RIGHT_LABEL     if tp_label == TP_LEFT_LABEL else TP_LEFT_LABEL
    color_other = [0.00, 0.80, 1.00] if tp_label == TP_LEFT_LABEL else [1.00, 0.10, 0.10]

    img_sl = _sag_sl(sag_img, x_idx)
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower', alpha=0.80)

    if sag_spineps is not None:
        overlay_mask(ax, _sag_sl(sag_spineps == other_label, x_idx),
                     color_other, 0.22)
        overlay_mask(ax, _sag_sl(sag_spineps == tp_label, x_idx),
                     color_this, 0.85)

    if sag_tss is not None:
        overlay_mask(ax, _sag_sl(sag_tss == SACRUM_LABEL, x_idx),
                     [1.00, 0.55, 0.00], 0.60)   # brighter sacrum for proximity check

    # Indicator lines — orange (min-dist) is primary here
    _hline(ax, z_tv_sag, LINE_TV,      label='TV mid')
    _hline(ax, z_md_sag, LINE_MINDIST, label='closest approach')

    ax.legend(handles=[
        mpatches.Patch(color=color_this,         label=f'{side_name} TP'),
        mpatches.Patch(color=color_other,        label=f'{"R" if side_name=="Left" else "L"} TP (faint)'),
        mpatches.Patch(color=[1.00, 0.55, 0.00], label='Sacrum'),
        mpatches.Patch(color=LINE_TV,            label='TV mid'),
        mpatches.Patch(color=LINE_MINDIST,       label='closest approach'),
    ], loc='lower right', fontsize=6, framealpha=0.55)

    dist_str  = f'{dist_mm:.1f} mm' if np.isfinite(dist_mm) else 'N/A'
    contact   = dist_mm <= CONTACT_DIST_MM if np.isfinite(dist_mm) else False
    flag      = '✗ contact → II/III' if contact else '✓ no contact'
    ax.set_title(
        f'Type II/III check — {side_name} TP  (x={x_idx})\n'
        f'TP–Sacrum min dist: {dist_str}  {flag}',
        fontsize=10, color='white'
    )
    ax.axis('off')


# ── Sagittal TSS level confirmation  [row 2, col 0] ──────────────────────────

def _panel_sag_tss_confirm(ax,
                             sag_img_sl:  np.ndarray,
                             tss_sag_sl:  np.ndarray,
                             tv_name:     str,
                             z_tv_sag:    Optional[float],
                             z_md_sag:    Optional[float]):
    ax.imshow(norm(sag_img_sl).T, cmap='gray', origin='lower', alpha=0.80)
    patches = []
    for label, (color, name) in TSS_LABEL_COLORS.items():
        m = tss_sag_sl == label
        if not m.any():
            continue
        alpha = 0.70 if label in (L5_LABEL, L6_LABEL, SACRUM_LABEL) else 0.28
        overlay_mask(ax, m, color, alpha)
        patches.append(mpatches.Patch(color=color, label=name))
    if patches:
        ax.legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.55)
    _hline(ax, z_tv_sag, LINE_TV,      label='TV mid')
    _hline(ax, z_md_sag, LINE_MINDIST, label='min-dist')
    ax.set_title(f'Sagittal TSS — Level Confirm  (TV={tv_name})\n'
                 f'L5/L6 + Sacrum highlighted',
                 fontsize=10, color='white')
    ax.axis('off')


# ── Plain axial T2w  [row 2, col 1] ──────────────────────────────────────────

def _panel_axial_plain(ax, img_sl: np.ndarray, z_idx: int):
    ax.imshow(norm(img_sl).T, cmap='gray', origin='lower')
    ax.set_title(f'Axial T2w — TV mid  z={z_idx}\n(anatomy reference)',
                 fontsize=10, color='white')
    ax.axis('off')


# ── Classification summary  [row 2, col 2] ───────────────────────────────────

def _panel_summary(ax,
                    study_id:      str,
                    result:        Optional[dict],
                    tv_name:       str,
                    span_left_mm:  float,
                    span_right_mm: float,
                    dist_L_mm:     float,
                    dist_R_mm:     float):
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    def _d(v: float) -> str:
        return f'{v:.1f}' if np.isfinite(v) else 'N/A'

    lines = [
        f'Study  : {study_id}',
        f'TV     : {tv_name}',
        '',
        '─── Measured (from masks) ─────────────────',
        f'  Left  TP height : {_d(span_left_mm):>6} mm'
        f'  (thresh {TP_HEIGHT_MM:.0f} mm)',
        f'  Right TP height : {_d(span_right_mm):>6} mm'
        f'  (thresh {TP_HEIGHT_MM:.0f} mm)',
        f'  Left  TP–Sacrum : {_d(dist_L_mm):>6} mm'
        f'  (thresh {CONTACT_DIST_MM:.0f} mm)',
        f'  Right TP–Sacrum : {_d(dist_R_mm):>6} mm'
        f'  (thresh {CONTACT_DIST_MM:.0f} mm)',
        '',
    ]

    if result is None:
        lines += ['─── Classifier output ─────────────────────',
                  '  (run 04_detect_lstv.py + --lstv_json)']
    else:
        ct       = result.get('castellvi_type') or 'None'
        detected = result.get('lstv_detected', False)
        lines += [
            '─── Classifier output ─────────────────────',
            f'  Castellvi : {ct}',
            f'  LSTV      : {"YES  !" if detected else "No"}',
            '',
        ]
        for side in ('left', 'right'):
            sd = result.get(side) or {}
            if not sd:
                continue
            cls   = sd.get('classification', '?')
            tp_h  = sd.get('tp_height_mm', 0.0)
            tp_d  = sd.get('dist_mm', float('inf'))
            lines += [
                f'  {"Left " if side=="left" else "Right"} → {cls}',
                f'    classifier height : {tp_h:.1f} mm',
                f'    classifier dist   : {_d(tp_d)} mm',
            ]
            if sd.get('note'):
                lines.append(f'    NOTE: {sd["note"]}')
            lines.append('')

        if result.get('errors'):
            lines += ['  Errors:'] + [f'    {e}' for e in result['errors']]

    ax.text(0.05, 0.97, '\n'.join(lines),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=8.2, family='monospace', color='white',
            linespacing=1.35)
    ax.set_title('Classification Summary', fontsize=11, color='white')


# ============================================================================
# CORE VISUALIZER
# ============================================================================

def visualize_study(
    study_id:      str,
    registered_dir: Path,
    nifti_dir:     Path,
    spineps_dir:   Path,
    totalspine_dir: Path,
    output_dir:    Path,
    result:        Optional[dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_lstv_overlay.png"

    reg = registered_dir / study_id

    def try_load(path, label):
        p = Path(path) if path is not None else None
        if p is not None and p.exists():
            try:
                return load_canonical(p)
            except Exception as e:
                logger.warning(f"  [{study_id}] Cannot load {label}: {e}")
        elif p is not None:
            logger.warning(f"  [{study_id}] Missing: {p.name}")
        return None, None

    # ── Load volumes ─────────────────────────────────────────────────────────
    spineps_reg, spineps_nii = try_load(
        reg / f"{study_id}_spineps_reg.nii.gz", 'SPINEPS reg')
    tss_reg, _ = try_load(
        reg / f"{study_id}_tss_reg.nii.gz",     'TSS reg')

    sag_bg,      sag_nii    = try_load(find_t2w(nifti_dir, study_id, 'sag'), 'Sag T2w')
    orig_spineps, _         = try_load(find_original_spineps_seg(spineps_dir, study_id), 'SPINEPS orig')
    orig_tss_sag, _         = try_load(find_original_tss_sag(totalspine_dir, study_id),  'TSS orig sag')

    ax_bg, ax_bg_nii = try_load(find_t2w(nifti_dir, study_id, 'ax'), 'Axial T2w')
    if ax_bg is None:
        ax_bg, ax_bg_nii = spineps_reg, spineps_nii

    tss_native, _ = try_load(find_native_axial_tss(totalspine_dir, study_id), 'TSS native axial')
    using_native  = tss_native is not None

    if ax_bg is None:
        logger.error(f"  [{study_id}] No axial background — skipping")
        return

    # ── Type casts ────────────────────────────────────────────────────────────
    if spineps_reg  is not None: spineps_reg  = spineps_reg.astype(int)
    if tss_reg      is not None: tss_reg      = tss_reg.astype(int)
    if orig_spineps is not None: orig_spineps = orig_spineps.astype(int)
    if orig_tss_sag is not None: orig_tss_sag = orig_tss_sag.astype(int)
    if tss_native   is not None: tss_native   = tss_native.astype(int)

    vox_ax  = voxel_size_mm(ax_bg_nii)
    vox_sag = voxel_size_mm(sag_nii) if sag_nii is not None else vox_ax

    # ── Target vertebra ───────────────────────────────────────────────────────
    tss_labels = tss_reg if tss_reg is not None else tss_native
    tv_label   = (L6_LABEL
                  if tss_labels is not None and L6_LABEL in np.unique(tss_labels)
                  else L5_LABEL)
    tv_name    = 'L6' if tv_label == L6_LABEL else 'L5'

    # ── Axial masks ───────────────────────────────────────────────────────────
    zeros = np.zeros(ax_bg.shape, dtype=bool)
    z_range = get_tv_z_range(tss_labels, tv_label) if tss_labels is not None else None

    if z_range is not None and spineps_reg is not None:
        z_min, z_max = z_range
        z_tv         = (z_min + z_max) // 2
        tp_left_ax   = isolate_tp_at_tv(spineps_reg, TP_LEFT_LABEL,  z_min, z_max)
        tp_right_ax  = isolate_tp_at_tv(spineps_reg, TP_RIGHT_LABEL, z_min, z_max)
    else:
        z_tv = ax_bg.shape[2] // 2
        tp_left_ax = tp_right_ax = zeros

    sacrum_ax = (tss_labels == SACRUM_LABEL) if tss_labels is not None else zeros

    # ── Min-distance slices (per side) ───────────────────────────────────────
    z_md_L, dist_L = min_dist_z_for_tp_sacrum(tp_left_ax,  sacrum_ax, vox_ax)
    z_md_R, dist_R = min_dist_z_for_tp_sacrum(tp_right_ax, sacrum_ax, vox_ax)
    # Combined: pick side with smaller gap for the shared axial panel
    if np.isfinite(dist_L) or np.isfinite(dist_R):
        z_md_combined = z_md_L if dist_L <= dist_R else z_md_R
    else:
        z_md_combined = z_tv
    min_dist_combined = min(dist_L, dist_R)

    logger.info(f"  [{study_id}]  z_tv={z_tv}  "
                f"z_md_L={z_md_L}({dist_L:.1f}mm)  z_md_R={z_md_R}({dist_R:.1f}mm)")

    # ── Max-height sagittal slices (per side) ─────────────────────────────────
    x_left,  span_L = best_x_for_tp_height(orig_spineps, orig_tss_sag,
                                            TP_LEFT_LABEL,  tv_label, vox_sag[2])
    x_right, span_R = best_x_for_tp_height(orig_spineps, orig_tss_sag,
                                            TP_RIGHT_LABEL, tv_label, vox_sag[2])

    logger.info(f"  [{study_id}]  x_left={x_left}({span_L:.1f}mm)  "
                f"x_right={x_right}({span_R:.1f}mm)")

    # ── Reference axial z for [0,2]: midpoint between z_tv and z_md_combined ──
    z_ref = (z_tv + z_md_combined) // 2

    # ── Sagittal midline x for TSS confirm ───────────────────────────────────
    x_mid = (orig_tss_sag.shape[0] // 2 if orig_tss_sag is not None
             else sag_bg.shape[0] // 2 if sag_bg is not None else 0)

    # ── Axial z → sag z conversions ──────────────────────────────────────────
    z_tv_sag    = ax_z_to_sag_z(ax_bg_nii, sag_nii, z_tv)
    z_md_L_sag  = ax_z_to_sag_z(ax_bg_nii, sag_nii, z_md_L)
    z_md_R_sag  = ax_z_to_sag_z(ax_bg_nii, sag_nii, z_md_R)
    z_md_c_sag  = ax_z_to_sag_z(ax_bg_nii, sag_nii, z_md_combined)

    # ── TSS display volume ────────────────────────────────────────────────────
    tss_disp = tss_native if using_native else tss_reg

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(21, 21))
    fig.patch.set_facecolor('#0d0d1a')
    for a in axes.flat:
        a.set_facecolor('#0d0d1a')

    castellvi = result.get('castellvi_type') if result else None
    fig.suptitle(
        f"Study {study_id}   |   Castellvi: {castellvi or 'N/A'}   |   TV: {tv_name}",
        fontsize=15, color='white', y=0.999,
    )

    # ── ROW 0 — Type I check ─────────────────────────────────────────────────

    # [0,0]  Left  TP sagittal max height
    if sag_bg is not None:
        _panel_sag_tp_height(
            axes[0, 0], sag_bg, orig_spineps, orig_tss_sag,
            TP_LEFT_LABEL, 'Left',
            x_left, span_L, z_tv_sag, z_md_L_sag, tv_name,
        )
    else:
        _unavailable(axes[0, 0], 'Sagittal T2w not found')

    # [0,1]  Right TP sagittal max height
    if sag_bg is not None:
        _panel_sag_tp_height(
            axes[0, 1], sag_bg, orig_spineps, orig_tss_sag,
            TP_RIGHT_LABEL, 'Right',
            x_right, span_R, z_tv_sag, z_md_R_sag, tv_name,
        )
    else:
        _unavailable(axes[0, 1], 'Sagittal T2w not found')

    # [0,2]  TSS Labels axial at z_ref (between z_tv and z_md)
    tss_sl_ref = _tss_axial_slice(ax_bg, tss_disp, z_ref)
    _panel_tss_axial(
        axes[0, 2],
        _ax_sl(ax_bg, z_ref),
        tss_sl_ref,
        None, None, None,          # no extra TP overlays on this panel
        using_native, z_ref,
        subtitle=f'z_ref = mean(z_tv={z_tv}, z_md={z_md_combined})',
    )

    # ── ROW 1 — Type II / III check ──────────────────────────────────────────

    # [1,0]  Left  TP proximity sagittal (same x as row 0, orange line primary)
    if sag_bg is not None:
        _panel_sag_tp_proximity(
            axes[1, 0], sag_bg, orig_spineps, orig_tss_sag,
            TP_LEFT_LABEL, 'Left',
            x_left, span_L, dist_L,
            z_tv_sag, z_md_L_sag, tv_name,
        )
    else:
        _unavailable(axes[1, 0], 'Sagittal T2w not found')

    # [1,1]  Right TP proximity sagittal
    if sag_bg is not None:
        _panel_sag_tp_proximity(
            axes[1, 1], sag_bg, orig_spineps, orig_tss_sag,
            TP_RIGHT_LABEL, 'Right',
            x_right, span_R, dist_R,
            z_tv_sag, z_md_R_sag, tv_name,
        )
    else:
        _unavailable(axes[1, 1], 'Sagittal T2w not found')

    # [1,2]  TSS Labels axial at z_md_combined + undilated TP + sacrum masks
    tss_sl_md = _tss_axial_slice(ax_bg, tss_disp, z_md_combined)
    dist_label = (f'L={dist_L:.1f}mm  R={dist_R:.1f}mm'
                  if (np.isfinite(dist_L) or np.isfinite(dist_R)) else 'no masks')
    _panel_tss_axial(
        axes[1, 2],
        _ax_sl(ax_bg, z_md_combined),
        tss_sl_md,
        _ax_sl(tp_left_ax,  z_md_combined),
        _ax_sl(tp_right_ax, z_md_combined),
        _ax_sl(sacrum_ax,   z_md_combined),
        using_native, z_md_combined,
        subtitle=f'Min-dist slice  [{dist_label}]  [Type II/III]',
    )

    # ── ROW 2 — Context + summary ─────────────────────────────────────────────

    # [2,0]  Sagittal TSS level confirmation
    if orig_tss_sag is not None:
        sag_bg_sl = (_sag_sl(sag_bg, x_mid)
                     if sag_bg is not None
                     else _sag_sl(orig_tss_sag, x_mid).astype(float))
        _panel_sag_tss_confirm(
            axes[2, 0],
            sag_bg_sl,
            _sag_sl(orig_tss_sag, x_mid),
            tv_name, z_tv_sag, z_md_c_sag,
        )
    else:
        _unavailable(axes[2, 0], 'Sagittal TSS not found')

    # [2,1]  Plain axial T2w at TV mid
    _panel_axial_plain(axes[2, 1], _ax_sl(ax_bg, z_tv), z_tv)

    # [2,2]  Classification summary with measured values
    _panel_summary(
        axes[2, 2], study_id, result, tv_name,
        span_left_mm=span_L, span_right_mm=span_R,
        dist_L_mm=dist_L, dist_R_mm=dist_R,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  [{study_id}] OK -> {out_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV Overlay Visualizer v7')
    parser.add_argument('--registered_dir',  required=True)
    parser.add_argument('--nifti_dir',       required=True)
    parser.add_argument('--spineps_dir',     required=True)
    parser.add_argument('--totalspine_dir',  required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--study_id',        default=None)
    parser.add_argument('--lstv_json',       default=None)
    args = parser.parse_args()

    registered_dir = Path(args.registered_dir)
    nifti_dir      = Path(args.nifti_dir)
    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)

    results_by_id = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                results_by_id = {r['study_id']: r for r in json.load(f)}
            logger.info(f"Loaded {len(results_by_id)} detection results")

    if args.study_id:
        study_ids = [args.study_id]
    else:
        study_ids = sorted(d.name for d in registered_dir.iterdir() if d.is_dir())
        logger.info(f"Batch mode: {len(study_ids)} studies")

    errors = 0
    for sid in study_ids:
        try:
            visualize_study(
                study_id       = sid,
                registered_dir = registered_dir,
                nifti_dir      = nifti_dir,
                spineps_dir    = spineps_dir,
                totalspine_dir = totalspine_dir,
                output_dir     = output_dir,
                result         = results_by_id.get(sid),
            )
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())
            errors += 1

    logger.info(f"Done. {len(study_ids)-errors}/{len(study_ids)} PNGs -> {output_dir}")


if __name__ == '__main__':
    main()
