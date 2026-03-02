"""
lstv_angles_v4.py – Seilanian Toosi 2025 angle calculations via midline sagittal slice
========================================================================================
All five paper angles (A, B, C, D, D1) are defined on lines drawn in a midsagittal view
(lateral radiograph or sagittal MRI).  Rather than fighting 3D plane-fitting instability
we reconstruct an optimal midsagittal 2D slice from the 3D segmentation and measure
every angle directly in that plane – exactly as the paper figures illustrate.

Coordinate conventions in the 2D working slice
------------------------------------------------
  x-axis  = AP direction, increasing anterior
  y-axis  = CC direction, increasing cranial
Slopes are expressed as  m = ΔCC / ΔAP  (rise / run in the sagittal plane).

Angle helpers
  _angle_from_horizontal(m)  → angle of a line from horizontal [0°,90°], = arctan(|m|)
  _angle_between_lines(m1,m2) → acute inter-line angle [0°,90°]
  For A-angle: angle of sacral endplate from horizontal  (paper Fig 2a; Ferguson angle)
  For B, D, D1: acute angle between two surface lines
  For C: largest angle between posterior-body lines of {TV-1, TV, TV+1} and {S1, S2}

Labels (TotalSpineSeg TSS):
  Sacrum = 50 | Lumbar L5..L1 = 20..24 | Transitional = 25 (if renumbered externally)
TSS instance labels passed in  masks  dict  are already resolved by the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class LSTVAngles:
    """Measured angles (degrees) for one case."""
    A: Optional[float] = None   # sacral superior surface vs horizontal
    B: Optional[float] = None   # L3 superior endplate vs sacral superior surface
    C: Optional[float] = None   # largest posterior-body line angle (TV±1 / sacrum±1)
    D: Optional[float] = None   # sacrum sup vs TV sup (lumbosacral angle)
    D1: Optional[float] = None  # TV sup vs TV-1 sup
    delta: Optional[float] = None  # D − D1  (signed; ≤8.5° → Type-2 LSTV)

    # Sanity flags  (True = value present AND within physiologic range)
    flags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "A": self.A, "B": self.B, "C": self.C,
            "D": self.D, "D1": self.D1, "delta": self.delta,
            "flags": self.flags,
        }


# ---------------------------------------------------------------------------
# Sanity bounds (Seilanian Toosi 2025, Table 1 + Fig discussion)
# ---------------------------------------------------------------------------

BOUNDS = {
    "A":  (12.0,  55.0),
    "B":  (20.0,  70.0),
    "C":  (10.0,  60.0),
    "D":  (5.0,   50.0),
    "D1": (0.0,   48.0),
    "delta": (-18.0, 40.0),
}


# ============================================================
# Primary entry point
# ============================================================

def compute_angles(
    label_volume: np.ndarray,
    *,
    tv_label: int,
    tv_minus1_label: int,
    sacrum_label: int = 50,
    l3_label: Optional[int] = None,
    sp_corpus_volume: Optional[np.ndarray] = None,
    voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> LSTVAngles:
    """
    Compute LSTV classification angles from a 3D integer label volume.

    Parameters
    ----------
    label_volume : 3-D integer array of segmentation labels (voxel space, any orientation)
    tv_label : label id of the transitional vertebra (TV / most caudal lumbar)
    tv_minus1_label : label id of the vertebra immediately cranial to TV
    sacrum_label : label id of the sacrum (default 50 for TSS)
    l3_label : label id of L3 (for B-angle); if None B is skipped
    sp_corpus_volume : optional SPINEPS corpus sub-label volume for refined body masks
    voxel_spacing_mm : (d0, d1, d2) voxel size along each array axis in mm

    Returns
    -------
    LSTVAngles
    """
    spacing = np.array(voxel_spacing_mm, dtype=float)

    # -----------------------------------------------------------------------
    # 1.  Identify anatomical axes
    # -----------------------------------------------------------------------
    cc_axis, ap_axis, ml_axis = _identify_axes(label_volume, sacrum_label, tv_minus1_label)
    logger.debug("Axes — CC:%d  AP:%d  ML:%d", cc_axis, ap_axis, ml_axis)

    # -----------------------------------------------------------------------
    # 2.  Find the optimal midline ML slice column
    # -----------------------------------------------------------------------
    all_spine_labels = [sacrum_label, tv_label, tv_minus1_label]
    if l3_label is not None:
        all_spine_labels.append(l3_label)

    # For C-angle we also need TV+1 (one cranial to TV-1); try tv_minus1_label-1
    # if available (caller should pass as separate param in production; for now infer)
    tv_plus1_label: Optional[int] = None  # will be handled if present

    ml_center, ml_band = _optimal_midline(
        label_volume, all_spine_labels, ml_axis, spacing
    )
    logger.debug("Optimal midline: ML index %d (band ±%d)", ml_center, ml_band)

    # -----------------------------------------------------------------------
    # 3.  Extract 2D sagittal slices for each vertebra
    # -----------------------------------------------------------------------
    slice_info = _extract_sagittal_slices(
        label_volume,
        labels=all_spine_labels,
        cc_axis=cc_axis,
        ap_axis=ap_axis,
        ml_axis=ml_axis,
        ml_center=ml_center,
        ml_band=ml_band,
        sp_corpus=sp_corpus_volume,
    )
    # slice_info[label] = 2D binary array, axes = (ap_idx, cc_idx)
    # Physical size per pixel: (spacing[ap_axis], spacing[cc_axis])

    # -----------------------------------------------------------------------
    # 4.  Fit endplate lines in 2D and compute angles
    # -----------------------------------------------------------------------
    result = LSTVAngles()

    # Precompute sacrum superior slope (needed for A, B, D)
    m_s_sup: Optional[float] = None
    s_mask = slice_info.get(sacrum_label)
    if s_mask is not None and s_mask.any():
        m_s_sup = _fit_endplate_2d(s_mask, surface="superior",
                                    sp_ap=spacing[ap_axis], sp_cc=spacing[cc_axis])

    # A-angle ------------------------------------------------------------------
    if m_s_sup is not None:
        result.A = _angle_from_horizontal(m_s_sup)
        logger.debug("A-angle m=%.3f  →  A=%.1f°", m_s_sup, result.A)

    # B-angle ------------------------------------------------------------------
    if l3_label is not None:
        l3_mask = slice_info.get(l3_label)
        if l3_mask is not None and l3_mask.any() and m_s_sup is not None:
            m_l3_sup = _fit_endplate_2d(l3_mask, surface="superior",
                                         sp_ap=spacing[ap_axis], sp_cc=spacing[cc_axis])
            if m_l3_sup is not None:
                result.B = _angle_between_lines(m_l3_sup, m_s_sup)
                logger.debug("B-angle m_L3=%.3f m_S=%.3f  →  B=%.1f°",
                              m_l3_sup, m_s_sup, result.B)

    # D & D1 angles ------------------------------------------------------------
    tv_mask = slice_info.get(tv_label)
    tv1_mask = slice_info.get(tv_minus1_label)

    m_tv_sup = None
    m_tv1_sup = None

    if tv_mask is not None and tv_mask.any():
        m_tv_sup = _fit_endplate_2d(tv_mask, surface="superior",
                                     sp_ap=spacing[ap_axis], sp_cc=spacing[cc_axis])
    if tv1_mask is not None and tv1_mask.any():
        m_tv1_sup = _fit_endplate_2d(tv1_mask, surface="superior",
                                      sp_ap=spacing[ap_axis], sp_cc=spacing[cc_axis])

    if m_s_sup is not None and m_tv_sup is not None:  # noqa: F821
        result.D = _angle_between_lines(m_s_sup, m_tv_sup)
        logger.debug("D-angle m_S=%.3f m_TV=%.3f  →  D=%.1f°",
                      m_s_sup, m_tv_sup, result.D)

    if m_tv_sup is not None and m_tv1_sup is not None:
        result.D1 = _angle_between_lines(m_tv_sup, m_tv1_sup)
        logger.debug("D1-angle m_TV=%.3f m_TV1=%.3f  →  D1=%.1f°",
                      m_tv_sup, m_tv1_sup, result.D1)

    if result.D is not None and result.D1 is not None:
        result.delta = result.D - result.D1
        logger.debug("delta = D(%.1f) − D1(%.1f) = %.1f°", result.D, result.D1, result.delta)

    # C-angle ------------------------------------------------------------------
    # Posterior body lines of TV-1 and sacrum (and TV if available)
    # C = largest angle among pairs of posterior-margin lines
    c_labels = [tv_minus1_label, sacrum_label]
    if tv_label in slice_info:
        c_labels.append(tv_label)
    c_slopes = []
    for lbl in c_labels:
        msk = slice_info.get(lbl)
        if msk is not None and msk.any():
            m_post = _fit_posterior_wall_2d(msk, sp_ap=spacing[ap_axis], sp_cc=spacing[cc_axis])
            if m_post is not None:
                c_slopes.append(m_post)
                logger.debug("C posterior slope label=%d  m=%.3f", lbl, m_post)

    if len(c_slopes) >= 2:
        from itertools import combinations
        result.C = max(
            _angle_between_lines(m1, m2)
            for m1, m2 in combinations(c_slopes, 2)
        )
        logger.debug("C-angle = %.1f°", result.C)

    # -----------------------------------------------------------------------
    # 5.  Sanity checks
    # -----------------------------------------------------------------------
    _apply_sanity_flags(result)

    return result


# ============================================================
# Axis identification
# ============================================================

def _identify_axes(
    vol: np.ndarray,
    sacrum_label: int,
    cranial_label: int,
) -> Tuple[int, int, int]:
    """
    Determine which array axis is CC, AP, ML.

    Strategy:
      CC: axis along which the sacrum centroid and the cranial-label centroid
          are maximally separated (normalised by array extent).
      ML: of the remaining two axes, the one along which the combined spine
          mask has the greatest normalised extent (spine wider in ML).
      AP: the remaining axis.
    """
    ndim = vol.ndim
    assert ndim == 3, "Volume must be 3-D"

    def centroid1d(mask: np.ndarray, axis: int) -> float:
        coords = np.where(mask)[axis]
        return float(coords.mean()) if len(coords) else 0.0

    sac_mask = vol == sacrum_label
    cra_mask = vol == cranial_label

    if not sac_mask.any() or not cra_mask.any():
        # fallback: assume standard (ML, AP, CC) = (0, 1, 2)
        logger.warning("Cannot detect axes – defaulting to (CC=2, AP=1, ML=0)")
        return 2, 1, 0

    separations = []
    for ax in range(3):
        c_s = centroid1d(sac_mask, ax)
        c_c = centroid1d(cra_mask, ax)
        sep = abs(c_s - c_c) / vol.shape[ax]
        separations.append(sep)

    cc_axis = int(np.argmax(separations))

    # Among the remaining two axes, ML = axis with greatest normalised extent
    remaining = [ax for ax in range(3) if ax != cc_axis]
    combined = sac_mask | cra_mask
    extents = []
    for ax in remaining:
        proj = combined.any(axis=tuple(a for a in range(3) if a != ax))
        occupied = int(np.sum(proj))
        extents.append(occupied / vol.shape[ax])

    ml_axis = remaining[int(np.argmax(extents))]
    ap_axis = [ax for ax in remaining if ax != ml_axis][0]

    return cc_axis, ap_axis, ml_axis


# ============================================================
# Optimal midline detection
# ============================================================

def _optimal_midline(
    vol: np.ndarray,
    labels: list,
    ml_axis: int,
    spacing: np.ndarray,
    band_mm: float = 10.0,
) -> Tuple[int, int]:
    """
    Find the ML index that maximises bone coverage across all spine labels.

    Returns (ml_center_index, half_band_voxels)
    where  ±half_band  defines the thick slab used for the slice projection.

    'Optimal midline' is determined by:
      1. Build a combined mask of all relevant spine labels.
      2. Sum voxels along each ML column (collapsing AP and CC).
      3. Smooth that 1-D profile with a Gaussian (σ = 3 voxels) to find the
         centre of the bone column rather than a noisy peak.
      4. ml_center = argmax of the smoothed profile.
    """
    combined = np.zeros(vol.shape, dtype=bool)
    for lbl in labels:
        combined |= (vol == lbl)

    if not combined.any():
        logger.warning("No spine voxels found for midline detection – using centre of volume")
        return vol.shape[ml_axis] // 2, 2

    # Sum along all axes except ml_axis
    sum_axes = tuple(ax for ax in range(vol.ndim) if ax != ml_axis)
    profile = combined.sum(axis=sum_axes).astype(float)

    # Smooth
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(profile, sigma=3.0)
    ml_center = int(np.argmax(smoothed))

    # Half-band in voxels from physical band_mm
    ml_spacing = float(spacing[ml_axis])
    half_band = max(1, int(np.round(band_mm / (2.0 * ml_spacing))))

    logger.debug(
        "Midline profile peak at ML=%d (band ±%d vox = ±%.1fmm); "
        "profile max=%.0f vox",
        ml_center, half_band, half_band * ml_spacing, smoothed[ml_center],
    )
    return ml_center, half_band


# ============================================================
# Sagittal slice extraction
# ============================================================

def _extract_sagittal_slices(
    vol: np.ndarray,
    labels: list,
    *,
    cc_axis: int,
    ap_axis: int,
    ml_axis: int,
    ml_center: int,
    ml_band: int,
    sp_corpus: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    For each label, extract a 2-D binary mask in the midline slab.

    The slab spans ml_center ± ml_band along ml_axis.
    Within the slab we take a MAX projection along ml_axis  → any voxel of
    that label present anywhere in the slab appears in the 2-D slice.
    (This is equivalent to what you'd see on a thick-slab MIP.)

    The returned 2-D arrays have axes (ap_idx, cc_idx) regardless of the
    input volume's axis ordering.
    """
    lo = max(0, ml_center - ml_band)
    hi = min(vol.shape[ml_axis] - 1, ml_center + ml_band) + 1

    # Slicing helper: build a tuple of slices that grabs the ML band
    def ml_slab(v: np.ndarray) -> np.ndarray:
        slc = [slice(None)] * v.ndim
        slc[ml_axis] = slice(lo, hi)
        return v[tuple(slc)]

    result: Dict[int, np.ndarray] = {}

    for lbl in labels:
        mask = vol == lbl
        if sp_corpus is not None:
            # Refine: keep only voxels that are also in corpus
            mask = mask & (sp_corpus > 0)
            if not mask.any():
                mask = vol == lbl  # fall back if corpus stripped too much

        slab = ml_slab(mask)  # shape still 3-D

        # Max-project along the slab's ml_axis position
        slab_2d = slab.any(axis=ml_axis)  # True wherever any slab voxel is present
        # slab_2d has the two remaining axes in their original order

        # Reorder to (ap, cc)
        remaining_axes = [ax for ax in range(3) if ax != ml_axis]
        # remaining_axes[i] → physical axis; we want order (ap_axis, cc_axis)
        pos_ap = remaining_axes.index(ap_axis)
        pos_cc = remaining_axes.index(cc_axis)
        if pos_ap == 0 and pos_cc == 1:
            result[lbl] = slab_2d
        else:
            result[lbl] = slab_2d.T  # swap to (ap, cc)

    return result


# ============================================================
# 2-D line fitting helpers
# ============================================================

def _fit_endplate_2d(
    mask_2d: np.ndarray,
    surface: str = "superior",
    sp_ap: float = 1.0,
    sp_cc: float = 1.0,
) -> Optional[float]:
    """
    Fit a line to the superior or inferior endplate of a vertebra in 2-D.

    mask_2d : (n_ap, n_cc) binary array, axes = (AP, CC)
    surface : 'superior' → upper (cranial) boundary  |  'inferior' → lower boundary

    Returns the slope  m = ΔCC_mm / ΔAP_mm  of the fitted line, or None if
    there are fewer than 4 usable points.

    Method:
      For each AP column, find the extreme CC index (max for superior, min for
      inferior).  Convert to physical coordinates (mm).  Ordinary least-squares
      linear regression of CC_mm on AP_mm.  Outlier-resistant: points >2 std
      from the fitted line are removed and the fit is repeated once.
    """
    n_ap, n_cc = mask_2d.shape
    ap_pts: list = []
    cc_pts: list = []

    for ap_i in range(n_ap):
        col = np.where(mask_2d[ap_i, :])[0]
        if len(col) == 0:
            continue
        cc_i = int(col.max()) if surface == "superior" else int(col.min())
        ap_pts.append(ap_i * sp_ap)
        cc_pts.append(cc_i * sp_cc)

    if len(ap_pts) < 4:
        return None

    ap_arr = np.array(ap_pts)
    cc_arr = np.array(cc_pts)

    m, _ = _ols_slope(ap_arr, cc_arr)

    # One pass of outlier rejection (|residual| > 2σ)
    residuals = cc_arr - (m * ap_arr + _ols_intercept(ap_arr, cc_arr, m))
    std = residuals.std()
    if std > 0:
        keep = np.abs(residuals) <= 2.0 * std
        if keep.sum() >= 4:
            m, _ = _ols_slope(ap_arr[keep], cc_arr[keep])

    return m


def _fit_posterior_wall_2d(
    mask_2d: np.ndarray,
    sp_ap: float = 1.0,
    sp_cc: float = 1.0,
) -> Optional[float]:
    """
    Fit a line to the posterior vertebral wall (posterior body margin).

    The posterior wall is the most-posterior (minimum AP index) occupied voxel
    at each CC level.  Assumes AP index increases anteriorly.

    Returns slope  m = ΔCC_mm / ΔAP_mm  (same convention as endplate fitter).
    """
    n_ap, n_cc = mask_2d.shape
    ap_pts: list = []
    cc_pts: list = []

    for cc_i in range(n_cc):
        row = np.where(mask_2d[:, cc_i])[0]
        if len(row) == 0:
            continue
        ap_i = int(row.min())  # most posterior
        ap_pts.append(ap_i * sp_ap)
        cc_pts.append(cc_i * sp_cc)

    if len(ap_pts) < 4:
        return None

    ap_arr = np.array(ap_pts)
    cc_arr = np.array(cc_pts)
    m, _ = _ols_slope(ap_arr, cc_arr)
    return m


# ============================================================
# Angle calculation from slopes
# ============================================================

def _angle_from_horizontal(m: float) -> float:
    """
    Angle of a line from the horizontal axis, degrees [0°, 90°].

    With x=AP, y=CC:  a horizontal endplate has m=0 → 0°.
    The sacral endplate at ~37° from horizontal has m = tan(37°) ≈ 0.75.
    """
    return float(np.degrees(np.arctan(abs(m))))


def _angle_between_lines(m1: float, m2: float) -> float:
    """
    Acute angle between two lines with slopes m1, m2 (degrees, [0°, 90°]).

    Uses the standard formula:  tan θ = |m1-m2| / (1 + m1·m2)
    Falls back to 90° if lines are perpendicular (denominator ≈ 0).
    """
    denom = 1.0 + m1 * m2
    if abs(denom) < 1e-6:
        return 90.0
    angle = float(np.degrees(np.arctan(abs(m1 - m2) / abs(denom))))
    return angle


# ============================================================
# OLS helpers
# ============================================================

def _ols_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Ordinary least-squares: returns (slope m, slope_se) for y ~ m*x + b.
    """
    x = x - x.mean()
    y = y - y.mean()
    ss_xx = float((x * x).sum())
    ss_xy = float((x * y).sum())
    if ss_xx < 1e-12:
        return 0.0, np.inf
    m = ss_xy / ss_xx
    residuals = y - m * x
    n = len(x)
    se = float(np.sqrt((residuals ** 2).sum() / max(n - 2, 1) / ss_xx)) if n > 2 else np.inf
    return m, se


def _ols_intercept(x: np.ndarray, y: np.ndarray, m: float) -> float:
    return float(y.mean() - m * x.mean())


# ============================================================
# Sanity checks
# ============================================================

def _apply_sanity_flags(result: LSTVAngles) -> None:
    for name, (lo, hi) in BOUNDS.items():
        val = getattr(result, name, None)
        if val is None:
            result.flags[name] = "MISSING"
        elif not (lo <= val <= hi):
            result.flags[name] = f"OUT_OF_RANGE [{lo},{hi}] got {val:.1f}"
        else:
            result.flags[name] = "OK"

    # Anatomy cross-check: D should exceed D1 in most normal anatomy
    if result.D is not None and result.D1 is not None:
        if result.D < result.D1:
            result.flags["D<<D1"] = (
                f"ANATOMY WARN: D({result.D:.1f}) < D1({result.D1:.1f}) — check TV labeling"
            )


# ============================================================
# Diagnostic / visualisation helpers
# ============================================================

def debug_slice_figure(
    slice_dict: Dict[int, np.ndarray],
    angles: LSTVAngles,
    label_names: Optional[Dict[int, str]] = None,
    out_path: Optional[str] = None,
) -> None:
    """
    Render the midline sagittal slices overlaid with fitted endplate lines.
    Saves or shows the figure (requires matplotlib).
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if out_path else "TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        logger.warning("matplotlib not available – skipping debug figure")
        return

    colours = {
        50: "#FF6B6B",   # sacrum – red
        20: "#4ECDC4",   # L5 – teal
        21: "#45B7D1",   # L4 – blue
        22: "#96CEB4",   # L3 – green
        23: "#FFEAA7",   # L2 – yellow
        24: "#DDA0DD",   # L1 – plum
        25: "#FFA500",   # TV – orange (if renumbered)
    }

    fig, ax = plt.subplots(figsize=(7, 12))
    ax.set_facecolor("#1a1a2e")

    for lbl, mask in slice_dict.items():
        if not mask.any():
            continue
        colour = colours.get(lbl, "#AAAAAA")
        # Overlay semitransparent filled region
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(colour)
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = 0.55
        ax.imshow(rgba.T, origin="lower", aspect="auto")

    ax.set_xlabel("AP →")
    ax.set_ylabel("CC (cranial) →")
    angle_text = (
        f"A={angles.A:.1f}°  B={angles.B or '–'}°  C={angles.C or '–'}°\n"
        f"D={angles.D or '–'}°  D1={angles.D1 or '–'}°  δ={angles.delta or '–'}°"
    )
    ax.set_title(f"Midline sagittal slice\n{angle_text}", fontsize=10)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Debug figure saved to %s", out_path)
    else:
        plt.show()
    plt.close(fig)


# ============================================================
# Pipeline convenience wrapper (called from 04_detect_lstv.py)
# ============================================================

def compute_angles_from_spineps(
    tss_label_path: str,
    sp_corpus_path: Optional[str],
    tv_label: int,
    tv_minus1_label: int,
    sacrum_label: int = 50,
    l3_label: Optional[int] = None,
    *,
    verbose: bool = False,
) -> LSTVAngles:
    """
    Drop-in replacement for the old lstv_angles.compute_angles().

    Parameters
    ----------
    tss_label_path  : path to TotalSpineSeg NIfTI label map
    sp_corpus_path  : path to SPINEPS sub-corpus label NIfTI (or None)
    tv_label        : integer label of the transitional vertebra in tss_label_path
    tv_minus1_label : integer label of the supra-adjacent vertebra
    sacrum_label    : TSS sacrum label (default 50)
    l3_label        : TSS L3 label (for B-angle; None to skip)
    verbose         : enable DEBUG logging
    """
    import nibabel as nib

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    nii = nib.load(tss_label_path)
    vol = np.asarray(nii.dataobj, dtype=np.int32)
    spacing = np.abs(np.diag(nii.affine)[:3])  # voxel sizes in mm

    sp_corpus_vol = None
    if sp_corpus_path is not None:
        try:
            sp_corpus_vol = np.asarray(nib.load(sp_corpus_path).dataobj, dtype=np.int32)
        except Exception as exc:
            logger.warning("Could not load sp_corpus_path %s: %s", sp_corpus_path, exc)

    return compute_angles(
        vol,
        tv_label=tv_label,
        tv_minus1_label=tv_minus1_label,
        sacrum_label=sacrum_label,
        l3_label=l3_label,
        sp_corpus_volume=sp_corpus_vol,
        voxel_spacing_mm=tuple(float(s) for s in spacing),
    )


# ============================================================
# lstv_engine.py compatibility layer
# ============================================================
# lstv_engine.py expects:
#   from lstv_angles import compute_vertebral_angles, apply_angle_lr_updates
# and an angle_result object with specific attributes.
# This shim bridges the v4 API to that interface.

@dataclass
class RadiologicCriterion:
    """Minimal criterion record (mirrors lstv_engine.py RadiologicCriterion)."""
    name:      str
    value:     str
    direction: str
    strength:  str
    lr_sac:    float
    lr_lumb:   float
    citation:  str
    finding:   str


@dataclass
class VertebralAnglesResult:
    """Result object compatible with lstv_engine.py step 8.5."""
    angles_available:      bool              = False
    angle_lr_keys_fired:   list              = field(default_factory=list)

    a_angle_deg:           Optional[float]   = None
    b_angle_deg:           Optional[float]   = None
    c_angle_deg:           Optional[float]   = None
    d_angle_deg:           Optional[float]   = None
    d1_angle_deg:          Optional[float]   = None
    delta_angle_deg:       Optional[float]   = None

    # Positive classification flags
    delta_positive:        bool  = False   # delta ≤ 8.5° → Type-2 LSTV
    delta_any_lstv:        bool  = False   # delta ≤ 15°  (borderline)
    c_positive:            bool  = False   # C ≤ 35.5°    → LSTV signal
    a_angle_elevated:      bool  = False   # A > 41°      → sacralization signal

    # Sanity flags forwarded from LSTVAngles.flags
    angle_flags:           dict  = field(default_factory=dict)
    computation_notes:     list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'angles_available':    self.angles_available,
            'a_angle_deg':         self.a_angle_deg,
            'b_angle_deg':         self.b_angle_deg,
            'c_angle_deg':         self.c_angle_deg,
            'd_angle_deg':         self.d_angle_deg,
            'd1_angle_deg':        self.d1_angle_deg,
            'delta_angle_deg':     self.delta_angle_deg,
            'delta_positive':      self.delta_positive,
            'delta_any_lstv':      self.delta_any_lstv,
            'c_positive':          self.c_positive,
            'a_angle_elevated':    self.a_angle_elevated,
            'angle_lr_keys_fired': self.angle_lr_keys_fired,
            'angle_flags':         self.angle_flags,
            'computation_notes':   self.computation_notes,
        }


# VERIDAH → TSS label mapping (mirrors lstv_engine.py VD_TO_TSS_VERT)
_VD_TO_TSS = {20: 41, 21: 42, 22: 43, 23: 44, 24: 45, 25: 45}
# TSS L3 label (for B-angle)
_TSS_L3 = 43
# TSS sacrum
_TSS_SACRUM = 50


def compute_vertebral_angles(
    sp_iso:           np.ndarray,
    vert_iso:         np.ndarray,
    tss_iso:          Optional[np.ndarray],
    tv_veridah_label: int,
    vox_mm:           float = 1.0,
    sp_corpus_label:  int   = 49,
    disc_above_dhi:   Optional[float] = None,
    disc_below_dhi:   Optional[float] = None,
) -> VertebralAnglesResult:
    """
    Wrapper called by lstv_engine.py step 8.5.

    Uses tss_iso (preferred) or vert_iso as the label volume.
    tv_veridah_label is converted to TSS label for tss_iso lookups.
    """
    result = VertebralAnglesResult()

    # Pick the volume to operate on
    if tss_iso is not None and tss_iso.any():
        vol      = tss_iso
        tv_label = _VD_TO_TSS.get(tv_veridah_label, 45)  # default to L5
        # tv-1 = one cranial: tv_label + 1 in TSS (L5=45 → L4=44, etc.)
        tv_minus1_label = tv_label - 1  # TSS decreases cranially: L5=45, L4=44
        l3_label = _TSS_L3 if _TSS_L3 in np.unique(vol) else None
    else:
        result.computation_notes.append("TSS not available — falling back to vert_iso")
        vol             = vert_iso
        tv_label        = tv_veridah_label
        tv_minus1_label = tv_veridah_label - 1  # VERIDAH also decreases cranially
        l3_label        = None  # VERIDAH L3 = label 22

    # Check minimum required labels are present
    labels_present = set(np.unique(vol).tolist()) - {0}
    missing = []
    for lbl, name in [(tv_label, 'TV'), (tv_minus1_label, 'TV-1'), (_TSS_SACRUM, 'sacrum')]:
        if lbl not in labels_present:
            missing.append(f"{name}(lbl={lbl})")
    if missing:
        result.computation_notes.append(f"Missing required labels: {missing}")
        logger.warning("compute_vertebral_angles: missing labels %s — skipping", missing)
        return result

    # Build sp_corpus_volume if available
    sp_corpus_vol = None
    if sp_corpus_label and sp_corpus_label in np.unique(sp_iso):
        sp_corpus_vol = (sp_iso == sp_corpus_label).astype(np.int32)

    spacing = (vox_mm, vox_mm, vox_mm)

    try:
        angles: LSTVAngles = compute_angles(
            vol,
            tv_label        = tv_label,
            tv_minus1_label = tv_minus1_label,
            sacrum_label    = _TSS_SACRUM,
            l3_label        = l3_label,
            sp_corpus_volume = sp_corpus_vol,
            voxel_spacing_mm = spacing,
        )
    except Exception as exc:
        result.computation_notes.append(f"compute_angles raised: {exc}")
        logger.error("compute_vertebral_angles inner call failed: %s", exc)
        return result

    # Populate VertebralAnglesResult from LSTVAngles
    result.a_angle_deg     = angles.A
    result.b_angle_deg     = angles.B
    result.c_angle_deg     = angles.C
    result.d_angle_deg     = angles.D
    result.d1_angle_deg    = angles.D1
    result.delta_angle_deg = angles.delta
    result.angle_flags     = angles.flags
    result.angles_available = any(
        v is not None for v in (angles.A, angles.D, angles.delta)
    )

    # Classification flags
    if angles.delta is not None:
        result.delta_positive = angles.delta <= 8.5   # ≤8.5°
        result.delta_any_lstv = angles.delta <= 15.0
    if angles.C is not None:
        result.c_positive = angles.C <= 35.5                # ≤35.5°
    if angles.A is not None:
        result.a_angle_elevated = angles.A > 41.0            # >41°

    logger.debug(
        "VertebralAngles: A=%.1f B=%s C=%s D=%.1f D1=%.1f delta=%.1f "
        "(delta_pos=%s c_pos=%s a_elev=%s)",
        angles.A or 0,
        f"{angles.B:.1f}" if angles.B else "–",
        f"{angles.C:.1f}" if angles.C else "–",
        angles.D or 0, angles.D1 or 0, angles.delta or 0,
        result.delta_positive, result.c_positive, result.a_angle_elevated,
    )
    return result


def apply_angle_lr_updates(
    lo_sac:    float,
    lo_lumb:   float,
    angle_result: VertebralAnglesResult,
    existing_criteria: list,
) -> Tuple[float, float, list]:
    """
    Apply Bayesian log-odds updates from angle findings and return
    (updated_lo_sac, updated_lo_lumb, new_RadiologicCriterion_list).

    LR values from Seilanian Toosi 2025 (Table 3 / text):
      delta ≤8.5°  → sens 92.3%, spec 87.9%  →  LR+ ≈ 7.7
      C ≤35.5°     → LSTV signal              →  LR+ ≈ 4.5
      A >41°       → sacralization shift       →  LR+ ≈ 2.0
    """
    new_criteria: list = []
    keys_fired:   list = []

    if angle_result.delta_positive:
        delta_update = float(np.log(7.7))
        lo_sac      += delta_update
        keys_fired.append('delta_le_8.5')
        new_criteria.append(RadiologicCriterion(
            name='delta_angle',
            value=f'{angle_result.delta_angle_deg:.1f}°',
            direction='sacralization',
            strength='primary',
            lr_sac=round(delta_update, 3),
            lr_lumb=0.0,
            citation='Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280',
            finding=(
                f'delta-angle {angle_result.delta_angle_deg:.1f}° ≤ 8.5° — '
                f'Type 2 LSTV (sens 92.3%, spec 87.9%, NPV 99.5%)'),
        ))
    elif angle_result.delta_any_lstv:
        delta_update = float(np.log(2.5))
        lo_sac      += delta_update
        keys_fired.append('delta_le_15')
        new_criteria.append(RadiologicCriterion(
            name='delta_angle_borderline',
            value=f'{angle_result.delta_angle_deg:.1f}°',
            direction='sacralization',
            strength='supporting',
            lr_sac=round(delta_update, 3),
            lr_lumb=0.0,
            citation='Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280',
            finding=f'delta-angle {angle_result.delta_angle_deg:.1f}° ≤ 15° — borderline LSTV signal',
        ))

    if angle_result.c_positive:
        c_update = float(np.log(4.5))
        lo_sac  += c_update
        keys_fired.append('c_le_35.5')
        new_criteria.append(RadiologicCriterion(
            name='c_angle',
            value=f'{angle_result.c_angle_deg:.1f}°',
            direction='sacralization',
            strength='secondary',
            lr_sac=round(c_update, 3),
            lr_lumb=0.0,
            citation='Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280',
            finding=f'C-angle {angle_result.c_angle_deg:.1f}° ≤ 35.5° — LSTV posterior alignment signal',
        ))

    if angle_result.a_angle_elevated:
        a_update = float(np.log(2.0))
        lo_sac  += a_update
        keys_fired.append('a_gt_41')
        new_criteria.append(RadiologicCriterion(
            name='a_angle_elevated',
            value=f'{angle_result.a_angle_deg:.1f}°',
            direction='sacralization',
            strength='supporting',
            lr_sac=round(a_update, 3),
            lr_lumb=0.0,
            citation='Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280',
            finding=f'A-angle {angle_result.a_angle_deg:.1f}° > 41° — elevated sacral tilt',
        ))

    angle_result.angle_lr_keys_fired = keys_fired
    return lo_sac, lo_lumb, new_criteria


# ============================================================
# 04_detect_lstv.py v5.2 interface: tp_in_correct_zone
# ============================================================

def tp_in_correct_zone(
    tp_mask:     np.ndarray,
    tss_vol:     np.ndarray,
    cc_axis:     int  = 2,
    si_positive: bool = True,
) -> Tuple[bool, str]:
    """
    Check whether a TP mask's craniocaudal centroid lies within the valid
    L5 transverse-process zone:

        lower bound: cranial (superior) face of L5-S1 disc (TSS label 100)
                     fallback: cranial face of sacrum        (TSS label 50)
        upper bound: caudal  (inferior) face of L4-L5 disc  (TSS label 95)
                     fallback: cranial  face of TSS L5        (label 45)

    In standard canonical (RAS) with si_positive=True: larger CC index = cranial.
    So:
        lower_bound = max CC of L5-S1 disc  (its cranial face)
        upper_bound = min CC of L4-L5 disc  (its caudal  face)

    Returns
    -------
    (ok: bool, message: str)
    """
    if not tp_mask.any():
        return True, "TP mask empty — skipping zone check"

    centroid_cc = float(np.mean(np.where(tp_mask)[cc_axis]))

    # ── upper bound: caudal (inferior) face of L4-L5 disc ──────────────────
    disc_l4l5 = (tss_vol == 95)
    tss_l5    = (tss_vol == 45)
    if disc_l4l5.any():
        cc_vals = np.where(disc_l4l5)[cc_axis]
        upper   = float(cc_vals.min() if si_positive else cc_vals.max())
        upper_src = "L4-L5 disc (TSS 95) caudal face"
    elif tss_l5.any():
        cc_vals = np.where(tss_l5)[cc_axis]
        upper   = float(cc_vals.max() if si_positive else cc_vals.min())
        upper_src = "TSS L5 (label 45) cranial face [fallback]"
    else:
        upper     = None
        upper_src = "unavailable"

    # ── lower bound: cranial (superior) face of L5-S1 disc ─────────────────
    disc_l5s1 = (tss_vol == 100)
    tss_sac   = (tss_vol == 50)
    if disc_l5s1.any():
        cc_vals = np.where(disc_l5s1)[cc_axis]
        lower   = float(cc_vals.max() if si_positive else cc_vals.min())
        lower_src = "L5-S1 disc (TSS 100) cranial face"
    elif tss_sac.any():
        cc_vals = np.where(tss_sac)[cc_axis]
        lower   = float(cc_vals.max() if si_positive else cc_vals.min())
        lower_src = "sacrum (TSS 50) cranial face [fallback]"
    else:
        lower     = None
        lower_src = "unavailable"

    # ── check ───────────────────────────────────────────────────────────────
    if lower is None and upper is None:
        return True, "bounds unavailable — cannot validate zone"

    if si_positive:
        below_disc = (lower is not None and centroid_cc < lower)
        above_l4l5 = (upper is not None and centroid_cc > upper)
    else:
        below_disc = (lower is not None and centroid_cc > lower)
        above_l4l5 = (upper is not None and centroid_cc < upper)

    if below_disc:
        return False, (
            f"TP centroid CC={centroid_cc:.1f} BELOW lower bound={lower:.1f} "
            f"({lower_src}) — TP displaced into sacrum"
        )
    if above_l4l5:
        return False, (
            f"TP centroid CC={centroid_cc:.1f} ABOVE upper bound={upper:.1f} "
            f"({upper_src}) — TP belongs to L4, not L5"
        )

    return True, (
        f"TP centroid CC={centroid_cc:.1f} in zone "
        f"[{lower if lower is not None else '?':.1f}, "
        f"{upper if upper is not None else '?':.1f}] "
        f"(lower={lower_src}, upper={upper_src})"
    )
