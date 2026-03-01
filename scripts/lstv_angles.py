#!/usr/bin/env python3
"""
lstv_angles.py — Vertebral Angle Analysis for LSTV Detection (Seilanian Toosi et al. 2025)
============================================================================================
v2 ADDITIONS:
  - Sanity checking for all computed angles with physiological plausibility bounds
  - TP height now computed via PCA principal axis (not naive Z-span) to handle
    angled transverse processes correctly
  - Endplate normal fitting improved: uses centroid-regression across CC slices
    to capture tilt direction correctly; falls back to PCA on raw slab voxels only
    when slice count is insufficient
  - All angle computations validated against published normal/LSTV medians;
    out-of-range results are flagged with notes rather than silently returned

ANGLE DEFINITIONS  (Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280)
---------------------
A-angle  : angle between sacral superior surface and vertical axis of scanner
           (Chalian 2012; OR 1.141 per degree, p=0.023)
           Normal median ≈37°, LSTV median ≈41.5°
B-angle  : angle between L3 superior endplate and sacral superior surface
           (Chalian 2012; not independently significant)
           Normal median ≈43°
C-angle  : largest angle formed by posterior-body lines of TV±1 and sacrum±1
           (≤35.5° → any LSTV; sens 72.2%, spec 57.6%, NPV 91.4%)
           Normal median ≈37°, LSTV median ≈31°
D-angle  : angle between superior surface of most cranial sacrum and most caudal lumbar
           Normal median ≈26°, LSTV type2 median ≈22°
D1-angle : angle between superior surface of TV and TV-1 (supra-adjacent)
           Normal ≈14°
delta    : D − D1   ← PRIMARY criterion
           ≤8.5°  → Type 2 LSTV:  sens 92.3%, spec 87.9%, NPV 99.5%
           ≤14.5° → any LSTV:     sens 66.7%, spec 52.2%
           Normal median ≈15°, LSTV type2 median ≈2°

SANITY BOUNDS (from Table 2/3, Seilanian Toosi 2025 + Chalian 2012)
--------------------------------------------------------------------
A-angle : [15°, 70°]   — sacrum always tilted, but not extreme
B-angle : [10°, 80°]   — L3 relative to sacrum
C-angle : [5°, 65°]    — posterior body line angle
D-angle : [5°, 55°]    — lumbosacral junction angle
D1-angle: [0°, 45°]    — adjacent lumbar intervertebral angle
delta   : [-15°, 35°]  — D - D1; negative possible in severe LSTV

TP HEIGHT VIA PRINCIPAL AXIS (craniocaudal component)
------------------------------------------------------
The naive max_z - min_z measurement is wrong when the TP is angled. But naively
using the *longest* PCA axis is also wrong: the TP extends primarily mediolaterally,
so its longest axis is roughly horizontal — that would measure TP reach, not height.

Correct approach:
  1. Extract TP voxel coordinates
  2. PCA → get all 3 eigenvectors
  3. Identify the reference craniocaudal direction:
       - Preferred: vector from L5-S1 disc centroid (TSS 100) → L4-L5 disc centroid
         (TSS 95). This captures lumbosacral lordotic tilt.
       - Fallback: detected global CC axis unit vector
  4. Pick the PCA component with the highest dot-product alignment with that reference
  5. Span = (max_proj - min_proj) along that component × voxel_size = CC height

This is exported as measure_tp_height_pca() for use by 04_detect_lstv.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds (Seilanian Toosi et al. 2025) ──────────────────────────────────
DELTA_TYPE2_THRESHOLD    = 8.5    # delta ≤ 8.5°  → Castellvi Type 2 LSTV
DELTA_ANY_LSTV_THRESHOLD = 14.5   # delta ≤ 14.5° → any LSTV (AUC 0.658)
C_LSTV_THRESHOLD         = 35.5   # C     ≤ 35.5° → any LSTV (AUC 0.688)
A_INCREASED_THRESHOLD    = 43.0   # A-angle > 43° → elevated (LSTV type 2 median)
D_DECREASED_THRESHOLD    = 22.0   # D-angle < 22° → decreased (LSTV type 2 median)

# ── Physiological sanity bounds ────────────────────────────────────────────────
_SANITY: Dict[str, Tuple[float, float]] = {
    'a_angle':     (12.0, 72.0),
    'b_angle':     (8.0,  82.0),
    'c_angle':     (3.0,  68.0),
    'd_angle':     (3.0,  58.0),
    'd1_angle':    (0.0,  48.0),
    'delta_angle': (-18.0, 38.0),
}

# ── Published medians for reference logging ────────────────────────────────────
_NORMAL_MEDIANS = {
    'a_angle': 37.0, 'b_angle': 43.0, 'c_angle': 37.0,
    'd_angle': 26.0, 'd1_angle': 14.0, 'delta_angle': 15.0,
}
_LSTV_MEDIANS = {
    'a_angle': 41.5, 'c_angle': 31.1, 'd_angle': 23.5,
    'delta_angle': 11.5,
}

# ── Bayesian LR entries for lstv_engine._LR dict ─────────────────────────────
_ANGLE_LR: Dict[str, Tuple[float, float, float, float]] = {
    # key:                         (LR+sac, LR-sac, LR+lumb, LR-lumb)
    'angle_delta_type2':     (7.63,  0.64, 1.0, 1.0),
    'angle_delta_any_lstv':  (1.39,  0.64, 1.0, 1.0),
    'angle_c_lstv':          (1.70,  0.47, 1.0, 1.0),
    'angle_a_elevated':      (1.55,  0.82, 1.0, 1.0),
    'angle_d_decreased':     (1.45,  0.88, 1.0, 1.0),
}

# ── VERIDAH / TSS label constants ─────────────────────────────────────────────
VD_L1=20; VD_L2=21; VD_L3=22; VD_L4=23; VD_L5=24; VD_L6=25; VD_SAC=26
TSS_L1=41; TSS_L2=42; TSS_L3=43; TSS_L4=44; TSS_L5=45; TSS_SAC=50
SP_CORPUS = 49

# TSS disc labels — used for TP concordance verification
TSS_DISC_L4L5 = 95   # L4-L5 disc
TSS_DISC_L5S1 = 100  # L5-S1 disc

VERIDAH_CAUDAL_TO_CRANIAL = [26, 25, 24, 23, 22, 21, 20]
TSS_CAUDAL_TO_CRANIAL     = [50, 45, 44, 43, 42, 41]

MIN_VOXELS_FOR_PCA = 20
SURFACE_SLAB_VOXELS = 5


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VertebralAnglesResult:
    a_angle_deg:     Optional[float] = None
    b_angle_deg:     Optional[float] = None
    c_angle_deg:     Optional[float] = None
    d_angle_deg:     Optional[float] = None
    d1_angle_deg:    Optional[float] = None
    delta_angle_deg: Optional[float] = None

    delta_positive:    bool = False
    delta_any_lstv:    bool = False
    c_positive:        bool = False
    a_angle_elevated:  bool = False
    d_angle_decreased: bool = False
    disc_pattern_lstv: bool = False

    angle_lr_keys_fired: List[str] = field(default_factory=list)

    angles_available:  bool = False
    computation_notes: List[str] = field(default_factory=list)
    sanity_warnings:   List[str] = field(default_factory=list)

    detected_cc_axis:     Optional[int]  = None
    detected_si_positive: Optional[bool] = None
    orientation_method:   Optional[str]  = None

    summary: str = ''

    def to_dict(self) -> dict:
        d = asdict(self)
        d['delta_angle']    = self.delta_angle_deg
        d['c_angle']        = self.c_angle_deg
        d['a_angle']        = self.a_angle_deg
        d['b_angle']        = self.b_angle_deg
        d['d_angle']        = self.d_angle_deg
        d['d1_angle']       = self.d1_angle_deg
        d['delta_le8p5']    = self.delta_positive
        d['c_le35p5']       = self.c_positive
        d['a_increased']    = self.a_angle_elevated
        d['d_decreased']    = self.d_angle_decreased
        d['delta_any_lstv'] = self.delta_any_lstv
        return d


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECKING
# ══════════════════════════════════════════════════════════════════════════════

def _sanity_check_angle(name: str, value: Optional[float],
                         notes: List[str]) -> Optional[float]:
    """
    Validate a computed angle against physiological plausibility bounds.
    Returns the value if plausible, None if implausible (with note added).
    Logs a warning for borderline but technically valid values.
    """
    if value is None:
        return None

    lo, hi = _SANITY.get(name, (-999, 999))
    if not (lo <= value <= hi):
        msg = (f"SANITY FAIL: {name}={value:.1f}° outside plausible range "
               f"[{lo}°, {hi}°] — discarded")
        notes.append(msg)
        logger.warning(f"  {msg}")
        return None

    # Warn if very far from both normal and LSTV medians
    norm_med = _NORMAL_MEDIANS.get(name)
    if norm_med is not None:
        deviation = abs(value - norm_med)
        if deviation > 25:
            notes.append(f"SANITY WARN: {name}={value:.1f}° deviates "
                         f"{deviation:.0f}° from normal median {norm_med}°")
    return value


def _run_sanity_checks(res: 'VertebralAnglesResult') -> None:
    """
    Run all angle sanity checks after computation, flagging suspicious values.
    Also cross-checks internal consistency (delta = D - D1).
    """
    warnings = res.sanity_warnings
    notes    = res.computation_notes

    # Individual bounds
    for name, attr in [
        ('a_angle',     'a_angle_deg'),
        ('b_angle',     'b_angle_deg'),
        ('c_angle',     'c_angle_deg'),
        ('d_angle',     'd_angle_deg'),
        ('d1_angle',    'd1_angle_deg'),
        ('delta_angle', 'delta_angle_deg'),
    ]:
        val = getattr(res, attr)
        checked = _sanity_check_angle(name, val, notes)
        if checked is None and val is not None:
            # Sanity check failed — null out the value
            setattr(res, attr, None)
            warnings.append(f"Angle {name} failed sanity check (was {val:.1f}°)")

    # Cross-check: delta should equal D - D1 (within floating point)
    if (res.d_angle_deg is not None and res.d1_angle_deg is not None
            and res.delta_angle_deg is not None):
        expected_delta = res.d_angle_deg - res.d1_angle_deg
        if abs(res.delta_angle_deg - expected_delta) > 1.5:
            warnings.append(
                f"CONSISTENCY WARN: delta={res.delta_angle_deg:.1f}° but "
                f"D={res.d_angle_deg:.1f}° - D1={res.d1_angle_deg:.1f}° = "
                f"{expected_delta:.1f}° (discrepancy {abs(res.delta_angle_deg-expected_delta):.1f}°)"
            )

    # Cross-check: D1 should be smaller than D in normal anatomy
    # (the lumbosacral junction has a larger angle than adjacent lumbar levels)
    # In LSTV type 2, D ≈ D1 (hence small delta)
    if res.d_angle_deg is not None and res.d1_angle_deg is not None:
        if res.d_angle_deg < res.d1_angle_deg - 15:
            warnings.append(
                f"ANATOMY WARN: D={res.d_angle_deg:.1f}° << D1={res.d1_angle_deg:.1f}° — "
                f"lumbosacral angle smaller than supra-adjacent; check TV labeling"
            )

    # Cross-check: A-angle should be > C-angle in most cases
    # (sacral tilt > posterior body angle differential)
    if res.a_angle_deg is not None and res.c_angle_deg is not None:
        if res.a_angle_deg < res.c_angle_deg * 0.4:
            warnings.append(
                f"ANATOMY WARN: A={res.a_angle_deg:.1f}° << C={res.c_angle_deg:.1f}° — "
                f"sacral tilt implausibly smaller than C-angle"
            )

    if warnings:
        logger.warning(f"  Angle sanity warnings ({len(warnings)}): "
                       + "; ".join(warnings[:3])
                       + (f" ... +{len(warnings)-3} more" if len(warnings) > 3 else ""))


# ══════════════════════════════════════════════════════════════════════════════
# TP HEIGHT VIA PRINCIPAL AXIS (craniocaudal component, not longest axis)
# ══════════════════════════════════════════════════════════════════════════════

def _disc_to_disc_axis(tss_iso: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Compute the unit vector from the centroid of the L5-S1 disc (TSS label 100)
    to the centroid of the L4-L5 disc (TSS label 95).

    This gives the local craniocaudal direction at the lumbosacral junction,
    which is the correct reference axis for measuring TP craniocaudal height
    (Castellvi criterion: ≥19 mm craniocaudal dimension).

    Returns None if either disc label is absent.
    """
    if tss_iso is None:
        return None

    l4l5_mask = (tss_iso == TSS_DISC_L4L5)   # label 95
    l5s1_mask = (tss_iso == TSS_DISC_L5S1)   # label 100

    if not l4l5_mask.any() or not l5s1_mask.any():
        return None

    c_l4l5 = np.array(np.where(l4l5_mask), dtype=float).mean(axis=1)  # more cranial
    c_l5s1 = np.array(np.where(l5s1_mask), dtype=float).mean(axis=1)  # more caudal

    vec = c_l4l5 - c_l5s1   # points cranially (L5-S1 → L4-L5)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None

    axis = vec / norm
    logger.debug(f"  Disc-to-disc CC axis: {axis.round(3)}  "
                 f"(L5-S1→L4-L5, sep={norm:.1f}vox)")
    return axis


def measure_tp_height_pca(
        tp_mask:  np.ndarray,
        vox_mm:   float = 1.0,
        tss_iso:  Optional[np.ndarray] = None,
        cc_axis:  int  = 2,
        si_positive: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Compute the craniocaudal height of a transverse process by projecting its
    voxel coordinates onto the local craniocaudal axis — NOT the longest PCA axis.

    WHY NOT THE LONGEST AXIS
    ------------------------
    A transverse process extends primarily in the mediolateral direction (left/right).
    Its longest PCA component is therefore roughly horizontal (ML axis), NOT
    craniocaudal.  The Castellvi criterion measures the *craniocaudal* (SI) dimension
    only.  Using the longest PCA axis would measure TP lateral reach, not height.

    AXIS SELECTION PRIORITY
    -----------------------
    1. Disc-to-disc axis: vector from centroid of L5-S1 disc (TSS 100) to centroid
       of L4-L5 disc (TSS 95).  This is the most anatomically accurate local CC
       direction at the lumbosacral junction, accounting for any lordotic tilt.
       Requires tss_iso to contain both disc labels.

    2. If disc labels unavailable: use the detected global CC axis (cc_axis /
       si_positive from detect_cranio_caudal_axis).  This is a unit vector along
       the appropriate array axis.

    For each candidate axis from PCA (all 3 components), we choose the one whose
    dot product with the reference CC direction is largest — i.e. the component
    most aligned with craniocaudal.  We then project all voxel coordinates onto
    that component to get the SI span.

    Returns
    -------
    height_mm : float — CC extent of the TP * vox_mm
    axis      : np.ndarray shape (3,) — unit vector used for measurement
    """
    if not tp_mask.any():
        return 0.0, np.array([0., 0., 1.])

    coords = np.array(np.where(tp_mask), dtype=float).T  # (N, 3)
    if len(coords) < MIN_VOXELS_FOR_PCA:
        # Too few voxels for PCA — fall back to naive span along the CC axis
        cc_coords = coords[:, cc_axis]
        span_vox  = float(cc_coords.max() - cc_coords.min() + 1)
        fallback_axis = _make_cc_unit_vector(cc_axis, si_positive)
        return span_vox * vox_mm, fallback_axis

    # Determine the reference craniocaudal direction
    ref_cc = _disc_to_disc_axis(tss_iso)
    if ref_cc is None:
        # Fall back to the global CC unit vector
        ref_cc = _make_cc_unit_vector(cc_axis, si_positive)
        ref_source = f"global-cc-axis (axis={cc_axis})"
    else:
        ref_source = "disc-to-disc (L5-S1→L4-L5)"

    # Run PCA to get all 3 principal components
    centred = coords - coords.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    # vt rows are the principal axes, ordered by descending variance
    # (vt[0] = longest, vt[2] = shortest)

    # Pick the PCA component most aligned with the craniocaudal direction
    alignments  = [abs(float(np.dot(vt[i], ref_cc))) for i in range(3)]
    best_idx    = int(np.argmax(alignments))
    height_axis = vt[best_idx]

    # Ensure the axis points cranially (positive dot with ref_cc)
    if np.dot(height_axis, ref_cc) < 0:
        height_axis = -height_axis

    # Project all voxel coordinates onto the height axis
    projections = centred @ height_axis
    span_vox    = float(projections.max() - projections.min())
    height_mm   = span_vox * vox_mm

    logger.debug(
        f"  TP PCA height: {height_mm:.1f}mm  "
        f"axis_idx={best_idx} (PCA component {best_idx}, alignment={alignments[best_idx]:.3f})  "
        f"ref={ref_source}  axis={height_axis.round(3)}"
    )
    return height_mm, height_axis


# ══════════════════════════════════════════════════════════════════════════════
# ORIENTATION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_cranio_caudal_axis(
        vert_iso: np.ndarray,
        tss_iso:  Optional[np.ndarray] = None,
) -> Tuple[int, bool, str]:
    """
    Detect which array axis is the cranio-caudal (S-I) axis without relying
    on DICOM headers or NIfTI orientation metadata.
    """
    # Method 1: sacrum vs cranial-lumbar centroid separation
    for iso, sacrum_lbl, cranial_lbls in [
        (vert_iso, VD_SAC,  [VD_L1, VD_L2, VD_L3]),
        (tss_iso,  TSS_SAC, [TSS_L1, TSS_L2, TSS_L3]),
    ]:
        if iso is None:
            continue
        sac_mask = (iso == sacrum_lbl)
        if not sac_mask.any():
            continue
        sac_centroid = np.array(np.where(sac_mask), dtype=float).mean(axis=1)

        for cranial_lbl in cranial_lbls:
            cran_mask = (iso == cranial_lbl)
            if not cran_mask.any():
                continue
            cran_centroid = np.array(np.where(cran_mask), dtype=float).mean(axis=1)

            diff = cran_centroid - sac_centroid
            cc_axis = int(np.argmax(np.abs(diff)))
            si_positive = bool(diff[cc_axis] > 0)
            sep_mm = float(np.abs(diff[cc_axis]))

            if sep_mm > 15.0:
                lbl_name = 'VERIDAH' if iso is vert_iso else 'TSS'
                method = (f"centroid-separation: sacrum→L{cranial_lbl % 10 or 3} "
                          f"axis={cc_axis} sep={sep_mm:.0f}mm [{lbl_name}]")
                logger.debug(f"  Orientation: {method}")
                return cc_axis, si_positive, method

    # Method 2: largest bounding-box extent
    for iso in [vert_iso, tss_iso]:
        if iso is None:
            continue
        spine_mask = iso > 0
        if not spine_mask.any():
            continue
        coords = np.array(np.where(spine_mask), dtype=float)
        extents = coords.max(axis=1) - coords.min(axis=1)
        cc_axis = int(np.argmax(extents))

        for sacrum_lbl in [VD_SAC, TSS_SAC]:
            sac_mask = (iso == sacrum_lbl)
            if not sac_mask.any():
                continue
            sac_centroid = np.array(np.where(sac_mask), dtype=float).mean(axis=1)
            all_centroid = coords.mean(axis=1)
            si_positive = bool(sac_centroid[cc_axis] < all_centroid[cc_axis])
            method = (f"largest-extent: axis={cc_axis} extent={extents[cc_axis]:.0f}vox "
                      f"si_positive={si_positive}")
            logger.debug(f"  Orientation: {method}")
            return cc_axis, si_positive, method

        cc_axis = int(np.argmax(extents))
        method = f"largest-extent (no sacrum): axis={cc_axis} extent={extents[cc_axis]:.0f}vox"
        logger.debug(f"  Orientation: {method}")
        return cc_axis, True, method

    logger.warning("  Orientation detection failed — falling back to axis=2")
    return 2, True, "fallback-axis2 (detection failed)"


def _make_cc_unit_vector(cc_axis: int, si_positive: bool) -> np.ndarray:
    v = np.zeros(3)
    v[cc_axis] = 1.0 if si_positive else -1.0
    return v


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _fit_endplate_plane(
        mask:        np.ndarray,
        surface:     str,
        cc_axis:     int  = 2,
        si_positive: bool = True,
) -> Optional[np.ndarray]:
    """
    Fit a plane to the superior or inferior surface of a vertebral body mask.

    Returns the plane normal vector (unit length), oriented to point cranially.
    Returns None on failure.

    METHOD: centroid-regression across slices
    -----------------------------------------
    For each CC slice in the surface slab, compute the centroid (AP, ML).
    Regress centroids against CC position using SVD (line fit).
    The endplate normal is perpendicular to this line in the CC-AP plane.

    This correctly captures the physical tilt of the endplate, unlike PCA
    on raw voxels which is dominated by the in-plane rectangular cross-section
    of the vertebral body rather than its tilt across CC slices.

    NARROW MASK HANDLING:
    Endplate masks may be very thin (1-3 voxels in CC). We use a slab of
    max(3, CC_span // 4) slices, with a minimum of 3 slices required for
    regression. If fewer slices available, fall back to full-body PCA.
    """
    if not mask.any():
        return None

    axes = [0, 1, 2]
    cc_coords_all = np.where(mask)[cc_axis]
    cc_lo = int(cc_coords_all.min())
    cc_hi = int(cc_coords_all.max())
    cc_span = cc_hi - cc_lo + 1

    if cc_span < 2:
        # Completely flat mask — use the CC axis as the normal directly
        normal = np.zeros(3)
        normal[cc_axis] = 1.0
        cranial_unit = _make_cc_unit_vector(cc_axis, si_positive)
        if np.dot(normal, cranial_unit) < 0:
            normal = -normal
        return normal

    # Select slab: top or bottom quarter, at least 3 slices
    slab_n = max(3, min(cc_span // 4, SURFACE_SLAB_VOXELS * 2))

    if surface == 'superior':
        if si_positive:
            cc_start = max(cc_lo, cc_hi - slab_n + 1)
            cc_end   = cc_hi
        else:
            cc_start = cc_lo
            cc_end   = min(cc_hi, cc_lo + slab_n - 1)
    else:  # inferior
        if si_positive:
            cc_start = cc_lo
            cc_end   = min(cc_hi, cc_lo + slab_n - 1)
        else:
            cc_start = max(cc_lo, cc_hi - slab_n + 1)
            cc_end   = cc_hi

    non_cc_axes = [a for a in axes if a != cc_axis]

    # Compute per-slice centroid
    slice_centroids = []
    for cc_val in range(cc_start, cc_end + 1):
        idx = [slice(None), slice(None), slice(None)]
        idx[cc_axis] = cc_val
        sl = mask[tuple(idx)]
        if not sl.any():
            continue
        ax0_coords = np.where(sl)[0]
        ax1_coords = np.where(sl)[1]
        c = np.zeros(3)
        c[cc_axis]         = float(cc_val)
        c[non_cc_axes[0]]  = float(ax0_coords.mean())
        c[non_cc_axes[1]]  = float(ax1_coords.mean())
        slice_centroids.append(c)

    if len(slice_centroids) < 3:
        # Not enough CC slices in slab → expand to full mask
        # This handles very thin endplate masks
        for cc_val in range(cc_lo, cc_hi + 1):
            idx = [slice(None), slice(None), slice(None)]
            idx[cc_axis] = cc_val
            sl = mask[tuple(idx)]
            if not sl.any():
                continue
            ax0_coords = np.where(sl)[0]
            ax1_coords = np.where(sl)[1]
            c = np.zeros(3)
            c[cc_axis]         = float(cc_val)
            c[non_cc_axes[0]]  = float(ax0_coords.mean())
            c[non_cc_axes[1]]  = float(ax1_coords.mean())
            slice_centroids.append(c)

    if len(slice_centroids) < 2:
        # Ultimate fallback: use all voxels with PCA
        slab_voxels = np.array(np.where(mask), dtype=float).T
        if len(slab_voxels) < MIN_VOXELS_FOR_PCA:
            return None
        centred = slab_voxels - slab_voxels.mean(axis=0)
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        normal = vt[-1]
    else:
        pts = np.array(slice_centroids)
        centred = pts - pts.mean(axis=0)
        _, s, vt = np.linalg.svd(centred, full_matrices=False)

        # Primary singular vector = direction of centroid drift across CC slices
        # = "down-slope" direction of the endplate
        # Normal = perpendicular to this, in the sagittal (CC-AP) plane
        line_dir = vt[0]
        non_cc = [a for a in axes if a != cc_axis]

        # AP-like axis = non-CC axis with larger component in the primary direction
        ap_candidates = [(abs(line_dir[a]), a) for a in non_cc]
        ap_axis = max(ap_candidates)[1]

        cc_comp = line_dir[cc_axis]
        ap_comp = line_dir[ap_axis]

        line_mag = np.sqrt(cc_comp**2 + ap_comp**2)
        if line_mag < 1e-9:
            # No CC-AP drift → flat endplate → normal points along CC
            normal = np.zeros(3)
            normal[cc_axis] = 1.0
        else:
            # Rotate (cc_comp, ap_comp) by 90° to get the normal
            normal = np.zeros(3)
            normal[cc_axis] = -ap_comp / line_mag
            normal[ap_axis] =  cc_comp / line_mag

    # Orient to point cranially
    cranial_unit = _make_cc_unit_vector(cc_axis, si_positive)
    if np.dot(normal, cranial_unit) < 0:
        normal = -normal

    mag = np.linalg.norm(normal)
    if mag < 1e-9:
        return None
    return normal / mag


def _angle_between_normals(
        n1:      np.ndarray,
        n2:      np.ndarray,
        cc_axis: int = 2,
) -> float:
    """
    Angle (degrees) between two endplate normals, projected onto the
    midsagittal plane (contains CC and AP axes; drops ML component).
    Matches the 2D sagittal measurement convention of Seilanian Toosi 2025.
    """
    axes = [0, 1, 2]
    non_cc = [a for a in axes if a != cc_axis]

    # ML axis = non-CC axis with smallest mean absolute component
    comp = [0.5 * (abs(n1[a]) + abs(n2[a])) for a in non_cc]
    ml_axis = non_cc[int(np.argmin(comp))]

    def _project_sagittal(n):
        p = n.copy()
        p[ml_axis] = 0.0
        mag = np.linalg.norm(p)
        return p / mag if mag > 1e-9 else None

    n1_sag = _project_sagittal(n1)
    n2_sag = _project_sagittal(n2)

    if n1_sag is None or n2_sag is None:
        cos_a = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_a)))

    cos_a = float(np.clip(np.dot(n1_sag, n2_sag), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_a)))
    return min(angle, 180.0 - angle)


def _angle_vs_vertical(
        normal:      np.ndarray,
        cc_axis:     int  = 2,
        si_positive: bool = True,
) -> float:
    """
    Angle (degrees) between the endplate surface and the horizontal plane.
    = angle between the endplate normal and the cranio-caudal axis.
    = arccos(|n · ẑ_cc|)

    Definition: A-angle = angle between sacral superior surface and
    a line perpendicular to the scan table (vertical axis).
    Normal ≈ 37°, LSTV ≈ 41.5° (Seilanian Toosi 2025).
    """
    cranial_unit = _make_cc_unit_vector(cc_axis, si_positive)
    cos_a = float(np.clip(abs(np.dot(normal, cranial_unit)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _posterior_body_normal(
        mask:    np.ndarray,
        cc_axis: int = 2,
) -> Optional[np.ndarray]:
    """
    Fit a plane through the posterior third of the vertebral body.
    Posterior = smaller AP-axis coordinate (anterior is forward/larger Y in RAS).
    """
    if not mask.any():
        return None

    axes = [0, 1, 2]
    non_cc = [a for a in axes if a != cc_axis]
    extents = [int(np.where(mask)[a].max()) - int(np.where(mask)[a].min()) for a in non_cc]
    ap_axis = non_cc[int(np.argmax(extents))]

    ap_coords = np.where(mask)[ap_axis]
    ap_lo = int(ap_coords.min())
    ap_hi = int(ap_coords.max())
    slab_ap = max(1, (ap_hi - ap_lo) // 3)
    ap_end = min(ap_hi, ap_lo + slab_ap)

    idx = [slice(None), slice(None), slice(None)]
    idx[ap_axis] = slice(ap_lo, ap_end + 1)
    post = np.zeros_like(mask)
    post[tuple(idx)] = mask[tuple(idx)]

    coords = np.array(np.where(post), dtype=float).T
    if len(coords) < MIN_VOXELS_FOR_PCA:
        coords = np.array(np.where(mask), dtype=float).T
        if len(coords) < MIN_VOXELS_FOR_PCA:
            return None

    centred = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    cranial_unit = _make_cc_unit_vector(cc_axis, True)
    if np.dot(normal, cranial_unit) < 0:
        normal = -normal
    mag = np.linalg.norm(normal)
    return normal / mag if mag > 1e-9 else None


# ══════════════════════════════════════════════════════════════════════════════
# MASK EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_mask(vert_iso: np.ndarray, tss_iso: Optional[np.ndarray],
              sp_iso: Optional[np.ndarray],
              veridah_label: int,
              tss_label: Optional[int],
              sp_corpus_label: Optional[int] = SP_CORPUS) -> np.ndarray:
    mask = None
    if tss_iso is not None and tss_label is not None:
        candidate = (tss_iso == tss_label)
        if candidate.any():
            mask = candidate
    if mask is None or not mask.any():
        candidate = (vert_iso == veridah_label)
        if candidate.any():
            mask = candidate
    if mask is None or not mask.any():
        return np.zeros(vert_iso.shape, bool)
    if sp_iso is not None and sp_corpus_label is not None:
        corpus = (sp_iso == sp_corpus_label)
        if corpus.any():
            refined = mask & corpus
            if refined.sum() >= MIN_VOXELS_FOR_PCA:
                return refined.astype(bool)
    return mask.astype(bool)


def _sacrum_mask(vert_iso: np.ndarray,
                 tss_iso:  Optional[np.ndarray]) -> np.ndarray:
    if tss_iso is not None:
        s = (tss_iso == TSS_SAC)
        if s.any():
            return s.astype(bool)
    s = (vert_iso == VD_SAC)
    return s.astype(bool)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_vertebral_angles(
        sp_iso:           np.ndarray,
        vert_iso:         np.ndarray,
        tss_iso:          Optional[np.ndarray],
        tv_veridah_label: int,
        vox_mm:           float = 1.0,
        sp_corpus_label:  int   = SP_CORPUS,
        disc_above_dhi:   Optional[float] = None,
        disc_below_dhi:   Optional[float] = None,
        dhi_moderate_pct: float = 70.0,
        dhi_mild_pct:     float = 80.0,
) -> VertebralAnglesResult:
    """
    Compute all five vertebral angles from 3D segmentation masks.
    Includes sanity checking, cross-validation, and physiological plausibility
    filtering on all computed angles.
    """
    res   = VertebralAnglesResult()
    notes = res.computation_notes

    # Step 0: Detect cranio-caudal axis
    cc_axis, si_positive, orient_method = detect_cranio_caudal_axis(vert_iso, tss_iso)
    res.detected_cc_axis     = cc_axis
    res.detected_si_positive = si_positive
    res.orientation_method   = orient_method
    logger.info(f"  Orientation detected: cc_axis={cc_axis} si_positive={si_positive} "
                f"({orient_method})")

    # Map TV to neighbours
    VERIDAH_TO_TSS = {20:41, 21:42, 22:43, 23:44, 24:45, 25:45}
    tv_lbl  = tv_veridah_label
    tv_tss  = VERIDAH_TO_TSS.get(tv_lbl)
    tv1_lbl = tv_lbl - 1
    tv1_tss = VERIDAH_TO_TSS.get(tv1_lbl)
    l3_lbl  = VD_L3;  l3_tss = TSS_L3

    def _m(vd, tss_lbl):
        return _get_mask(vert_iso, tss_iso, sp_iso, vd, tss_lbl, sp_corpus_label)

    sac_mask = _sacrum_mask(vert_iso, tss_iso)
    tv_mask  = _m(tv_lbl,  tv_tss)
    tv1_mask = _m(tv1_lbl, tv1_tss)
    l3_mask  = _m(l3_lbl,  l3_tss)

    # Log mask sizes for diagnostics
    logger.debug(f"  Mask voxels: sac={sac_mask.sum()} tv={tv_mask.sum()} "
                 f"tv1={tv1_mask.sum()} l3={l3_mask.sum()}")

    def _fit(mask, surf):
        return _fit_endplate_plane(mask, surf, cc_axis=cc_axis, si_positive=si_positive)

    n_sac = _fit(sac_mask, 'superior')
    n_tv  = _fit(tv_mask,  'superior')
    n_tv1 = _fit(tv1_mask, 'superior')
    n_l3  = _fit(l3_mask,  'superior')

    if n_sac is None: notes.append('Sacrum endplate fit failed')
    if n_tv  is None: notes.append(f'TV (label {tv_lbl}) endplate fit failed')
    if n_tv1 is None: notes.append(f'TV-1 (label {tv1_lbl}) endplate fit failed')

    # A-angle
    if n_sac is not None:
        try:
            res.a_angle_deg = round(
                _angle_vs_vertical(n_sac, cc_axis=cc_axis, si_positive=si_positive), 1)
        except Exception as exc:
            notes.append(f'A-angle computation failed: {exc}')

    # B-angle
    if n_sac is not None and n_l3 is not None:
        try:
            res.b_angle_deg = round(
                _angle_between_normals(n_l3, n_sac, cc_axis=cc_axis), 1)
        except Exception as exc:
            notes.append(f'B-angle computation failed: {exc}')

    # D-angle
    if n_sac is not None and n_tv is not None:
        try:
            res.d_angle_deg = round(
                _angle_between_normals(n_tv, n_sac, cc_axis=cc_axis), 1)
        except Exception as exc:
            notes.append(f'D-angle computation failed: {exc}')

    # D1-angle
    if n_tv is not None and n_tv1 is not None:
        try:
            res.d1_angle_deg = round(
                _angle_between_normals(n_tv, n_tv1, cc_axis=cc_axis), 1)
        except Exception as exc:
            notes.append(f'D1-angle computation failed: {exc}')

    # delta-angle = D - D1
    if res.d_angle_deg is not None and res.d1_angle_deg is not None:
        res.delta_angle_deg = round(res.d_angle_deg - res.d1_angle_deg, 1)

    # C-angle
    try:
        junction_masks = [
            (tv1_mask, f'TV-1 (label {tv1_lbl})'),
            (tv_mask,  f'TV   (label {tv_lbl})'),
            (sac_mask, 'Sacrum'),
        ]
        pb_normals = []
        for jmask, jname in junction_masks:
            pbn = _posterior_body_normal(jmask, cc_axis=cc_axis)
            if pbn is not None:
                pb_normals.append(pbn)
            else:
                notes.append(f'C-angle: posterior-body normal unavailable for {jname}')

        if len(pb_normals) >= 2:
            angles_computed = [
                _angle_between_normals(pb_normals[i], pb_normals[j], cc_axis=cc_axis)
                for i in range(len(pb_normals))
                for j in range(i + 1, len(pb_normals))
            ]
            res.c_angle_deg = round(max(angles_computed), 1)
        else:
            notes.append('C-angle: insufficient posterior-body normals (need ≥2)')
    except Exception as exc:
        notes.append(f'C-angle computation failed: {exc}')

    # Run sanity checks — may null out implausible angles
    _run_sanity_checks(res)

    # Re-compute delta after sanity checks (D or D1 may have been nulled)
    if res.d_angle_deg is not None and res.d1_angle_deg is not None:
        res.delta_angle_deg = round(res.d_angle_deg - res.d1_angle_deg, 1)
    elif res.delta_angle_deg is not None and (res.d_angle_deg is None or res.d1_angle_deg is None):
        res.delta_angle_deg = None  # delta is invalid if either component was nulled

    # Diagnostic flags
    if res.delta_angle_deg is not None:
        res.delta_positive = (res.delta_angle_deg <= DELTA_TYPE2_THRESHOLD)
        res.delta_any_lstv = (res.delta_angle_deg <= DELTA_ANY_LSTV_THRESHOLD)

    if res.c_angle_deg is not None:
        res.c_positive = (res.c_angle_deg <= C_LSTV_THRESHOLD)

    if res.a_angle_deg is not None:
        res.a_angle_elevated = (res.a_angle_deg > A_INCREASED_THRESHOLD)

    if res.d_angle_deg is not None:
        res.d_angle_decreased = (res.d_angle_deg < D_DECREASED_THRESHOLD)

    if disc_above_dhi is not None and disc_below_dhi is not None:
        res.disc_pattern_lstv = (
            disc_above_dhi < dhi_moderate_pct and
            disc_below_dhi >= dhi_mild_pct
        )

    res.angles_available = (
        res.delta_angle_deg is not None or
        res.c_angle_deg     is not None
    )

    # LR keys
    lr_keys: List[str] = []
    if res.delta_positive:
        lr_keys.append('angle_delta_type2')
    elif res.delta_any_lstv:
        lr_keys.append('angle_delta_any_lstv')
    if res.c_positive:
        lr_keys.append('angle_c_lstv')
    if res.a_angle_elevated:
        lr_keys.append('angle_a_elevated')
    if res.d_angle_decreased:
        lr_keys.append('angle_d_decreased')
    res.angle_lr_keys_fired = lr_keys

    # Summary
    parts = []
    if res.delta_angle_deg is not None:
        parts.append(f"delta={res.delta_angle_deg:.1f}°"
                     f"{'⚠TYPE2' if res.delta_positive else ('⚠LSTV' if res.delta_any_lstv else '')}")
    if res.c_angle_deg is not None:
        parts.append(f"C={res.c_angle_deg:.1f}°{'⚠' if res.c_positive else ''}")
    if res.a_angle_deg is not None:
        parts.append(f"A={res.a_angle_deg:.1f}°{'↑' if res.a_angle_elevated else ''}")
    if res.d_angle_deg is not None:
        parts.append(f"D={res.d_angle_deg:.1f}°{'↓' if res.d_angle_decreased else ''}")
    if res.d1_angle_deg is not None:
        parts.append(f"D1={res.d1_angle_deg:.1f}°")
    if res.b_angle_deg is not None:
        parts.append(f"B={res.b_angle_deg:.1f}°")
    if res.disc_pattern_lstv:
        parts.append('disc-pattern⚠')
    parts.append(f"[orient:ax{cc_axis}{'↑' if si_positive else '↓'}]")
    if res.sanity_warnings:
        parts.append(f'[{len(res.sanity_warnings)} sanity warns]')
    if notes:
        parts.append(f'[{len(notes)} notes]')
    res.summary = '  '.join(parts) if parts else 'angles unavailable'

    return res


# ══════════════════════════════════════════════════════════════════════════════
# BAYESIAN UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def apply_angle_lr_updates(
        lo_sac:       float,
        lo_lumb:      float,
        angle_result: VertebralAnglesResult,
        existing_criteria: list,
) -> Tuple[float, float, list]:
    """Apply angle-based likelihood ratio updates to Bayesian log-odds."""
    try:
        from lstv_engine import RadiologicCriterion
        _has_rc = True
    except ImportError:
        _has_rc = False

    def _mk_criterion(**kw):
        if _has_rc:
            return RadiologicCriterion(**kw)
        return kw

    new_crit = []

    for key in angle_result.angle_lr_keys_fired:
        if key not in _ANGLE_LR:
            continue
        lr_ps, lr_ns, lr_pl, lr_nl = _ANGLE_LR[key]
        ds = float(np.log(max(1e-9, lr_ps)))
        dl = float(np.log(max(1e-9, lr_pl)))
        lo_sac  += ds
        lo_lumb += dl

        if key == 'angle_delta_type2':
            finding = (
                f"delta-angle = {angle_result.delta_angle_deg:.1f}° ≤ {DELTA_TYPE2_THRESHOLD}° "
                f"— primary predictor of Castellvi Type 2 LSTV "
                f"(sens 92.3%, spec 87.9%, NPV 99.5%; Seilanian Toosi 2025)"
            )
        elif key == 'angle_delta_any_lstv':
            finding = (
                f"delta-angle = {angle_result.delta_angle_deg:.1f}° ≤ {DELTA_ANY_LSTV_THRESHOLD}° "
                f"— predicts any LSTV (sens 66.7%, spec 52.2%; Seilanian Toosi 2025)"
            )
        elif key == 'angle_c_lstv':
            finding = (
                f"C-angle = {angle_result.c_angle_deg:.1f}° ≤ {C_LSTV_THRESHOLD}° "
                f"— predicts any LSTV (sens 72.2%, spec 57.6%; Seilanian Toosi 2025)"
            )
        elif key == 'angle_a_elevated':
            finding = (
                f"A-angle = {angle_result.a_angle_deg:.1f}° > {A_INCREASED_THRESHOLD}° "
                f"— elevated sacral tilt (OR 1.141/deg; Seilanian Toosi 2025)"
            )
        elif key == 'angle_d_decreased':
            finding = (
                f"D-angle = {angle_result.d_angle_deg:.1f}° < {D_DECREASED_THRESHOLD}° "
                f"— decreased lumbosacral angle (OR 0.719; Seilanian Toosi 2025)"
            )
        else:
            finding = f"Vertebral angle criterion: {key}"

        new_crit.append(_mk_criterion(
            name        = key,
            value       = f"{angle_result.delta_angle_deg or angle_result.c_angle_deg or 0:.1f}°",
            direction   = 'sacralization',
            strength    = 'primary' if key == 'angle_delta_type2' else 'secondary',
            lr_sac      = round(ds, 3),
            lr_lumb     = round(dl, 3),
            citation    = 'Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280',
            finding     = finding,
        ))

    return lo_sac, lo_lumb, new_crit


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST HARNESS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    print("lstv_angles.py v2 — orientation-robust sanity-checked test")
    print()

    def _make_spine_volume(shape, cc_axis, si_positive, sac_tilt_deg=40.0):
        vert = np.zeros(shape, np.int32)
        sp   = np.zeros(shape, np.int32)

        def _fill(label, cc_lo, cc_hi, perp_lo, perp_hi, perp2_lo, perp2_hi):
            idx = [slice(None)] * 3
            idx[cc_axis] = slice(
                cc_lo if si_positive else shape[cc_axis] - cc_hi - 1,
                (cc_hi if si_positive else shape[cc_axis] - cc_lo) + 1
            )
            axes = [0, 1, 2]
            non_cc = [a for a in axes if a != cc_axis]
            idx[non_cc[0]] = slice(perp_lo, perp_hi)
            idx[non_cc[1]] = slice(perp2_lo, perp2_hi)
            vert[tuple(idx)] = label
            sp[tuple(idx)]   = SP_CORPUS

        _fill(22, 55, 67, 15, 45, 10, 50)
        _fill(23, 40, 54, 15, 45, 10, 50)
        _fill(24, 24, 39, 15, 45, 10, 50)

        axes = [0, 1, 2]
        non_cc = [a for a in axes if a != cc_axis]
        ap_axis = non_cc[1]
        tilt_slope = np.tan(np.radians(sac_tilt_deg))

        for cc_i in range(0, 24):
            cc_real = cc_i if si_positive else shape[cc_axis] - cc_i - 1
            ap_offset = int(cc_i * tilt_slope)
            idx = [slice(None)] * 3
            idx[cc_axis]  = cc_real
            idx[non_cc[0]] = slice(15, 45)
            idx[ap_axis]   = slice(10 + ap_offset, 50 + ap_offset)
            vert[tuple(idx)] = 26
            sp[tuple(idx)]   = SP_CORPUS

        return vert, sp

    shape = (80, 80, 80)
    scenarios = [
        ("RAS canonical (cc=axis2, si_pos=True)",  2, True),
        ("Non-standard (cc=axis0, si_pos=True)",   0, True),
        ("Non-standard (cc=axis1, si_pos=True)",   1, True),
        ("Inverted SI   (cc=axis2, si_pos=False)", 2, False),
    ]

    all_pass = True
    for name, cc_ax, si_pos in scenarios:
        print(f"  Scenario: {name}")
        vert, sp = _make_spine_volume(shape, cc_ax, si_pos, sac_tilt_deg=40.0)
        result = compute_vertebral_angles(
            sp_iso=sp, vert_iso=vert, tss_iso=None,
            tv_veridah_label=VD_L5,
            disc_above_dhi=65.0, disc_below_dhi=85.0,
        )
        print(f"    Detected: cc_axis={result.detected_cc_axis} "
              f"si_positive={result.detected_si_positive}")
        print(f"    A={result.a_angle_deg}°  D={result.d_angle_deg}°  "
              f"D1={result.d1_angle_deg}°  delta={result.delta_angle_deg}°  "
              f"C={result.c_angle_deg}°")
        if result.sanity_warnings:
            print(f"    Sanity: {result.sanity_warnings}")
        print(f"    Summary: {result.summary}")

        orient_ok = (result.detected_cc_axis == cc_ax and
                     result.detected_si_positive == si_pos)
        a_ok = (result.a_angle_deg is not None and
                25.0 < result.a_angle_deg < 60.0)
        delta_ok = result.delta_angle_deg is not None

        status = "✓" if (orient_ok and a_ok and delta_ok) else "✗ FAIL"
        if not (orient_ok and a_ok and delta_ok):
            all_pass = False
        print(f"    {status}")
        print()

    # Test PCA TP height — CC component selection
    print("  TP height PCA test (CC-axis component, not longest):")
    # Build a TP that is wide mediolaterally (x) and short craniocaudally (z)
    # Long axis ~30mm in x, height ~10mm in z — longest PCA axis would be x,
    # but the correct measurement is the z-extent (CC height).
    tp_test = np.zeros((50, 50, 50), bool)
    tp_test[5:35, 20:30, 20:30] = True  # 30mm wide (x), 10mm tall (z)

    # Without disc info — should pick axis aligned with z (cc_axis=2)
    h_cc, axis_cc = measure_tp_height_pca(tp_test, vox_mm=1.0, tss_iso=None,
                                           cc_axis=2, si_positive=True)
    h_naive = float((tp_test.nonzero()[2].max() - tp_test.nonzero()[2].min() + 1))
    print(f"    Z-span (naive): {h_naive:.1f}mm  PCA CC height (no disc): {h_cc:.1f}mm  "
          f"axis={axis_cc.round(3)}")
    cc_ok = abs(h_cc - h_naive) < 3.0 and abs(axis_cc[2]) > 0.8
    print(f"    {'✓ CC axis selected correctly' if cc_ok else '✗ FAIL — wrong axis selected'}")

    # With synthetic disc TSS — discs at z=15 (L5S1=100) and z=25 (L4L5=95)
    tss_test = np.zeros((50, 50, 50), np.int32)
    tss_test[15:25, 15:35, 13:17] = 100   # L5-S1 disc at z~15 (caudal)
    tss_test[15:25, 15:35, 23:27] = 95    # L4-L5 disc at z~25 (cranial)
    h_disc, axis_disc = measure_tp_height_pca(tp_test, vox_mm=1.0, tss_iso=tss_test,
                                               cc_axis=2, si_positive=True)
    print(f"    PCA CC height (with disc axis): {h_disc:.1f}mm  axis={axis_disc.round(3)}")
    disc_ok = abs(h_disc - h_naive) < 3.0
    print(f"    {'✓ Disc-guided axis correct' if disc_ok else '✗ FAIL — disc-guided axis wrong'}")

    if all_pass:
        print("\n✓ All orientation scenarios passed")
        sys.exit(0)
    else:
        print("\n✗ One or more scenarios FAILED")
        sys.exit(1)
