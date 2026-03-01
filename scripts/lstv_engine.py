#!/usr/bin/env python3
"""
lstv_engine.py — LSTV Morphometrics Engine (Radiologically Grounded, v5)
=========================================================================
v5 ADDITIONS: Vertebral angle analysis (Seilanian Toosi et al. 2025).
All prior v4 functionality retained; angles are an additive layer.

WHAT'S NEW IN v5
-----------------
6. Vertebral angle analysis from sagittal MRI masks (Seilanian Toosi 2025):
     - A-angle: sacral tilt vs vertical (independent predictor, OR 1.141)
     - B-angle: L3 vs sacrum superior endplate (Chalian 2012)
     - C-angle: largest posterior-body angle at LS junction (≤35.5° → LSTV)
     - D-angle: TV vs S1 superior endplate
     - delta-angle = D − D1: PRIMARY criterion for Type 2 LSTV
       (≤8.5° → sens 92.3%, spec 87.9%)
   Computed by lstv_angles.py using PCA plane-fitting on segmentation masks.

7. Disc dehydration ASYMMETRY criterion added to Bayesian model:
     L4-L5 dehydrated + L5-S1 preserved = strongest combined predictor
     (OR 19.87 for non-dehydrated L5-S1; Seilanian Toosi 2025)

[All prior v4 documentation retained below]

RADIOLOGIC DEFINITION OF LSTV
------------------------------
An LSTV is a congenital spinal anomaly in which the last mobile lumbar
vertebra (the "transitional vertebra," TV) displays morphologic features
intermediate between a lumbar and a sacral segment...

PROBABILITY MODEL
-----------------
Uses Bayesian log-odds updating with a spine-clinic prior:
  P(sacralization) = 0.12  (Apazidis 2011)
  P(lumbarization) = 0.04
[See v4 docstring for full model description]

REFERENCES
----------
Castellvi AE et al. Spine. 1984;9(1):31–35.
Konin GP & Walz DM. Semin Musculoskelet Radiol. 2010;14(1):67–76.
Nardo L et al. Radiology. 2012;265(2):497–503.
Hughes RJ & Saifuddin A. Skeletal Radiol. 2006;35(5):299–316.
Farshad-Amacker NA et al. Eur Spine J. 2014;23(2):396–402.
Seyfert S. Neuroradiology. 1997;39(8):584–587.
Quinlan JF et al. J Bone Joint Surg Br. 1984;66(4):556–558.
Farfan HF et al. J Bone Joint Surg Am. 1972;54(3):492–510.
Panjabi MM et al. Spine. 1992;17(3):299–306.
Apazidis A et al. Spine. 2011;36(13):E854–E860.
Andrasinova T et al. Pain Physician. 2018;21(4):333–342.
Tokala DP et al. Eur Spine J. 2005;14(1):21–26.
O'Brien MF et al. Spine. 2019;44(16):1171–1179.
Luoma K et al. Spine. 2004;29(1):55–61.
MacDonald DB. Spine. 2002;27(24):2886–2891.
Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;1(5):271–280.  ← NEW v5
Chalian M et al. World J Radiol. 2012;4(3):97–101.              ← NEW v5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label, zoom as ndizoom

logger = logging.getLogger(__name__)

ISO_MM = 1.0

# ── TotalSpineSeg label tables ────────────────────────────────────────────────
TSS_CORD      = 1
TSS_CANAL     = 2
TSS_CERVICAL  : Dict[int, str] = {11:'C1',12:'C2',13:'C3',14:'C4',15:'C5',16:'C6',17:'C7'}
TSS_THORACIC  : Dict[int, str] = {
    21:'T1',22:'T2',23:'T3',24:'T4',25:'T5',26:'T6',
    27:'T7',28:'T8',29:'T9',30:'T10',31:'T11',32:'T12',
}
TSS_LUMBAR    : Dict[int, str] = {41:'L1',42:'L2',43:'L3',44:'L4',45:'L5'}
TSS_SACRUM    = 50
TSS_DISCS     : Dict[int, str] = {
    91:'T12-L1', 92:'L1-L2', 93:'L2-L3', 94:'L3-L4',
    95:'L4-L5',  100:'L5-S1',
}

# ── VERIDAH label tables ──────────────────────────────────────────────────────
VD_L1=20; VD_L2=21; VD_L3=22; VD_L4=23; VD_L5=24; VD_L6=25; VD_SAC=26
VD_IVD_BASE = 100
VD_EP_BASE  = 200
VERIDAH_LUMBAR  : Dict[int, str] = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6'}
VERIDAH_NAMES   : Dict[int, str] = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6',26:'Sacrum'}
VERIDAH_TV_SEARCH = [25, 24, 23, 22, 21, 20]
VD_TO_TSS_VERT  : Dict[int, int] = {20:41, 21:42, 22:43, 23:44, 24:45}

# ── SPINEPS subregion labels ──────────────────────────────────────────────────
SP_TP_L    = 43;  SP_TP_R   = 44
SP_SACRUM  = 26
SP_ARCUS   = 41;  SP_SPINOUS = 42
SP_SAL     = 45;  SP_SAR    = 46
SP_CORPUS  = 49
SP_CORD    = 60;  SP_CANAL  = 61

# ── Morphology thresholds ─────────────────────────────────────────────────────
TP_HEIGHT_MM      = 19.0
CONTACT_DIST_MM   = 2.0
TV_SHAPE_LUMBAR   = 0.68
TV_SHAPE_SACRAL   = 0.52
DHI_NORMAL_PCT    = 80.0
DHI_MILD_PCT      = 80.0
DHI_MODERATE_PCT  = 70.0
DHI_REDUCED_PCT   = 50.0
EXPECTED_LUMBAR   = 5
EXPECTED_THORACIC = 12

# ── Vertebral angle thresholds (Seilanian Toosi et al. 2025) ──────────────────
# Exported so 04_detect_lstv.py and 06_visualize_3d.py can import them directly.
DELTA_ANGLE_TYPE2_THRESHOLD = 8.5    # delta ≤ 8.5°  → Type 2 LSTV  Sens 92.3%  Spec 87.9%
C_ANGLE_LSTV_THRESHOLD      = 35.5   # C     ≤ 35.5° → any LSTV     Sens 72.2%  Spec 57.6%
A_ANGLE_NORMAL_MEDIAN       = 41.0   # A-angle normal median (Chalian 2012)
D_ANGLE_NORMAL_MEDIAN       = 13.5   # D-angle normal median (Seilanian Toosi 2025)

# ── Bayesian probability model ─────────────────────────────────────────────────
PRIOR_SACRALIZATION = 0.12
PRIOR_LUMBARIZATION = 0.04

_LR: Dict[str, Tuple[float, float, float, float]] = {
    # name:                        (LR+sac, LR-sac, LR+lumb, LR-lumb)
    'count_4':              (28.0,  0.90,   0.10,   1.05),
    'count_6':              ( 0.10, 1.05,  22.0,   0.95),
    'count_5':              ( 0.88, 1.00,   0.92,   1.00),
    'castellvi_iii_iv':     ( 9.5,  0.70,   1.0,    1.0),
    'castellvi_ii':         ( 5.2,  0.75,   1.0,    1.0),
    'castellvi_i':          ( 2.8,  0.85,   1.0,    1.0),
    'disc_absent':          ( 9.0,  0.88,   0.15,   1.05),
    'disc_dhi_lt50':        ( 5.8,  0.80,   0.20,   1.05),
    'disc_dhi_50_70':       ( 2.4,  0.90,   0.55,   1.02),
    'disc_dhi_70_80':       ( 1.5,  0.95,   0.80,   1.01),
    'disc_below_normal':    ( 0.28, 1.05,   4.5,    0.80),
    'body_sacral_like':     ( 3.8,  0.82,   0.30,   1.08),
    'body_transitional':    ( 1.9,  0.92,   0.60,   1.05),
    'body_lumbar_like':     ( 0.38, 1.08,   3.2,    0.88),
    'tv_l4_norm_lt80':      ( 2.2,  0.90,   0.55,   1.04),
    'tv_l4_norm_gt95':      ( 0.55, 1.03,   2.5,    0.92),
    'disc_ratio_low':       ( 3.5,  0.85,   0.40,   1.06),
    'disc_above_normal':    ( 1.4,  0.92,   1.2,    0.98),
    'l6_disc_preserved':    ( 0.20, 1.05,   6.5,    0.80),
    # v5: Disc dehydration asymmetric pattern (Seilanian Toosi 2025)
    # L4-L5 dehydrated + L5-S1 preserved = OR 19.87 for non-dehydrated LS disc
    'disc_pattern_l4dehy_l5preserved': (4.2, 0.60, 3.8, 0.65),
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class LSTVMaskSet:
    study_id:    str
    sp_iso:      np.ndarray
    vert_iso:    np.ndarray
    tss_iso:     Optional[np.ndarray]
    sp_labels:   frozenset
    vert_labels: frozenset
    tss_labels:  frozenset


@dataclass
class DiscMetrics:
    level:       str
    height_mm:   Optional[float] = None
    vert_sup_h:  Optional[float] = None
    vert_inf_h:  Optional[float] = None
    dhi_pct:     Optional[float] = None
    grade:       Optional[str]   = None
    source:      Optional[str]   = None
    is_absent:   bool            = False


@dataclass
class TVBodyShape:
    h_mm:        Optional[float] = None
    ap_mm:       Optional[float] = None
    ml_mm:       Optional[float] = None
    h_ap_ratio:  Optional[float] = None
    h_ml_ratio:  Optional[float] = None
    shape_class: Optional[str]   = None
    ref_l4_h_ap: Optional[float] = None
    norm_ratio:  Optional[float] = None
    ref_l3_h_ap:      Optional[float] = None
    caudal_gradient:  Optional[float] = None
    gradient_note:    Optional[str]   = None
    source:           Optional[str]   = None


@dataclass
class RibAnomalyResult:
    thoracic_count:      Optional[int]  = None
    expected_thoracic:   int            = EXPECTED_THORACIC
    count_anomaly:       bool           = False
    count_description:   Optional[str]  = None
    lumbar_rib_l1:       bool           = False
    lumbar_rib_l1_h_mm:  Optional[float]= None
    any_anomaly:         bool           = False
    description:         Optional[str]  = None


@dataclass
class RadiologicCriterion:
    name:        str
    value:       str
    direction:   str
    strength:    str
    lr_sac:      float
    lr_lumb:     float
    citation:    str
    finding:     str


@dataclass
class LSTVProbabilities:
    p_sacralization:      float
    p_lumbarization:      float
    p_normal:             float
    p_transitional:       float
    log_odds_sac_evidence:  float
    log_odds_lumb_evidence: float
    n_criteria:             int
    prior_sacralization: float = PRIOR_SACRALIZATION
    prior_lumbarization: float = PRIOR_LUMBARIZATION
    dominant_class:      str   = ''
    confidence_pct:      float = 0.0
    calibration_note:    str   = ''


@dataclass
class SurgicalRelevance:
    wrong_level_risk:        str   = 'low'
    wrong_level_risk_pct:    float = 0.05
    level_ambiguity_note:    str   = ''
    nerve_root_ambiguity:    bool  = False
    nerve_root_note:         str   = ''
    bertolotti_probability:  float      = 0.0
    bertolotti_criteria:     List[str]  = field(default_factory=list)
    surgical_flags:                 List[str] = field(default_factory=list)
    approach_considerations:        List[str] = field(default_factory=list)
    recommended_counting_method:    str       = ''
    intraop_neuromonitoring_note:   str       = ''
    level_identification_protocol:  str       = ''
    calibration_note:               str       = ''


@dataclass
class LSTVMorphometrics:
    """
    Complete LSTV morphometric result for one study.
    v5: adds vertebral_angles (Seilanian Toosi 2025) and disc asymmetry criterion.
    """
    study_id: str
    error:    Optional[str] = None

    lumbar_count_tss:       Optional[int] = None
    lumbar_count_veridah:   Optional[int] = None
    lumbar_count_consensus: Optional[int] = None
    lumbar_count_anomaly:   bool          = False
    lumbar_count_note:      Optional[str] = None

    tv_label_veridah:  Optional[int] = None
    tv_name:           Optional[str] = None
    tv_tss_label:      Optional[int] = None
    has_l6:            bool          = False

    tv_shape:  Optional[TVBodyShape] = None
    disc_above: Optional[DiscMetrics] = None
    disc_below: Optional[DiscMetrics] = None

    relative_disc_ratio: Optional[float] = None
    relative_disc_note:  Optional[str]   = None

    rib_anomaly: Optional[RibAnomalyResult] = None

    lstv_phenotype:       Optional[str] = None
    phenotype_confidence: Optional[str] = None
    phenotype_criteria:   List[str]     = field(default_factory=list)
    phenotype_rationale:  Optional[str] = None
    primary_criteria_met: List[str]     = field(default_factory=list)

    probabilities:         Optional[LSTVProbabilities]     = None
    radiologic_evidence:   List[RadiologicCriterion]       = field(default_factory=list)
    surgical_relevance:    Optional[SurgicalRelevance]     = None

    # ── v5: Vertebral angles (Seilanian Toosi 2025) ───────────────────────────
    # delta-angle ≤ 8.5° → Type 2 LSTV (sens 92.3%, spec 87.9%)
    # C-angle ≤ 35.5°    → any LSTV    (sens 72.2%, spec 57.6%)
    vertebral_angles:  Optional[object] = None  # VertebralAnglesResult from lstv_angles.py

    def to_dict(self) -> dict:
        d = asdict(self)
        # vertebral_angles is from lstv_angles — convert if it has to_dict()
        va = self.vertebral_angles
        if va is not None and hasattr(va, 'to_dict'):
            d['vertebral_angles'] = va.to_dict()
        return d


# ═════════════════════════════════════════════════════════════════════════════
# NIfTI HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path.name}: cannot reduce to 3D (shape={data.shape})")
    return data, nii


def _voxel_mm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def _resample(vol: np.ndarray, vox_mm: np.ndarray, target: float = ISO_MM) -> np.ndarray:
    factors = (vox_mm / target).tolist()
    return ndizoom(vol.astype(np.int32), factors,
                   order=0, mode='nearest', prefilter=False).astype(np.int32)


# ═════════════════════════════════════════════════════════════════════════════
# MASK LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_lstv_masks(study_id: str,
                    spineps_dir: Path,
                    totalspine_dir: Path) -> LSTVMaskSet:
    seg_dir   = spineps_dir / 'segmentations' / study_id
    sp_path   = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path  = (totalspine_dir / study_id / 'sagittal'
                 / f"{study_id}_sagittal_labeled.nii.gz")

    sp_raw, sp_nii  = _load_canonical(sp_path)
    vert_raw, _     = _load_canonical(vert_path)
    vox_mm          = _voxel_mm(sp_nii)

    sp_iso   = _resample(sp_raw.astype(np.int32),   vox_mm)
    vert_iso = _resample(vert_raw.astype(np.int32), vox_mm)

    tss_iso = None
    if tss_path.exists():
        try:
            tss_raw, tss_nii = _load_canonical(tss_path)
            tss_vox          = _voxel_mm(tss_nii)
            tss_iso          = _resample(tss_raw.astype(np.int32), tss_vox)
        except Exception as exc:
            logger.warning(f"[{study_id}] TSS load failed: {exc}")

    return LSTVMaskSet(
        study_id    = study_id,
        sp_iso      = sp_iso,
        vert_iso    = vert_iso,
        tss_iso     = tss_iso,
        sp_labels   = frozenset(np.unique(sp_iso).tolist())   - {0},
        vert_labels = frozenset(np.unique(vert_iso).tolist()) - {0},
        tss_labels  = (frozenset(np.unique(tss_iso).tolist()) - {0}
                       if tss_iso is not None else frozenset()),
    )


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════

def _si_height(mask: np.ndarray) -> Optional[float]:
    if not mask.any(): return None
    zc = np.where(mask)[2]
    return float((int(zc.max()) - int(zc.min()) + 1) * ISO_MM)


def _ap_depth(mask: np.ndarray) -> Optional[float]:
    if not mask.any(): return None
    yc = np.where(mask)[1]
    return float((int(yc.max()) - int(yc.min()) + 1) * ISO_MM)


def _ml_width(mask: np.ndarray) -> Optional[float]:
    if not mask.any(): return None
    xc = np.where(mask)[0]
    return float((int(xc.max()) - int(xc.min()) + 1) * ISO_MM)


# ═════════════════════════════════════════════════════════════════════════════
# LUMBAR COUNT  (unchanged from v4)
# ═════════════════════════════════════════════════════════════════════════════

def count_lumbar_tss(tss_iso: np.ndarray, tss_labels: frozenset) -> Tuple[int, List[str]]:
    detected = [name for lbl, name in TSS_LUMBAR.items() if lbl in tss_labels]
    return len(detected), detected


def detect_l6_veridah(vert_labels: frozenset) -> bool:
    return VD_L6 in vert_labels


def count_lumbar_veridah(vert_labels: frozenset) -> Tuple[int, List[str]]:
    detected = [name for lbl, name in VERIDAH_LUMBAR.items() if lbl in vert_labels]
    return len(detected), detected


def reconcile_lumbar_count(tss_count: int, veridah_count: int,
                            tss_names: List[str], veridah_names: List[str],
                            vert_labels: frozenset) -> Tuple[int, str]:
    has_l6_veridah = detect_l6_veridah(vert_labels)
    if has_l6_veridah:
        consensus = tss_count + 1
        note = (f"TSS={tss_count} + VERIDAH L6 label 25 → consensus={consensus} "
                f"lumbar vertebrae — LUMBARIZATION indicator (Hughes & Saifuddin 2006)")
        return consensus, note
    if tss_count == veridah_count:
        return tss_count, f"TSS={tss_count}, VERIDAH={veridah_count} — consistent"
    if tss_count < veridah_count:
        return tss_count, (f"TSS={tss_count} < VERIDAH={veridah_count} without L6; "
                           f"TSS trusted (VERIDAH over-segmentation likely)")
    return tss_count, (f"TSS={tss_count} > VERIDAH={veridah_count}; TSS trusted")


# ═════════════════════════════════════════════════════════════════════════════
# TV BODY SHAPE  (unchanged from v4)
# ═════════════════════════════════════════════════════════════════════════════

def _vert_shape(iso: np.ndarray, vert_label: int, source: str) -> Optional[TVBodyShape]:
    mask = (iso == vert_label)
    if not mask.any(): return None
    h = _si_height(mask); ap = _ap_depth(mask); ml = _ml_width(mask)
    if h is None or ap is None or ap == 0: return None
    h_ap  = h / ap
    h_ml  = (h / ml) if (ml and ml > 0) else None
    shape = ('lumbar-like' if h_ap > TV_SHAPE_LUMBAR
             else 'sacral-like' if h_ap < TV_SHAPE_SACRAL
             else 'transitional')
    return TVBodyShape(h_mm=h, ap_mm=ap, ml_mm=ml,
                       h_ap_ratio=round(h_ap, 3),
                       h_ml_ratio=round(h_ml, 3) if h_ml else None,
                       shape_class=shape, source=source)


def analyze_tv_body_shape(masks: LSTVMaskSet,
                           tv_veridah_label: int,
                           tv_tss_label: Optional[int]) -> TVBodyShape:
    shape: Optional[TVBodyShape] = None
    if tv_tss_label is not None and masks.tss_iso is not None:
        shape = _vert_shape(masks.tss_iso, tv_tss_label, 'TSS')
    if shape is None:
        shape = _vert_shape(masks.vert_iso, tv_veridah_label, 'VERIDAH')
    if shape is None:
        return TVBodyShape()

    l4_shape: Optional[TVBodyShape] = None
    if masks.tss_iso is not None and 44 in masks.tss_labels:
        l4_shape = _vert_shape(masks.tss_iso, 44, 'TSS')
    if l4_shape is None and VD_L4 in masks.vert_labels:
        l4_shape = _vert_shape(masks.vert_iso, VD_L4, 'VERIDAH')
    if l4_shape and l4_shape.h_ap_ratio and shape.h_ap_ratio:
        shape.ref_l4_h_ap = l4_shape.h_ap_ratio
        shape.norm_ratio  = round(shape.h_ap_ratio / l4_shape.h_ap_ratio, 3)

    l3_shape: Optional[TVBodyShape] = None
    if masks.tss_iso is not None and 43 in masks.tss_labels:
        l3_shape = _vert_shape(masks.tss_iso, 43, 'TSS')
    if l3_shape is None and VD_L3 in masks.vert_labels:
        l3_shape = _vert_shape(masks.vert_iso, VD_L3, 'VERIDAH')
    if (l3_shape and l3_shape.h_ap_ratio and l4_shape and l4_shape.h_ap_ratio
            and shape.h_ap_ratio):
        shape.ref_l3_h_ap = l3_shape.h_ap_ratio
        vals = [l3_shape.h_ap_ratio, l4_shape.h_ap_ratio, shape.h_ap_ratio]
        xs   = [0, 1, 2]; n = len(xs)
        sx = sum(xs); sy = sum(vals); sxy = sum(x*y for x,y in zip(xs,vals))
        sx2 = sum(x*x for x in xs)
        slope = (n * sxy - sx * sy) / (n * sx2 - sx * sx + 1e-9)
        shape.caudal_gradient = round(float(slope), 4)
        if slope < -0.04:
            shape.gradient_note = (
                f"Steep caudal H/AP gradient ({slope:.3f}/level): "
                f"L3={l3_shape.h_ap_ratio:.2f} → L4={l4_shape.h_ap_ratio:.2f} → "
                f"TV={shape.h_ap_ratio:.2f} — progressive shortening supports "
                f"sacral incorporation (Nardo 2012)")
        elif slope > 0.02:
            shape.gradient_note = (
                f"Positive caudal gradient ({slope:.3f}/level): "
                f"TV taller than upper lumbar — consistent with lumbarization")
        else:
            shape.gradient_note = (
                f"Flat caudal gradient ({slope:.3f}/level): "
                f"consistent body height trend L3→L4→TV")
    return shape


# ═════════════════════════════════════════════════════════════════════════════
# DISC HEIGHT METRICS  (unchanged from v4)
# ═════════════════════════════════════════════════════════════════════════════

def _disc_height_mm(iso: np.ndarray, label: int) -> Optional[float]:
    return _si_height(iso == label)


def _vert_si_height(iso: np.ndarray, label: int) -> Optional[float]:
    return _si_height(iso == label)


def _disc_grade(dhi: Optional[float]) -> Optional[str]:
    if dhi is None: return None
    if dhi >= DHI_MILD_PCT:     return 'Normal'
    if dhi >= DHI_MODERATE_PCT: return 'Mild reduction'
    if dhi >= DHI_REDUCED_PCT:  return 'Moderate reduction'
    return 'Severely reduced / absent'


def measure_disc_metrics(masks: LSTVMaskSet,
                          disc_label_tss: Optional[int],
                          sup_tss_label: Optional[int],
                          inf_tss_label: Optional[int],
                          sup_vd_label:  Optional[int],
                          inf_vd_label:  Optional[int],
                          level_name: str) -> DiscMetrics:
    dm = DiscMetrics(level=level_name)
    disc_h = None; sup_h = None; inf_h = None

    if (disc_label_tss is not None and masks.tss_iso is not None
            and disc_label_tss in masks.tss_labels):
        disc_h    = _disc_height_mm(masks.tss_iso, disc_label_tss)
        dm.source = 'TSS'
        if sup_tss_label and sup_tss_label in masks.tss_labels:
            sup_h = _vert_si_height(masks.tss_iso, sup_tss_label)
        if inf_tss_label and inf_tss_label in masks.tss_labels:
            inf_h = _vert_si_height(masks.tss_iso, inf_tss_label)

    if disc_h is None and sup_vd_label is not None:
        vd_disc_lbl = VD_IVD_BASE + sup_vd_label
        if vd_disc_lbl in masks.vert_labels:
            disc_h    = _disc_height_mm(masks.vert_iso, vd_disc_lbl)
            dm.source = 'VERIDAH'
        if sup_vd_label in masks.vert_labels:
            sup_h = _vert_si_height(masks.vert_iso, sup_vd_label)
        if inf_vd_label and inf_vd_label in masks.vert_labels:
            inf_h = _vert_si_height(masks.vert_iso, inf_vd_label)

    dm.height_mm  = round(disc_h, 2) if disc_h else None
    dm.vert_sup_h = round(sup_h, 2)  if sup_h  else None
    dm.vert_inf_h = round(inf_h, 2)  if inf_h  else None
    dm.is_absent  = (disc_h is None or disc_h == 0.0)

    denom_heights = [h for h in (sup_h, inf_h) if h and h > 0]
    if disc_h and denom_heights:
        ref_h      = float(np.mean(denom_heights))
        dhi        = (disc_h / ref_h) * 100.0
        dm.dhi_pct = round(dhi, 1)
        dm.grade   = _disc_grade(dhi)

    return dm


def get_tv_adjacent_discs(masks: LSTVMaskSet,
                           tv_veridah_label: int,
                           tv_tss_label: Optional[int]) -> Tuple[DiscMetrics, DiscMetrics]:
    if tv_veridah_label == VD_L5:
        disc_above = measure_disc_metrics(masks, 95, 44, 45, VD_L4, VD_L5, 'L4-L5')
        disc_below = measure_disc_metrics(masks, 100, 45, TSS_SACRUM, VD_L5, VD_SAC, 'L5-S1')
    elif tv_veridah_label == VD_L4:
        disc_above = measure_disc_metrics(masks, 94, 43, 44, VD_L3, VD_L4, 'L3-L4')
        disc_below = measure_disc_metrics(masks, 95, 44, 45, VD_L4, VD_L5, 'L4-L5')
    elif tv_veridah_label == VD_L6:
        disc_above = measure_disc_metrics(masks, None, None, None, VD_L5, VD_L6, 'L5-L6')
        disc_below = measure_disc_metrics(masks, None, None, None, VD_L6, VD_SAC, 'L6-S1')
    else:
        tv_name    = VERIDAH_NAMES.get(tv_veridah_label, str(tv_veridah_label))
        disc_above = DiscMetrics(level=f'above-{tv_name}')
        disc_below = DiscMetrics(level=f'below-{tv_name}')
    return disc_above, disc_below


def compute_relative_disc_ratio(disc_above: Optional[DiscMetrics],
                                  disc_below: Optional[DiscMetrics]) -> Tuple[Optional[float], str]:
    if disc_above is None or disc_below is None:
        return None, 'Adjacent disc data unavailable'
    if disc_above.dhi_pct is None or disc_below.dhi_pct is None:
        return None, 'DHI unavailable for one or both adjacent discs'
    if disc_above.dhi_pct <= 0:
        return None, 'Disc above has zero DHI — ratio undefined'
    ratio = round(disc_below.dhi_pct / disc_above.dhi_pct, 3)
    if ratio < 0.50:
        note = (f"TV-disc/above-disc DHI ratio = {ratio:.2f} — severely disproportionate "
                f"disc narrowing (strongly supports sacralization, Farshad-Amacker 2014)")
    elif ratio < 0.65:
        note = (f"TV-disc/above-disc DHI ratio = {ratio:.2f} — disproportionate narrowing "
                f"at TV level (Farshad-Amacker 2014; Konin 2010)")
    elif ratio < 0.80:
        note = (f"TV-disc/above-disc DHI ratio = {ratio:.2f} — mild relative narrowing")
    else:
        note = (f"TV-disc/above-disc DHI ratio = {ratio:.2f} — proportionate disc heights")
    return ratio, note


# ═════════════════════════════════════════════════════════════════════════════
# RIB ANOMALY  (unchanged from v4)
# ═════════════════════════════════════════════════════════════════════════════

def detect_rib_anomaly(masks: LSTVMaskSet) -> RibAnomalyResult:
    result = RibAnomalyResult()
    if masks.tss_iso is not None:
        thr_present = [lbl for lbl in TSS_THORACIC if lbl in masks.tss_labels]
        result.thoracic_count = len(thr_present)
        if result.thoracic_count != EXPECTED_THORACIC:
            result.count_anomaly = True
            delta = result.thoracic_count - EXPECTED_THORACIC
            if delta < 0:
                result.count_description = (
                    f"Only {result.thoracic_count} thoracic vertebrae detected "
                    f"(expected {EXPECTED_THORACIC})")
            else:
                result.count_description = (
                    f"{result.thoracic_count} thoracic vertebrae detected "
                    f"(expected {EXPECTED_THORACIC})")

    if SP_TP_L in masks.sp_labels or SP_TP_R in masks.sp_labels:
        l1_mask = (masks.vert_iso == VD_L1) if VD_L1 in masks.vert_labels else None
        if l1_mask is not None and l1_mask.any():
            zc   = np.where(l1_mask)[2]
            z_lo = max(0, int(zc.min()) - 5)
            z_hi = min(masks.sp_iso.shape[2] - 1, int(zc.max()) + 5)
            max_tp_h = 0.0
            for tp_lbl in (SP_TP_L, SP_TP_R):
                if tp_lbl not in masks.sp_labels: continue
                tp_full  = (masks.sp_iso == tp_lbl)
                tp_at_l1 = np.zeros_like(tp_full)
                tp_at_l1[:, :, z_lo:z_hi + 1] = tp_full[:, :, z_lo:z_hi + 1]
                if tp_at_l1.any():
                    h = _si_height(tp_at_l1)
                    if h and h > max_tp_h: max_tp_h = h
            if max_tp_h > 0:
                result.lumbar_rib_l1_h_mm = round(max_tp_h, 1)
                if max_tp_h >= TP_HEIGHT_MM:
                    result.lumbar_rib_l1 = True

    result.any_anomaly = result.count_anomaly or result.lumbar_rib_l1
    parts = []
    if result.count_description: parts.append(result.count_description)
    if result.lumbar_rib_l1:
        parts.append(
            f"Suspected lumbar rib at L1 (L1 TP height="
            f"{result.lumbar_rib_l1_h_mm:.1f}mm ≥ {TP_HEIGHT_MM}mm)")
    result.description = '; '.join(parts) if parts else None
    return result


# ═════════════════════════════════════════════════════════════════════════════
# BAYESIAN PROBABILITY MODEL  (v5: adds disc asymmetry pattern)
# ═════════════════════════════════════════════════════════════════════════════

def _log_odds(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(lo: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(lo, -30, 30))))


def _apply_lr(lo_sac: float, lo_lumb: float,
              key: str, positive: bool = True) -> Tuple[float, float, float, float]:
    lr_ps, lr_ns, lr_pl, lr_nl = _LR[key]
    if positive:
        ds = np.log(lr_ps); dl = np.log(lr_pl)
    else:
        ds = np.log(lr_ns); dl = np.log(lr_nl)
    return lo_sac + ds, lo_lumb + dl, float(ds), float(dl)


def compute_lstv_probability(
        lumbar_count:   int,
        castellvi_type: Optional[str],
        tv_shape:       Optional[TVBodyShape],
        disc_above:     Optional[DiscMetrics],
        disc_below:     Optional[DiscMetrics],
        rel_disc_ratio: Optional[float] = None,
) -> Tuple[LSTVProbabilities, List[RadiologicCriterion]]:
    """
    Bayesian posterior probability for LSTV phenotype.
    v5: adds disc dehydration asymmetry pattern criterion (Seilanian Toosi 2025).
    Angle-based LR updates are applied separately in analyze_lstv() step 8.5.
    """
    lo_sac  = _log_odds(PRIOR_SACRALIZATION)
    lo_lumb = _log_odds(PRIOR_LUMBARIZATION)
    evidence_sac  = 0.0
    evidence_lumb = 0.0
    criteria:  List[RadiologicCriterion] = []
    n_fired = 0

    has_castellvi = bool(castellvi_type and castellvi_type not in ('None', None))
    ct = castellvi_type or ''

    # ── Lumbar count ─────────────────────────────────────────────────────────
    if lumbar_count == 4:
        lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, 'count_4')
        evidence_sac += ds; evidence_lumb += dl; n_fired += 1
        criteria.append(RadiologicCriterion(
            name='lumbar_count', value='4', direction='sacralization', strength='primary',
            lr_sac=round(ds,3), lr_lumb=round(dl,3),
            citation='Nardo L et al. Radiology. 2012;265(2):497–503',
            finding='4 lumbar vertebrae — L5 incorporated into sacrum (primary sacralization criterion)'))
    elif lumbar_count == 6:
        lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, 'count_6')
        evidence_sac += ds; evidence_lumb += dl; n_fired += 1
        criteria.append(RadiologicCriterion(
            name='lumbar_count', value='6', direction='lumbarization', strength='primary',
            lr_sac=round(ds,3), lr_lumb=round(dl,3),
            citation='Hughes RJ & Saifuddin A. Skeletal Radiol. 2006;35(5):299–316',
            finding='6 lumbar vertebrae — S1 acquired lumbar characteristics (primary lumbarization criterion)'))
    else:
        lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, 'count_5')
        evidence_sac += ds; evidence_lumb += dl
        criteria.append(RadiologicCriterion(
            name='lumbar_count', value='5', direction='normal', strength='supporting',
            lr_sac=round(ds,3), lr_lumb=round(dl,3),
            citation='Apazidis A et al. Spine. 2011;36(13):E854–E860',
            finding='5 lumbar vertebrae — normal count'))

    # ── Castellvi ────────────────────────────────────────────────────────────
    if has_castellvi:
        if any(x in ct for x in ('III', 'IV')):
            key = 'castellvi_iii_iv'
            desc = f'Castellvi {ct} — complete/mixed bony fusion of TP with sacral ala'
        elif 'II' in ct:
            key = 'castellvi_ii'
            desc = f'Castellvi {ct} — diarthrodial pseudo-articulation, fibrocartilaginous joint'
        else:
            key = 'castellvi_i'
            desc = f'Castellvi {ct} — dysplastic TP ≥{TP_HEIGHT_MM}mm, no sacral contact'
        lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, key)
        evidence_sac += ds; n_fired += 1
        criteria.append(RadiologicCriterion(
            name='castellvi', value=ct, direction='sacralization', strength='primary',
            lr_sac=round(ds,3), lr_lumb=round(dl,3),
            citation='Castellvi AE et al. Spine. 1984;9(1):31–35; Konin GP & Walz DM. 2010',
            finding=desc))

    # ── Disc below ───────────────────────────────────────────────────────────
    if disc_below:
        dhi = disc_below.dhi_pct
        level = disc_below.level
        if disc_below.is_absent:
            lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, 'disc_absent')
            evidence_sac += ds; evidence_lumb += dl; n_fired += 1
            criteria.append(RadiologicCriterion(
                name='disc_below', value='absent', direction='sacralization', strength='primary',
                lr_sac=round(ds,3), lr_lumb=round(dl,3),
                citation='Seyfert S. Neuroradiology. 1997;39(8):584–587',
                finding=f'Disc {level} absent — possible complete disc fusion'))
        elif dhi is not None:
            if dhi < DHI_REDUCED_PCT:
                key = 'disc_dhi_lt50'; strength = 'primary'; direction = 'sacralization'
                finding = (f'Disc {level} severely reduced: DHI={dhi:.0f}% < {DHI_REDUCED_PCT}% — '
                           f'most reliable MRI sacralization sign (Seyfert 1997; Farfan 1972)')
            elif dhi < DHI_MODERATE_PCT:
                key = 'disc_dhi_50_70'; strength = 'secondary'; direction = 'sacralization'
                finding = f'Disc {level} moderately reduced: DHI={dhi:.0f}%'
            elif dhi < DHI_MILD_PCT:
                key = 'disc_dhi_70_80'; strength = 'supporting'; direction = 'sacralization'
                finding = f'Disc {level} mildly reduced: DHI={dhi:.0f}%'
            else:
                key = 'disc_below_normal'; strength = 'primary'; direction = 'lumbarization'
                finding = (f'Disc {level} preserved: DHI={dhi:.0f}% ≥ {DHI_MILD_PCT}% — '
                           f'mobile disc below TV supports lumbarization (Konin 2010)')
            lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, key)
            evidence_sac += ds; evidence_lumb += dl; n_fired += 1
            criteria.append(RadiologicCriterion(
                name='disc_below_dhi', value=f'{dhi:.0f}%',
                direction=direction, strength=strength,
                lr_sac=round(ds,3), lr_lumb=round(dl,3),
                citation='Seyfert S. 1997; Farfan HF et al. 1972; Konin GP & Walz DM. 2010',
                finding=finding))

    # ── Relative disc ratio ───────────────────────────────────────────────────
    if rel_disc_ratio is not None and disc_above is not None and disc_above.dhi_pct:
        if rel_disc_ratio < 0.65:
            lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, 'disc_ratio_low')
            evidence_sac += ds; evidence_lumb += dl; n_fired += 1
            criteria.append(RadiologicCriterion(
                name='relative_disc_ratio', value=f'{rel_disc_ratio:.2f}',
                direction='sacralization', strength='secondary',
                lr_sac=round(ds,3), lr_lumb=round(dl,3),
                citation='Farshad-Amacker NA et al. Eur Spine J. 2014;23(2):396–402',
                finding=(f'TV-disc/above-disc DHI ratio = {rel_disc_ratio:.2f} < 0.65 — '
                         f'disproportionate lumbosacral narrowing (Farshad-Amacker 2014)')))

    # ── TV body shape ─────────────────────────────────────────────────────────
    if tv_shape and tv_shape.h_ap_ratio:
        sc = tv_shape.shape_class; h_ap = tv_shape.h_ap_ratio
        if sc == 'sacral-like':
            key = 'body_sacral_like'; strength = 'secondary'; direction = 'sacralization'
            finding = (f'TV body sacral-like: H/AP={h_ap:.2f} < {TV_SHAPE_SACRAL} (Nardo 2012)')
        elif sc == 'transitional':
            key = 'body_transitional'; strength = 'supporting'; direction = 'sacralization'
            finding = f'TV body transitional: H/AP={h_ap:.2f} (Nardo 2012)'
        else:
            key = 'body_lumbar_like'; strength = 'supporting'
            direction = 'lumbarization' if lumbar_count == 6 else 'normal'
            finding = (f'TV body lumbar-like: H/AP={h_ap:.2f} > {TV_SHAPE_LUMBAR} (Nardo 2012)')
        lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, key)
        evidence_sac += ds; evidence_lumb += dl; n_fired += 1
        criteria.append(RadiologicCriterion(
            name='tv_body_shape', value=f'H/AP={h_ap:.2f} ({sc})',
            direction=direction, strength=strength,
            lr_sac=round(ds,3), lr_lumb=round(dl,3),
            citation='Nardo L et al. Radiology. 2012;265(2):497–503; Panjabi MM et al. Spine. 1992',
            finding=finding))

        if tv_shape.norm_ratio:
            nr = tv_shape.norm_ratio
            if nr < 0.80:
                lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, 'tv_l4_norm_lt80')
                evidence_sac += ds; n_fired += 1
                criteria.append(RadiologicCriterion(
                    name='tv_l4_normalised_ratio', value=f'{nr:.2f}',
                    direction='sacralization', strength='supporting',
                    lr_sac=round(ds,3), lr_lumb=round(dl,3),
                    citation='Panjabi MM et al. Spine. 1992',
                    finding=f'TV/L4 H:AP={nr:.2f} < 0.80 — TV notably squarer than L4'))
            elif nr > 0.95 and lumbar_count == 6:
                lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, 'tv_l4_norm_gt95')
                evidence_lumb += dl; n_fired += 1
                criteria.append(RadiologicCriterion(
                    name='tv_l4_normalised_ratio', value=f'{nr:.2f}',
                    direction='lumbarization', strength='supporting',
                    lr_sac=round(ds,3), lr_lumb=round(dl,3),
                    citation='Panjabi MM et al. Spine. 1992',
                    finding=f'TV/L4 H:AP={nr:.2f} > 0.95 with L6 — supports lumbarization'))

        if tv_shape.caudal_gradient is not None:
            grad = tv_shape.caudal_gradient
            if grad < -0.04:
                criteria.append(RadiologicCriterion(
                    name='caudal_body_gradient', value=f'{grad:.4f}/level',
                    direction='sacralization', strength='supporting',
                    lr_sac=0.0, lr_lumb=0.0,
                    citation='Nardo L et al. Radiology. 2012',
                    finding=f'Caudal H/AP gradient = {grad:.3f}/level — progressive shortening'))

    # ── Disc above ────────────────────────────────────────────────────────────
    if disc_above and disc_above.dhi_pct and disc_above.dhi_pct >= DHI_MILD_PCT:
        lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb, 'disc_above_normal')
        evidence_sac += ds; n_fired += 1
        criteria.append(RadiologicCriterion(
            name='disc_above_dhi', value=f'{disc_above.dhi_pct:.0f}%',
            direction='supporting', strength='supporting',
            lr_sac=round(ds,3), lr_lumb=round(dl,3),
            citation='Konin GP & Walz DM. Semin Musculoskelet Radiol. 2010',
            finding=(f'Disc above TV ({disc_above.level}) normal: DHI={disc_above.dhi_pct:.0f}%')))

    # ── v5: Disc dehydration asymmetry pattern (Seilanian Toosi 2025) ─────────
    # L4-L5 dehydrated + L5-S1 preserved = OR 19.87 for L5-S1 non-dehydration
    # This captures the specific PATTERN independent of the individual disc criteria.
    if (disc_above is not None and disc_below is not None
            and disc_above.dhi_pct is not None and disc_below.dhi_pct is not None):
        above_dehy      = disc_above.dhi_pct < DHI_MODERATE_PCT   # < 70%
        below_preserved = disc_below.dhi_pct >= DHI_MILD_PCT       # ≥ 80%
        if above_dehy and below_preserved:
            lo_sac, lo_lumb, ds, dl = _apply_lr(lo_sac, lo_lumb,
                                                   'disc_pattern_l4dehy_l5preserved')
            evidence_sac += ds; evidence_lumb += dl; n_fired += 1
            criteria.append(RadiologicCriterion(
                name='disc_asymmetric_pattern',
                value=(f'above DHI={disc_above.dhi_pct:.0f}% (dehy) / '
                       f'below DHI={disc_below.dhi_pct:.0f}% (preserved)'),
                direction='sacralization',
                strength='secondary',
                lr_sac=round(ds, 3), lr_lumb=round(dl, 3),
                citation='Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;1(5):271–280',
                finding=(
                    f'Disc above TV dehydrated (DHI={disc_above.dhi_pct:.0f}%) with '
                    f'disc below TV preserved (DHI={disc_below.dhi_pct:.0f}%) — '
                    f'asymmetric dehydration pattern: strongest combined LSTV predictor '
                    f'(OR 19.87 for non-dehydrated LS disc; Seilanian Toosi 2025)'
                )
            ))

    # ── Convert to probabilities ──────────────────────────────────────────────
    p_sac  = _sigmoid(lo_sac)
    p_lumb = _sigmoid(lo_lumb)
    p_norm = max(0.0, 1.0 - p_sac - p_lumb)
    p_transit = 0.0
    if (has_castellvi and lumbar_count == 5
            and disc_below and disc_below.dhi_pct and disc_below.dhi_pct >= DHI_REDUCED_PCT):
        p_transit = min(0.25, p_sac * 0.35)
    total = p_sac + p_lumb + max(0.001, p_norm)
    p_sac  = round(p_sac  / total, 4)
    p_lumb = round(p_lumb / total, 4)
    p_norm = round(max(0.0, 1.0 - p_sac - p_lumb), 4)
    ranked = sorted([('sacralization', p_sac), ('lumbarization', p_lumb),
                     ('normal', p_norm)], key=lambda x: x[1], reverse=True)

    probs = LSTVProbabilities(
        p_sacralization = p_sac,
        p_lumbarization = p_lumb,
        p_normal        = p_norm,
        p_transitional  = round(p_transit, 4),
        log_odds_sac_evidence  = round(evidence_sac, 3),
        log_odds_lumb_evidence = round(evidence_lumb, 3),
        n_criteria      = n_fired,
        dominant_class  = ranked[0][0],
        confidence_pct  = round(ranked[0][1] * 100, 1),
        calibration_note = (
            f"Bayesian LR model v5. Prior: P(sac)={PRIOR_SACRALIZATION:.0%}, "
            f"P(lumb)={PRIOR_LUMBARIZATION:.0%} (Apazidis 2011). "
            f"{n_fired} criteria evaluated (angles applied as step 8.5). "
            f"Seilanian Toosi 2025 disc-pattern criterion included.")
    )
    return probs, criteria


# ═════════════════════════════════════════════════════════════════════════════
# SURGICAL RELEVANCE  (unchanged from v4)
# ═════════════════════════════════════════════════════════════════════════════

def assess_surgical_relevance(
        lumbar_count:   int,
        tv_name:        Optional[str],
        castellvi_type: Optional[str],
        phenotype:      Optional[str],
        probs:          LSTVProbabilities,
        disc_below:     Optional[DiscMetrics],
        tv_shape:       Optional[TVBodyShape],
) -> SurgicalRelevance:
    sr = SurgicalRelevance()
    has_lstv      = phenotype in ('sacralization', 'lumbarization', 'transitional_indeterminate')
    has_castellvi = bool(castellvi_type and castellvi_type not in ('None', None))
    p_dominant    = max(probs.p_sacralization, probs.p_lumbarization)

    if not has_lstv and not has_castellvi:
        sr.wrong_level_risk     = 'low'
        sr.wrong_level_risk_pct = 0.05
        sr.level_ambiguity_note = ('Normal anatomy. Standard counting reliable. '
                                   'Confirm to sacrum intraoperatively.')
    elif lumbar_count == 4:
        sr.wrong_level_risk     = 'critical'
        sr.wrong_level_risk_pct = round(min(0.90, 0.60 + p_dominant * 0.30), 3)
        sr.level_ambiguity_note = (
            f'CRITICAL: Only {lumbar_count} mobile lumbar vertebrae. '
            f'What reports label "L4" = anatomic L5. '
            f'Mandatory pre-operative spine-wide scout. '
            f'Reference: Tokala 2005; O\'Brien 2019.')
    elif lumbar_count == 6:
        sr.wrong_level_risk     = 'high'
        sr.wrong_level_risk_pct = round(min(0.75, 0.40 + p_dominant * 0.30), 3)
        sr.level_ambiguity_note = (
            f'HIGH RISK: 6 lumbar vertebrae. L6-S1 may be mistaken for L5-S1. '
            f'Reference: Tokala 2005; Hughes & Saifuddin 2006.')
    elif has_castellvi:
        sr.wrong_level_risk     = 'moderate'
        sr.wrong_level_risk_pct = round(min(0.45, 0.18 + p_dominant * 0.25), 3)
        sr.level_ambiguity_note = (
            f'MODERATE: Castellvi {castellvi_type} — partial TP-sacrum fixation. '
            f'Confirm level to sacrum under fluoroscopy.')
    else:
        sr.wrong_level_risk     = 'low-moderate'
        sr.wrong_level_risk_pct = round(min(0.25, 0.10 + p_dominant * 0.15), 3)
        sr.level_ambiguity_note = (
            f'Transitional morphology without Castellvi TP finding. '
            f'Level confirmation from sacrum recommended.')

    if lumbar_count == 6:
        sr.nerve_root_ambiguity = True
        sr.nerve_root_note = ('Lumbarization: root at L6-S1 is functionally L5-equivalent. '
                              'Dermatomal EMG essential when radiculopathy present. '
                              'Reference: Farshad-Amacker 2014; Luoma 2004.')
    elif lumbar_count == 4:
        sr.nerve_root_ambiguity = True
        sr.nerve_root_note = ('"L4" in reports = anatomically L5. '
                              'Root at TV-sacrum junction is functionally L5/S1 equivalent. '
                              'Reference: Tokala 2005; O\'Brien 2019.')
    elif has_castellvi:
        sr.nerve_root_ambiguity = True
        sr.nerve_root_note = (f'Castellvi {castellvi_type}: asymmetric TP-sacrum fixation '
                              f'may cause ipsilateral foraminal compromise. Quinlan 1984.')

    flags = []
    if sr.wrong_level_risk in ('high', 'critical'):
        flags.append('⚠  MANDATORY: Count from S1 upward. Identify sacrum on fluoroscopy BEFORE counting.')
    if has_castellvi:
        bilateral = 'b' in (castellvi_type or '').lower()
        if bilateral:
            flags.append(f'Bilateral Castellvi {castellvi_type}: pseudo-bilateral sacral fixation.')
        else:
            flags.append(f'Unilateral Castellvi {castellvi_type}: asymmetric — assess for coronal scoliosis (Quinlan 1984).')
    if disc_below and disc_below.dhi_pct and disc_below.dhi_pct < DHI_REDUCED_PCT:
        flags.append(f'TV disc severely narrowed (DHI={disc_below.dhi_pct:.0f}%): cage geometry adjustment needed.')
    if has_castellvi and any(x in (castellvi_type or '') for x in ('III', 'IV')):
        flags.append('Castellvi III/IV: CT-guided navigation recommended at TV level.')
    if lumbar_count != EXPECTED_LUMBAR:
        flags.append('INTRAOPERATIVE: Trace to sacrum. Annotate level on image before incision.')
    sr.surgical_flags = flags

    approach = []
    if lumbar_count == 6:
        approach.append('ALIF at L6-S1: confirm true S1 endplate. L6 pedicles are lumbar-sized.')
        approach.append('Posterior: L6 TP is lumbar-type, identifiable bilaterally.')
    elif lumbar_count == 4:
        approach.append('Pedicle screws at TV: transitional pedicle — CT trajectory planning required.')
        approach.append('ALIF: true S1 endplate must be confirmed — promontory may be more cephalad.')
    if has_castellvi and 'II' in (castellvi_type or ''):
        approach.append('Type II pseudo-joint: resection may be required for neural decompression (Luoma 2004).')
    if has_castellvi and 'III' in (castellvi_type or ''):
        approach.append('Type III fusion: osteotomy may be required for lumbosacral junction exposure.')
    sr.approach_considerations = approach

    if has_lstv or lumbar_count != EXPECTED_LUMBAR:
        sr.level_identification_protocol = (
            'PROTOCOL: (1) Full-spine sagittal MRI/CT scout. (2) Identify sacrum from ala morphology. '
            '(3) Count lumbar levels superiorly from S1. (4) Correlate with last rib (T12). '
            '(5) Note mobile segment count in operative report. (6) Mark target on fluoroscopy. '
            '(7) Consider O-arm/CT for complex cases. Reference: O\'Brien 2019; Farshad-Amacker 2014.')
        sr.recommended_counting_method = (
            'Count superiorly from S1. Iliolumbar ligament attaches to L5 equivalent. '
            'Do not rely on imaging label alone (O\'Brien 2019).')
    else:
        sr.recommended_counting_method = 'Standard counting from C2 or last rib is reliable.'

    if sr.nerve_root_ambiguity:
        sr.intraop_neuromonitoring_note = (
            'LSTV nerve root naming discrepancy. Free-running EMG L4/L5/S1 bilaterally. '
            'Triggered EMG during pedicle screw placement at transitional level. '
            'Dermatomal SSEP if deficit localisation uncertain. Reference: MacDonald 2002.')
    else:
        sr.intraop_neuromonitoring_note = 'Standard IONM protocol. No LSTV-specific monitoring concern.'

    bert_crit = []; p_bert = 0.0
    if has_castellvi:
        p_base = {'iii_iv': 0.60, 'ii': 0.50, 'i': 0.30}.get(
            'iii_iv' if any(x in ct for x in ('III','IV')) else 'ii' if 'II' in ct else 'i', 0.30)
        p_bert += p_base
        bert_crit.append(f'Castellvi {castellvi_type}: TP pseudo-joint/fusion (Andrasinova 2018; Quinlan 1984)')
    if disc_below and disc_below.dhi_pct and disc_below.dhi_pct < DHI_REDUCED_PCT:
        p_bert += 0.10
        bert_crit.append(f'Severe TV disc narrowing (DHI={disc_below.dhi_pct:.0f}%)')
    if has_lstv and not has_castellvi:
        p_bert += 0.15
        bert_crit.append('LSTV without Castellvi: mechanical instability at TV level')
    if has_lstv and lumbar_count == EXPECTED_LUMBAR:
        bert_crit.append('LSTV prevalence: 2–3× increased LBP risk (Konin 2010; Luoma 2004)')
    sr.bertolotti_probability = round(min(0.90, p_bert), 3)
    sr.bertolotti_criteria    = bert_crit

    return sr


# ═════════════════════════════════════════════════════════════════════════════
# LSTV PHENOTYPE CLASSIFICATION  (unchanged from v4)
# ═════════════════════════════════════════════════════════════════════════════

def classify_lstv_phenotype(
        lumbar_count:   int,
        tv_name:        Optional[str],
        castellvi_type: Optional[str],
        tv_shape:       Optional[TVBodyShape],
        disc_above:     Optional[DiscMetrics],
        disc_below:     Optional[DiscMetrics],
) -> Tuple[str, str, List[str], str, List[str]]:
    criteria:  List[str] = []
    primary:   List[str] = []
    sac_score  = 0
    lumb_score = 0
    has_castellvi = bool(castellvi_type and castellvi_type not in ('None', None))

    if has_castellvi:
        criteria.append(f"S1 ✓ Castellvi {castellvi_type} (primary)")
        primary.append(f"S1:Castellvi-{castellvi_type}")
        sac_score += 3
        if any(x in castellvi_type for x in ('III','IV')): sac_score += 2
        elif 'II' in castellvi_type: sac_score += 1

    if lumbar_count == 6:
        criteria.append("L1 ✓ 6-lumbar count (primary lumbarization)")
        primary.append("L1:6-lumbar-count"); lumb_score += 5
    elif lumbar_count == 4:
        criteria.append("S3 ✓ 4-lumbar count (primary sacralization)")
        primary.append("S3:4-lumbar-count"); sac_score += 5
    else:
        criteria.append(f"Lumbar count = {lumbar_count} (normal)")

    if tv_name == 'L6':
        criteria.append("L3 ✓ TV = L6 (lumbarization morphology)")
        primary.append("L3:TV-is-L6"); lumb_score += 3
    elif tv_name == 'L5':
        criteria.append("TV = L5 (standard lowest lumbar)")

    if tv_shape and tv_shape.h_ap_ratio:
        ratio_str = f"H/AP={tv_shape.h_ap_ratio:.2f}"
        if tv_shape.shape_class == 'sacral-like':
            criteria.append(f"TV body sacral-like — {ratio_str} < {TV_SHAPE_SACRAL} (Nardo 2012)")
            sac_score += 2
            if not has_castellvi: primary.append("S4:sacral-like-body")
        elif tv_shape.shape_class == 'transitional':
            criteria.append(f"TV body transitional — {ratio_str}"); sac_score += 1
        else:
            criteria.append(f"TV body lumbar-like — {ratio_str}"); lumb_score += 2
        if tv_shape.norm_ratio and tv_shape.norm_ratio < 0.80:
            criteria.append(f"TV/L4 H:AP={tv_shape.norm_ratio:.2f} — TV squarer than L4")
            sac_score += 1

    if disc_below:
        dhi = disc_below.dhi_pct; level = disc_below.level
        if dhi is not None:
            if dhi < DHI_REDUCED_PCT:
                criteria.append(f"S2 ✓ Disc below ({level}) severely reduced — DHI={dhi:.0f}%")
                primary.append(f"S2:disc-below-DHI-{dhi:.0f}pct"); sac_score += 4
            elif dhi < DHI_MODERATE_PCT:
                criteria.append(f"Disc below moderately reduced — DHI={dhi:.0f}%"); sac_score += 2
            elif dhi < DHI_MILD_PCT:
                criteria.append(f"Disc below mildly reduced — DHI={dhi:.0f}%"); sac_score += 1
            else:
                criteria.append(f"L2 ✓ Disc below ({level}) preserved — DHI={dhi:.0f}%")
                primary.append(f"L2:disc-below-preserved-DHI-{dhi:.0f}pct"); lumb_score += 3
        elif disc_below.is_absent:
            criteria.append(f"S2 ✓ Disc below ({disc_below.level}) absent")
            primary.append("S2:disc-below-absent"); sac_score += 3

    if disc_above and disc_above.dhi_pct and disc_above.dhi_pct >= DHI_MILD_PCT:
        criteria.append(f"Disc above ({disc_above.level}) normal — DHI={disc_above.dhi_pct:.0f}%")

    # Decision tree
    if lumbar_count == 6:
        phenotype = 'lumbarization'; confidence = 'high'
        rationale = (f"LUMBARIZATION confirmed: 6 lumbar vertebrae. "
                     f"{'Castellvi ' + castellvi_type + ' co-present.' if has_castellvi else ''}")
        return phenotype, confidence, criteria, rationale, primary

    if lumbar_count == 4:
        phenotype = 'sacralization'; confidence = 'high'
        rationale = (f"SACRALIZATION confirmed: 4 lumbar vertebrae — L5 incorporated into sacrum. "
                     f"{'Castellvi ' + castellvi_type + ' co-present.' if has_castellvi else ''}")
        return phenotype, confidence, criteria, rationale, primary

    disc_below_dhi  = disc_below.dhi_pct if disc_below else None
    disc_below_gone = disc_below.is_absent if disc_below else False
    has_s2 = (disc_below_dhi is not None and disc_below_dhi < DHI_REDUCED_PCT) or disc_below_gone

    if sac_score >= 6 or (has_castellvi and has_s2):
        phenotype  = 'sacralization'
        confidence = 'high' if sac_score >= 8 else 'moderate'
        rationale  = (f"SACRALIZATION: primary criteria: {', '.join(primary) or 'Castellvi+disc'}. "
                      f"sac_score={sac_score}")
    elif sac_score >= 4 and has_castellvi:
        phenotype = 'sacralization'; confidence = 'moderate'
        rationale = f"SACRALIZATION (moderate): Castellvi {castellvi_type} + morphometric support (score={sac_score})"
    elif has_castellvi and not has_s2 and sac_score < 4:
        phenotype = 'transitional_indeterminate'; confidence = 'low'
        rationale = (f"TRANSITIONAL INDETERMINATE: Castellvi {castellvi_type} confirmed but "
                     f"disc below preserved. May be isolated TP anomaly (Quinlan 1984).")
    elif not has_castellvi and sac_score < 4 and lumb_score < 4:
        phenotype = 'normal'; confidence = 'high'
        rationale = "NORMAL: 5 lumbar vertebrae, no Castellvi, preserved disc heights."
    else:
        phenotype = 'normal'; confidence = 'moderate'
        rationale = f"No primary LSTV criteria met (count=5, sac_score={sac_score}, lumb_score={lumb_score})."

    return phenotype, confidence, criteria, rationale, primary


# ═════════════════════════════════════════════════════════════════════════════
# FALLBACK SURGICAL RELEVANCE  (unchanged from v4)
# ═════════════════════════════════════════════════════════════════════════════

def _fallback_surgical_relevance(
        lumbar_count:   int,
        castellvi_type: Optional[str],
        phenotype:      Optional[str],
        disc_below:     Optional['DiscMetrics'],
) -> SurgicalRelevance:
    sr = SurgicalRelevance()
    has_castellvi = bool(castellvi_type and castellvi_type not in ('None', None))
    if lumbar_count == 4:
        sr.wrong_level_risk = 'critical'; sr.wrong_level_risk_pct = 0.75
    elif lumbar_count == 6:
        sr.wrong_level_risk = 'high'; sr.wrong_level_risk_pct = 0.55
    elif has_castellvi:
        sr.wrong_level_risk = 'moderate'; sr.wrong_level_risk_pct = 0.30
    elif phenotype in ('sacralization', 'lumbarization'):
        sr.wrong_level_risk = 'low-moderate'; sr.wrong_level_risk_pct = 0.15
    else:
        sr.wrong_level_risk = 'low'; sr.wrong_level_risk_pct = 0.05
    sr.nerve_root_ambiguity = (lumbar_count != EXPECTED_LUMBAR or has_castellvi)
    p_bert = 0.0
    if has_castellvi:
        ct = castellvi_type or ''
        p_bert = (0.60 if any(x in ct for x in ('III', 'IV')) else 0.50 if 'II' in ct else 0.30)
    elif phenotype in ('sacralization', 'lumbarization'):
        p_bert = 0.15
    if (disc_below and hasattr(disc_below, 'dhi_pct') and disc_below.dhi_pct is not None
            and disc_below.dhi_pct < DHI_REDUCED_PCT):
        p_bert = min(0.90, p_bert + 0.10)
    sr.bertolotti_probability = round(p_bert, 3)
    sr.calibration_note = 'fallback — full surgical relevance computation unavailable'
    sr.recommended_counting_method = (
        'Count superiorly from S1. Do not rely on imaging labels alone.'
        if sr.wrong_level_risk not in ('low',) else 'Standard counting reliable.')
    return sr


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT  (v5: adds step 8.5 vertebral angles)
# ═════════════════════════════════════════════════════════════════════════════

def _afmt(v: Optional[float]) -> str:
    return f'{v:.1f}' if v is not None else 'N/A'


def analyze_lstv(masks: LSTVMaskSet,
                 castellvi_result: Optional[dict] = None) -> LSTVMorphometrics:
    """
    Run complete LSTV morphometric analysis (v5).

    Additions vs v4:
    - Step 8.5: Vertebral angle analysis (Seilanian Toosi 2025)
      delta-angle ≤ 8.5° → Type 2 LSTV (sens 92.3%, spec 87.9%)
    - Disc asymmetry pattern in Bayesian model
    """
    result = LSTVMorphometrics(study_id=masks.study_id)

    try:
        # 1. Lumbar count
        tss_safe             = masks.tss_iso if masks.tss_iso is not None else np.array([], dtype=np.int32)
        tss_count, tss_names = count_lumbar_tss(tss_safe, masks.tss_labels)
        vd_count,  vd_names  = count_lumbar_veridah(masks.vert_labels)
        consensus, count_note = reconcile_lumbar_count(
            tss_count, vd_count, tss_names, vd_names, masks.vert_labels)

        result.lumbar_count_tss       = tss_count  if tss_count > 0 else None
        result.lumbar_count_veridah   = vd_count   if vd_count  > 0 else None
        result.lumbar_count_consensus = consensus
        result.lumbar_count_anomaly   = (consensus != EXPECTED_LUMBAR)
        result.lumbar_count_note      = count_note

        # 2. TV identification
        tv_label, tv_name = None, None
        for cand in VERIDAH_TV_SEARCH:
            if cand in masks.vert_labels:
                tv_label = cand; tv_name = VERIDAH_NAMES[cand]; break

        if tv_label is None:
            result.error = "No lumbar VERIDAH labels found"; return result

        result.tv_label_veridah = tv_label
        result.tv_name          = tv_name
        result.has_l6           = (tv_label == VD_L6)
        result.tv_tss_label     = VD_TO_TSS_VERT.get(tv_label)

        # 3. TV body shape
        result.tv_shape = analyze_tv_body_shape(masks, tv_label, result.tv_tss_label)

        # 4. Adjacent disc metrics
        result.disc_above, result.disc_below = get_tv_adjacent_discs(
            masks, tv_label, result.tv_tss_label)

        # 5. Relative disc ratio
        result.relative_disc_ratio, result.relative_disc_note = \
            compute_relative_disc_ratio(result.disc_above, result.disc_below)

        # 6. Rib anomaly
        result.rib_anomaly = detect_rib_anomaly(masks)

        # 7. Discrete phenotype classification
        castellvi_type = None
        if castellvi_result:
            castellvi_type = castellvi_result.get('castellvi_type')

        (result.lstv_phenotype,
         result.phenotype_confidence,
         result.phenotype_criteria,
         result.phenotype_rationale,
         result.primary_criteria_met) = classify_lstv_phenotype(
            lumbar_count   = consensus,
            tv_name        = tv_name,
            castellvi_type = castellvi_type,
            tv_shape       = result.tv_shape,
            disc_above     = result.disc_above,
            disc_below     = result.disc_below,
        )

        # 8. Bayesian probability model
        try:
            result.probabilities, result.radiologic_evidence = compute_lstv_probability(
                lumbar_count   = consensus,
                castellvi_type = castellvi_type,
                tv_shape       = result.tv_shape,
                disc_above     = result.disc_above,
                disc_below     = result.disc_below,
                rel_disc_ratio = result.relative_disc_ratio,
            )
        except Exception as exc:
            import traceback as _tb
            logger.error(f"  [{masks.study_id}] STEP 8 (probability model) FAILED: {exc}")
            logger.error(_tb.format_exc())

        # ────────────────────────────────────────────────────────────────────
        # 8.5. Vertebral angle analysis (NEW v5 — Seilanian Toosi et al. 2025)
        # ────────────────────────────────────────────────────────────────────
        try:
            from lstv_angles import compute_vertebral_angles, apply_angle_lr_updates
            angle_result = compute_vertebral_angles(
                sp_iso           = masks.sp_iso,
                vert_iso         = masks.vert_iso,
                tss_iso          = masks.tss_iso,
                tv_veridah_label = tv_label,
                vox_mm           = ISO_MM,
                sp_corpus_label  = SP_CORPUS,
            )
            result.vertebral_angles = angle_result

            # Update Bayesian probabilities with angle LR evidence
            if (angle_result.angles_available
                    and result.probabilities is not None
                    and angle_result.angle_lr_keys_fired):
                lo_sac_cur  = _log_odds(result.probabilities.p_sacralization)
                lo_lumb_cur = _log_odds(result.probabilities.p_lumbarization)
                lo_sac_upd, lo_lumb_upd, angle_crit = apply_angle_lr_updates(
                    lo_sac_cur, lo_lumb_cur, angle_result,
                    result.radiologic_evidence or [])

                if lo_sac_upd != lo_sac_cur or lo_lumb_upd != lo_lumb_cur:
                    p_sac_new  = _sigmoid(lo_sac_upd)
                    p_lumb_new = _sigmoid(lo_lumb_upd)
                    p_norm_new = max(0.0, 1.0 - p_sac_new - p_lumb_new)
                    total_new  = p_sac_new + p_lumb_new + max(0.001, p_norm_new)
                    result.probabilities.p_sacralization = round(p_sac_new / total_new, 4)
                    result.probabilities.p_lumbarization = round(p_lumb_new / total_new, 4)
                    result.probabilities.p_normal        = round(
                        max(0.0, 1.0 - result.probabilities.p_sacralization
                            - result.probabilities.p_lumbarization), 4)
                    result.probabilities.n_criteria     += len(angle_result.angle_lr_keys_fired)
                    if result.radiologic_evidence is None:
                        result.radiologic_evidence = []
                    result.radiologic_evidence.extend(angle_crit)

            if angle_result.angles_available:
                logger.info(
                    f"  [{masks.study_id}] Angles (Seilanian Toosi 2025): "
                    f"A={_afmt(angle_result.a_angle_deg)}°  "
                    f"C={_afmt(angle_result.c_angle_deg)}°  "
                    f"D={_afmt(angle_result.d_angle_deg)}°  "
                    f"delta={_afmt(angle_result.delta_angle_deg)}°  "
                    f"delta_positive={angle_result.delta_positive}  "
                    f"c_positive={angle_result.c_positive}"
                )
                if angle_result.delta_positive:
                    logger.warning(
                        f"  [{masks.study_id}] ⚠ ANGLE CRITERION: "
                        f"delta={_afmt(angle_result.delta_angle_deg)}° ≤ 8.5° — "
                        f"Type 2 LSTV signal (sens 92.3%, spec 87.9%; Seilanian Toosi 2025)")
                if angle_result.c_positive:
                    logger.warning(
                        f"  [{masks.study_id}] ⚠ ANGLE CRITERION: "
                        f"C={_afmt(angle_result.c_angle_deg)}° ≤ 35.5° — "
                        f"LSTV signal (sens 72.2%, spec 57.6%; Seilanian Toosi 2025)")
            else:
                logger.info(f"  [{masks.study_id}] Angles: not computed (insufficient mask data)")

        except ImportError:
            logger.warning(f"  [{masks.study_id}] lstv_angles.py not found — angle analysis skipped")
        except Exception as exc:
            import traceback as _tb
            logger.error(f"  [{masks.study_id}] STEP 8.5 (vertebral angles) FAILED: {exc}")
            logger.debug(_tb.format_exc())

        # 9. Surgical relevance
        try:
            if result.probabilities is not None:
                result.surgical_relevance = assess_surgical_relevance(
                    lumbar_count   = consensus,
                    tv_name        = tv_name,
                    castellvi_type = castellvi_type,
                    phenotype      = result.lstv_phenotype,
                    probs          = result.probabilities,
                    disc_below     = result.disc_below,
                    tv_shape       = result.tv_shape,
                )
            else:
                raise ValueError("probabilities unavailable")
        except Exception as exc:
            import traceback as _tb
            logger.error(f"  [{masks.study_id}] STEP 9 (surgical relevance) FAILED: {exc}")
            logger.debug(_tb.format_exc())
            logger.warning(f"  [{masks.study_id}] Using fallback surgical relevance")
            result.surgical_relevance = _fallback_surgical_relevance(
                lumbar_count   = consensus,
                castellvi_type = castellvi_type,
                phenotype      = result.lstv_phenotype,
                disc_below     = result.disc_below,
            )

        p_sac  = result.probabilities.p_sacralization if result.probabilities else 0.0
        p_lumb = result.probabilities.p_lumbarization if result.probabilities else 0.0
        va = result.vertebral_angles
        angle_str = (f"  delta={_afmt(va.delta_angle_deg)}°({'⚠' if va.delta_positive else '✓'})"
                     if va and va.angles_available else '')
        logger.info(
            f"  [{masks.study_id}] LSTV morphometrics v5: "
            f"TV={tv_name}, count={consensus}, "
            f"phenotype={result.lstv_phenotype} ({result.phenotype_confidence}), "
            f"P(sac)={p_sac:.1%}, P(lumb)={p_lumb:.1%}, "
            f"surgical_risk={result.surgical_relevance.wrong_level_risk}"
            f"{angle_str}"
        )

    except Exception as exc:
        import traceback as _tb
        result.error = str(exc)
        logger.error(f"  [{masks.study_id}] lstv_engine FATAL error: {exc}")
        logger.error(_tb.format_exc())
        if result.surgical_relevance is None:
            result.surgical_relevance = _fallback_surgical_relevance(
                lumbar_count   = result.lumbar_count_consensus or EXPECTED_LUMBAR,
                castellvi_type = (castellvi_result.get('castellvi_type')
                                  if castellvi_result else None),
                phenotype      = result.lstv_phenotype,
                disc_below     = result.disc_below,
            )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# PATHOLOGY SCORING  (v5: adds angle criteria score)
# ═════════════════════════════════════════════════════════════════════════════

def compute_lstv_pathology_score(detect_result: dict,
                                  morpho_result: Optional[dict] = None) -> float:
    """
    Scalar LSTV pathology burden score (v5).

    v5 additions:
    - delta-angle ≤ 8.5°  (Type 2): +3.0
    - delta-angle ≤ 14.5° (any LSTV): +1.5
    - C-angle ≤ 35.5°:    +1.5
    - A-angle elevated:    +0.5

    Castellvi:           IV=5  III=4  II=3  I=1
    Phenotype (high):    +3;  moderate: +2;  transitional: +1
    Probability boost:   P>0.85→+2;  P>0.70→+1
    Lumbar count anomaly: +2
    Disc below DHI<50%:  +2  | <70%: +1
    Relative disc ratio < 0.65: +1
    TV body sacral-like: +2  | transitional: +1
    Rib anomaly: +1
    """
    score = 0.0

    ct = detect_result.get('castellvi_type') or ''
    if   'IV'  in ct: score += 5
    elif 'III' in ct: score += 4
    elif 'II'  in ct: score += 3
    elif 'I'   in ct: score += 1

    if morpho_result is None:
        return score

    ph  = morpho_result.get('lstv_phenotype', 'normal')
    cnf = morpho_result.get('phenotype_confidence', 'low')
    if ph in ('sacralization', 'lumbarization'):
        score += 3.0 if cnf == 'high' else 2.0
    elif ph == 'transitional_indeterminate':
        score += 1.0

    probs = morpho_result.get('probabilities') or {}
    p_dom = max(probs.get('p_sacralization', 0), probs.get('p_lumbarization', 0))
    if p_dom > 0.85:   score += 2.0
    elif p_dom > 0.70: score += 1.0

    cnt = morpho_result.get('lumbar_count_consensus', 5)
    if cnt and cnt != EXPECTED_LUMBAR:
        score += 2.0

    db  = morpho_result.get('disc_below') or {}
    dhi = db.get('dhi_pct')
    if dhi is not None:
        if   dhi < DHI_REDUCED_PCT:  score += 2.0
        elif dhi < DHI_MODERATE_PCT: score += 1.0
    if db.get('is_absent'): score += 2.0

    rdr = morpho_result.get('relative_disc_ratio')
    if rdr is not None and rdr < 0.65:
        score += 1.0

    sh  = morpho_result.get('tv_shape') or {}
    shc = sh.get('shape_class', '')
    if   shc == 'sacral-like':  score += 2.0
    elif shc == 'transitional': score += 1.0

    rib = morpho_result.get('rib_anomaly') or {}
    if rib.get('any_anomaly'): score += 1.0

    # v5: Vertebral angle criteria (Seilanian Toosi 2025)
    angles = morpho_result.get('vertebral_angles') or {}
    if angles.get('delta_positive'):    score += 3.0   # sens 92.3% for Type 2
    elif angles.get('delta_any_lstv'):  score += 1.5   # any LSTV
    if angles.get('c_positive'):        score += 1.5   # C-angle criterion
    if angles.get('a_angle_elevated'):  score += 0.5   # A-angle predictor

    return score
