#!/usr/bin/env python3
"""
morphometrics_engine.py  —  Modular Spine Morphometrics Engine
==============================================================
All measurement functions are self-contained and importable.
No rendering dependencies (no plotly, no skimage.measure).

PUBLIC API
──────────
  load_study_masks(study_id, spineps_dir, totalspine_dir)
      → MaskSet (dataclass)
  run_all_morphometrics(masks: MaskSet)
      → MorphometricResult (dict-like with .to_dict())
  classify_stenosis(ap_mm, dsca_mm2)      → (ap_class, dsca_class)
  cord_compression_profile(cord, canal, iso_mm)
      → list[CordSlice]          ← full-length per-slice cord compression

COORDINATE CONVENTIONS
  All volumes are resampled to ISO_MM = 1.0 mm³ before measurement.
  Axes: X=mediolateral  Y=anteroposterior  Z=superoinferior (cranial→caudal)
  RAS canonical orientation enforced on load.

LABEL REFERENCE (see 04_detect_lstv.py header for full table)
  SPINEPS seg-spine_msk: 26=Sacrum 41=Arcus 42=Spinous 43=TP-L 44=TP-R
      45=SupArtL 46=SupArtR 47=InfArtL 48=InfArtR 49=CorpusBorder
      60=Cord 61=Canal 62=Endplate(merged) 100=IVD(merged)
  VERIDAH seg-vert_msk: 20=L1..25=L6 26=Sacrum 100+X=IVD-X 200+X=EP-X
  TotalSpineSeg: 1=cord 2=canal 11-45=C1-L5 50=Sacrum
      91=discT12L1 92-95=discL1L2..L4L5 100=discL5S
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, label as cc_label
from scipy.ndimage import zoom as ndizoom

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

ISO_MM = 1.0          # isotropic voxel target (mm)
TP_HEIGHT_MM    = 19.0
CONTACT_DIST_MM = 2.0

# SPINEPS seg-spine_msk labels
SP_SACRUM   = 26; SP_ARCUS = 41; SP_SPINOUS = 42
SP_TP_L = 43; SP_TP_R = 44
SP_SAL = 45; SP_SAR = 46; SP_IAL = 47; SP_IAR = 48
SP_CORPUS = 49; SP_CORD = 60; SP_CANAL = 61
SP_ENDPLATE = 62; SP_IVD = 100

# VERIDAH seg-vert_msk
VD_L1=20; VD_L2=21; VD_L3=22; VD_L4=23; VD_L5=24; VD_L6=25; VD_SAC=26
VD_IVD_BASE = 100; VD_EP_BASE = 200
VERIDAH_LUMBAR_LABELS = [VD_L6, VD_L5, VD_L4, VD_L3, VD_L2, VD_L1]
VERIDAH_NAMES = {20:'L1',21:'L2',22:'L3',23:'L4',24:'L5',25:'L6',26:'Sacrum'}

# TotalSpineSeg labels
TSS_CORD = 1; TSS_CANAL = 2
TSS_SACRUM = 50
TSS_LUMBAR = {41:'L1',42:'L2',43:'L3',44:'L4',45:'L5'}
TSS_DISCS = {91:'T12-L1',92:'L1-L2',93:'L2-L3',94:'L3-L4',95:'L4-L5',100:'L5-S1'}

# Lumbar level pairs (VERIDAH labels, display names)
LUMBAR_PAIRS = [
    (VD_L1, VD_L2, 'L1','L2'),
    (VD_L2, VD_L3, 'L2','L3'),
    (VD_L3, VD_L4, 'L3','L4'),
    (VD_L4, VD_L5, 'L4','L5'),
    (VD_L5, VD_SAC,'L5','S1'),
]
TSS_DISC_MAP = {'L1_L2':92,'L2_L3':93,'L3_L4':94,'L4_L5':95,'L5_S1':100}

# Canal shape by level (from literature)
CANAL_SHAPE = {
    'L1':('Oval','85-95%'),'L2':('Oval','90%'),
    'L3':('Triangular','80-95%'),'L4':('Triangular','95%'),
    'L5':('Trefoil','60-65%'),
}

# ─── Clinical thresholds (all in one place for easy audit) ────────────────────

class T:
    """Threshold namespace — single source of truth."""
    # Vertebral body
    COMPRESSION_BICONCAVE = 0.80   # Hm/Ha or Hm/Hp
    WEDGE_FRACTURE        = 0.80   # Ha/Hp
    CRUSH_POSTERIOR       = 0.80   # Hp/Ha
    HEIGHT_INTERVENTION   = 0.75   # <0.75 → moderate/severe
    SPONDYLO_MM           = 3.0    # sagittal translation

    # Disc
    DHI_SEVERE_PCT        = 50.0
    DHI_MODERATE_PCT      = 70.0
    DHI_MILD_PCT          = 85.0

    # Canal
    DSCA_NORMAL_MM2       = 100.0
    DSCA_RELATIVE_MM2     = 75.0
    DSCA_ABSOLUTE_MM2     = 70.0
    AP_NORMAL_MM          = 12.0
    AP_RELATIVE_MM        = 10.0
    AP_ABSOLUTE_MM        = 7.0

    # Lateral recess
    RECESS_DEPTH_MM       = 3.0
    RECESS_HEIGHT_MM      = 2.0

    # Cord compression (MSCC proxy)
    MSCC_MILD             = 0.50   # cord AP / canal AP ratio
    MSCC_MODERATE         = 0.67
    MSCC_SEVERE           = 0.80

    # Ligamentum flavum
    LFT_NORMAL_MM         = 3.5
    LFT_HYPERTROPHY_MM    = 4.0
    LFT_SEVERE_MM         = 5.0
    LFA_CUTOFF_MM2        = 105.90

    # Baastrup (kissing spine)
    BAASTRUP_CONTACT_MM   = 0.0
    BAASTRUP_RISK_MM      = 2.0

    # Facet tropism
    TROPISM_NORMAL_DEG    = 7.0
    TROPISM_SEVERE_DEG    = 10.0

    # Foraminal volume norms (mm³, elliptical cylinder, right side)
    FORAMEN_NORMS = {
        'L1_L2':{'R':579.92,'L':594.43,'sd_R':55,'sd_L':44},
        'L2_L3':{'R':688.22,'L':715.87,'sd_R':55,'sd_L':48},
        'L3_L4':{'R':761.70,'L':790.30,'sd_R':59,'sd_L':50},
        'L4_L5':{'R':787.82,'L':809.61,'sd_R':29,'sd_L':57},
        'L5_S1':{'R':824.24,'L':None,  'sd_R':68,'sd_L':None},
    }

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class MaskSet:
    """All resampled iso masks for one study."""
    study_id:   str
    sp_iso:     np.ndarray          # SPINEPS seg-spine_msk  at 1mm
    vert_iso:   np.ndarray          # VERIDAH seg-vert_msk   at 1mm
    tss_iso:    Optional[np.ndarray]# TotalSpineSeg sagittal at 1mm  (may be None)
    sp_labels:  set
    vert_labels:set
    tss_labels: set

@dataclass
class CordSlice:
    """Per-Z-slice cord compression measurement."""
    z_mm:       float
    cord_ap_mm: float
    canal_ap_mm:float
    mscc:       float               # cord_ap / canal_ap
    classification: str             # Normal / Mild / Moderate / Severe
    flagged:    bool

@dataclass
class CordCompressionProfile:
    """Full-length cord compression profile."""
    slices:     List[CordSlice]
    max_mscc:   float
    max_mscc_z_mm: float
    flagged_z_mm:  List[float]      # Z positions with MSCC > MODERATE threshold
    classification: str             # worst-case label

@dataclass
class LevelMetrics:
    """Morphometrics for a single inter-vertebral level."""
    level:      str                 # e.g. 'L4_L5'
    level_display: str              # e.g. 'L4-L5'

    # Disc
    disc_source:    Optional[str] = None
    dhi_pct:        Optional[float] = None
    dhi_grade:      Optional[str]  = None
    dhi_method2:    Optional[float] = None
    endplate_dist_mm: Optional[float] = None
    endplate_source:  Optional[str]  = None

    # Canal at this level
    level_ap_mm:    Optional[float] = None
    level_dsca_mm2: Optional[float] = None
    level_ap_class: Optional[str]  = None
    level_dsca_class: Optional[str] = None
    canal_shape:    Optional[str]  = None

    # Vertebral body (upper vertebra)
    ha_mm: Optional[float] = None
    hm_mm: Optional[float] = None
    hp_mm: Optional[float] = None
    compression_hm_ha: Optional[float] = None
    compression_hm_hp: Optional[float] = None
    wedge_ha_hp:       Optional[float] = None
    crush_hp_ha:       Optional[float] = None
    genant_grade:      Optional[int]   = None
    genant_label:      Optional[str]   = None

    # Spondylolisthesis
    sagittal_translation_mm: Optional[float] = None
    spondylolisthesis:       Optional[str]   = None

    # Foraminal
    foraminal_vol_L_mm3:  Optional[float] = None
    foraminal_vol_R_mm3:  Optional[float] = None
    foraminal_class_L:    Optional[str]   = None
    foraminal_class_R:    Optional[str]   = None
    foraminal_norm_pct_L: Optional[float] = None
    foraminal_norm_pct_R: Optional[float] = None

@dataclass
class MorphometricResult:
    """Complete morphometric result for one study."""
    study_id: str
    error:    Optional[str] = None

    # Global canal
    canal_source:           Optional[str]   = None
    canal_ap_mm:            Optional[float] = None
    canal_dsca_mm2:         Optional[float] = None
    canal_ap_class:         Optional[str]   = None
    canal_dsca_class:       Optional[str]   = None
    canal_absolute_stenosis:bool            = False

    # Cord (global summary)
    cord_source:            Optional[str]   = None
    cord_ap_mm:             Optional[float] = None
    cord_ml_mm:             Optional[float] = None
    cord_csa_mm2:           Optional[float] = None
    canal_csa_mm2:          Optional[float] = None
    mscc_proxy:             Optional[float] = None
    canal_occupation_ratio: Optional[float] = None

    # Full-length cord compression profile
    cord_compression_profile: Optional[dict] = None   # serialised CordCompressionProfile

    # Ligamentum flavum
    lft_proxy_mm:   Optional[float] = None
    lft_class:      Optional[str]   = None

    # Baastrup
    spinous_count:              Optional[int]   = None
    min_inter_process_gap_mm:   Optional[float] = None
    inter_process_gaps_mm:      Optional[List[float]] = None
    baastrup_contact:           bool = False
    baastrup_risk:              bool = False

    # Facet tropism
    facet_angle_l_deg: Optional[float] = None
    facet_angle_r_deg: Optional[float] = None
    facet_tropism_deg: Optional[float] = None
    facet_tropism_grade: Optional[str] = None

    # Per-level
    levels: List[LevelMetrics] = field(default_factory=list)

    # Flat dict of all level metrics for CSV export (populated by to_dict())
    _flat: Dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ('levels','_flat','cord_compression_profile')}
        # Flatten level metrics
        for lm in self.levels:
            prefix = lm.level
            for k, v in asdict(lm).items():
                if k not in ('level','level_display'):
                    d[f'{prefix}_{k}'] = v
        # Cord compression profile summary
        if self.cord_compression_profile:
            cp = self.cord_compression_profile
            d['cord_max_mscc']       = cp.get('max_mscc')
            d['cord_max_mscc_z_mm']  = cp.get('max_mscc_z_mm')
            d['cord_compression_cls']= cp.get('classification')
            d['cord_flagged_z_count']= len(cp.get('flagged_z_mm', []))
        return d

# ─── NIfTI helpers ────────────────────────────────────────────────────────────

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

def _resample_labels(vol: np.ndarray, vox_mm: np.ndarray,
                     target: float = ISO_MM) -> np.ndarray:
    factors = (vox_mm / target).tolist()
    return ndizoom(vol.astype(np.int32), factors,
                   order=0, mode='nearest', prefilter=False).astype(np.int32)

# ─── Mask loading ─────────────────────────────────────────────────────────────

def load_study_masks(study_id: str, spineps_dir: Path,
                     totalspine_dir: Path) -> MaskSet:
    """Load and resample all required mask volumes for one study."""
    seg_dir = spineps_dir / 'segmentations' / study_id
    sp_path   = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path  = (totalspine_dir / study_id / 'sagittal'
                 / f"{study_id}_sagittal_labeled.nii.gz")

    sp_raw, sp_nii = _load_canonical(sp_path)
    vox_mm = _voxel_mm(sp_nii)
    vert_raw, _ = _load_canonical(vert_path)

    sp_iso   = _resample_labels(sp_raw.astype(np.int32),   vox_mm)
    vert_iso = _resample_labels(vert_raw.astype(np.int32), vox_mm)

    tss_iso = None
    if tss_path.exists():
        try:
            tss_raw, tss_nii = _load_canonical(tss_path)
            tss_vox = _voxel_mm(tss_nii)
            tss_iso = _resample_labels(tss_raw.astype(np.int32), tss_vox)
        except Exception as e:
            logger.warning(f"[{study_id}] TSS load failed: {e}")

    return MaskSet(
        study_id   = study_id,
        sp_iso     = sp_iso,
        vert_iso   = vert_iso,
        tss_iso    = tss_iso,
        sp_labels  = set(np.unique(sp_iso).tolist())   - {0},
        vert_labels= set(np.unique(vert_iso).tolist()) - {0},
        tss_labels = (set(np.unique(tss_iso).tolist()) - {0}
                      if tss_iso is not None else set()),
    )

# ─── Geometry primitives ──────────────────────────────────────────────────────

def centroid_mm(mask: np.ndarray) -> Optional[np.ndarray]:
    c = np.array(np.where(mask))
    return c.mean(axis=1) * ISO_MM if c.size else None

def get_z_range(mask: np.ndarray) -> Optional[Tuple[int,int]]:
    if not mask.any(): return None
    zc = np.where(mask)[2]
    return int(zc.min()), int(zc.max())

def min_dist_3d(a: np.ndarray, b: np.ndarray
                ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    from scipy.ndimage import distance_transform_edt
    if not a.any() or not b.any():
        return float('inf'), None, None
    dt    = distance_transform_edt(~b) * ISO_MM
    dist_at = np.where(a, dt, np.inf)
    flat  = int(np.argmin(dist_at))
    va    = np.array(np.unravel_index(flat, a.shape), dtype=float)
    dist  = float(dt[tuple(va.astype(int))])
    cb    = np.array(np.where(b), dtype=float)
    d2    = ((cb.T - va)**2).sum(axis=1)
    vb    = cb[:, int(np.argmin(d2))]
    return dist, va * ISO_MM, vb * ISO_MM

def _ap_ml_from_mask(mask_slice: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """AP (Y-extent) and ML (Z-extent) of a 3D mask sampled at mid-X."""
    if mask_slice is None or not mask_slice.any():
        return None, None
    coords = np.array(np.where(mask_slice))
    xmid   = int(coords[0].mean())
    slab   = mask_slice[max(0,xmid-2):xmid+3, :, :]
    if not slab.any():
        slab = mask_slice
    yc = np.where(slab)[1]; zc = np.where(slab)[2]
    ap = (int(yc.max()) - int(yc.min()) + 1) * ISO_MM
    ml = (int(zc.max()) - int(zc.min()) + 1) * ISO_MM
    return ap, ml

# ─── Canal / stenosis ─────────────────────────────────────────────────────────

def classify_stenosis(ap_mm: Optional[float],
                      dsca_mm2: Optional[float]) -> Tuple[str, str]:
    ap_cls = dsca_cls = 'N/A'
    if ap_mm is not None:
        if ap_mm > T.AP_NORMAL_MM:
            ap_cls = 'Normal'
        elif ap_mm >= T.AP_RELATIVE_MM:
            ap_cls = 'Relative Stenosis'
        elif ap_mm >= T.AP_ABSOLUTE_MM:
            ap_cls = 'Absolute Stenosis'
        else:
            ap_cls = 'Critical Stenosis'
    if dsca_mm2 is not None:
        if dsca_mm2 > T.DSCA_NORMAL_MM2:
            dsca_cls = 'Normal'
        elif dsca_mm2 >= T.DSCA_RELATIVE_MM2:
            dsca_cls = 'Relative Stenosis'
        else:
            dsca_cls = 'Absolute Stenosis'
    return ap_cls, dsca_cls

def _canal_ap_dsca(canal_mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    ap, ml = _ap_ml_from_mask(canal_mask)
    if ap and ml:
        return ap, (np.pi / 4.0) * ap * ml
    return None, None

def canal_metrics_global(masks: MaskSet) -> dict:
    tss_canal = (masks.tss_iso == TSS_CANAL) if (TSS_CANAL in masks.tss_labels) else None
    sp_canal  = (masks.sp_iso  == SP_CANAL)  if (SP_CANAL  in masks.sp_labels)  else None
    active    = tss_canal if (tss_canal is not None and tss_canal.any()) else sp_canal
    source    = 'TSS' if (tss_canal is not None and tss_canal.any()) else 'SPINEPS'
    if active is None or not active.any():
        return {'canal_source': source}
    ap, dsca = _canal_ap_dsca(active)
    ap_cls, dsca_cls = classify_stenosis(ap, dsca)
    return {
        'canal_source': source,
        'canal_ap_mm':  ap, 'canal_dsca_mm2': dsca,
        'canal_ap_class': ap_cls, 'canal_dsca_class': dsca_cls,
        'canal_absolute_stenosis': (
            (ap is not None and ap < T.AP_ABSOLUTE_MM) or
            (dsca is not None and dsca < T.DSCA_ABSOLUTE_MM2)
        ),
    }

# ─── Full-length cord compression profile ────────────────────────────────────

def cord_compression_profile(cord_mask: np.ndarray,
                              canal_mask: np.ndarray,
                              step_z: int = 3) -> Optional[CordCompressionProfile]:
    """
    Per-slice cord/canal ratio along the full extent of the cord.

    Samples every `step_z` voxels along Z (superior→inferior).
    MSCC proxy = cord_AP / canal_AP at each slice.
    Classification per slice:
      <0.50 → Normal   0.50–0.67 → Mild   0.67–0.80 → Moderate   ≥0.80 → Severe

    Returns CordCompressionProfile or None if masks absent.
    """
    if cord_mask is None or not cord_mask.any():
        return None
    if canal_mask is None or not canal_mask.any():
        return None

    cord_z = np.where(cord_mask)[2]
    z_lo, z_hi = int(cord_z.min()), int(cord_z.max())
    slices: List[CordSlice] = []

    for z in range(z_lo, z_hi + 1, step_z):
        cord_sl  = cord_mask[:, :, z]
        canal_sl = canal_mask[:, :, z]
        if not cord_sl.any() or not canal_sl.any():
            continue

        cord_y  = np.where(cord_sl)[0]     # rows=Y in 2D XY slice
        canal_y = np.where(canal_sl)[0]
        cord_ap  = (int(cord_y.max())  - int(cord_y.min())  + 1) * ISO_MM
        canal_ap = (int(canal_y.max()) - int(canal_y.min()) + 1) * ISO_MM

        if canal_ap < 1.0:
            continue

        mscc = cord_ap / canal_ap

        if mscc >= T.MSCC_SEVERE:
            cls = 'Severe'
        elif mscc >= T.MSCC_MODERATE:
            cls = 'Moderate'
        elif mscc >= T.MSCC_MILD:
            cls = 'Mild'
        else:
            cls = 'Normal'

        slices.append(CordSlice(
            z_mm       = z * ISO_MM,
            cord_ap_mm = cord_ap,
            canal_ap_mm= canal_ap,
            mscc       = mscc,
            classification = cls,
            flagged    = mscc >= T.MSCC_MODERATE,
        ))

    if not slices:
        return None

    max_s     = max(slices, key=lambda s: s.mscc)
    flagged_z = [s.z_mm for s in slices if s.flagged]
    worst_cls = max_s.classification

    return CordCompressionProfile(
        slices         = slices,
        max_mscc       = max_s.mscc,
        max_mscc_z_mm  = max_s.z_mm,
        flagged_z_mm   = flagged_z,
        classification = worst_cls,
    )

def cord_metrics_global(masks: MaskSet) -> dict:
    tss_cord  = (masks.tss_iso == TSS_CORD)   if (TSS_CORD  in masks.tss_labels) else None
    tss_canal = (masks.tss_iso == TSS_CANAL)  if (TSS_CANAL in masks.tss_labels) else None
    sp_cord   = (masks.sp_iso  == SP_CORD)    if (SP_CORD   in masks.sp_labels)  else None
    sp_canal  = (masks.sp_iso  == SP_CANAL)   if (SP_CANAL  in masks.sp_labels)  else None

    active_cord  = tss_cord  if (tss_cord  is not None and tss_cord.any())  else sp_cord
    active_canal = tss_canal if (tss_canal is not None and tss_canal.any()) else sp_canal

    result: dict = {}
    if active_cord is not None and active_cord.any():
        ap, ml = _ap_ml_from_mask(active_cord)
        if ap and ml:
            result.update({
                'cord_source': 'TSS' if (tss_cord is not None) else 'SPINEPS',
                'cord_ap_mm': ap,
                'cord_ml_mm': ml,
                'cord_csa_mm2': (np.pi/4.0)*ap*ml,
            })

    if active_canal is not None and active_canal.any():
        cap, cml = _ap_ml_from_mask(active_canal)
        if cap and cml:
            result['canal_csa_mm2'] = (np.pi/4.0)*cap*cml
            if 'cord_ap_mm' in result:
                result['mscc_proxy'] = result['cord_ap_mm'] / cap
            if 'cord_csa_mm2' in result:
                result['canal_occupation_ratio'] = result['cord_csa_mm2'] / result['canal_csa_mm2']

    # Full-length compression profile
    prof = cord_compression_profile(active_cord, active_canal)
    if prof is not None:
        result['cord_compression_profile'] = {
            'max_mscc':      prof.max_mscc,
            'max_mscc_z_mm': prof.max_mscc_z_mm,
            'classification':prof.classification,
            'flagged_z_mm':  prof.flagged_z_mm,
            'slice_count':   len(prof.slices),
            'flagged_count': len(prof.flagged_z_mm),
            'slices': [
                {'z_mm': s.z_mm, 'cord_ap': s.cord_ap_mm,
                 'canal_ap': s.canal_ap_mm, 'mscc': round(s.mscc,3),
                 'cls': s.classification, 'flagged': s.flagged}
                for s in prof.slices
            ],
        }

    return result

# ─── Vertebral body morphometry ───────────────────────────────────────────────

def vertebral_heights(vert_mask: np.ndarray) -> Optional[dict]:
    """Ha/Hm/Hp from the Y-extent of the vertebral body mask."""
    if vert_mask is None or not vert_mask.any():
        return None
    coords = np.array(np.where(vert_mask))
    ymin, ymax = int(coords[1].min()), int(coords[1].max())
    y_range = ymax - ymin
    if y_range < 3:
        return None
    t = max(1, y_range // 3)

    def z_span(y_lo, y_hi):
        sub = vert_mask[:, y_lo:y_hi+1, :]
        if not sub.any(): return None
        zc = np.where(sub)[2]
        return (int(zc.max()) - int(zc.min()) + 1) * ISO_MM

    return {
        'Ha': z_span(ymin, ymin+t),
        'Hm': z_span(ymin+t, ymin+2*t),
        'Hp': z_span(ymin+2*t, ymax),
    }

def height_ratios(heights: Optional[dict]) -> dict:
    if not heights:
        return {}
    ha, hm, hp = heights.get('Ha'), heights.get('Hm'), heights.get('Hp')
    result = {}
    if ha and hm: result['Compression_Hm_Ha'] = hm / ha
    if hp and hm: result['Compression_Hm_Hp'] = hm / hp
    if ha and hp:
        result['Wedge_Ha_Hp']  = ha / hp
        result['Crush_Hp_Ha']  = hp / ha
    min_r = min((v for v in result.values() if v), default=1.0)
    result['Genant_Grade'] = (0 if min_r >= 0.80 else
                               1 if min_r >= 0.75 else
                               2 if min_r >= 0.60 else 3)
    result['Genant_Label'] = ['Normal','Mild (20-25%)','Moderate (25-40%)','Severe (>40%)'][
        result['Genant_Grade']]
    return result

# ─── Disc Height Index ────────────────────────────────────────────────────────

def disc_height_index_farfan(disc_mask: np.ndarray,
                              sup_vert: Optional[np.ndarray],
                              inf_vert: Optional[np.ndarray]) -> Optional[float]:
    """DHI = (Ha+Hp)/(Ds+Di) × 100  [Farfan method]."""
    if disc_mask is None or not disc_mask.any():
        return None
    coords = np.array(np.where(disc_mask))
    ymin, ymax = int(coords[1].min()), int(coords[1].max())
    zmin, zmax = int(coords[2].min()), int(coords[2].max())
    xmid = int(coords[0].mean())
    t = max(1, (ymax-ymin)//3)

    def z_h(y_lo, y_hi):
        sub = disc_mask[max(0,xmid-2):xmid+3, y_lo:y_hi+1, :]
        if not sub.any(): return None
        zc = np.where(sub)[2]
        return (int(zc.max()) - int(zc.min()) + 1) * ISO_MM

    ha = z_h(ymin, ymin+t); hp = z_h(ymax-t, ymax)

    def vert_depth(vm):
        if vm is None or not vm.any(): return None
        sub = vm[max(0,xmid-2):xmid+3, :, zmin:zmax+1]
        if not sub.any(): return None
        yc = np.where(sub)[1]
        return (int(yc.max()) - int(yc.min()) + 1) * ISO_MM

    ds = vert_depth(sup_vert); di = vert_depth(inf_vert)
    if ha and hp and ds and di and (ds+di) > 0:
        return ((ha+hp)/(ds+di))*100.0
    return None

def disc_height_method2(disc_mask: np.ndarray,
                        vert_mask: np.ndarray) -> Optional[float]:
    if not (disc_mask is not None and disc_mask.any()): return None
    if not (vert_mask  is not None and vert_mask.any()):  return None
    def mid_z(mask):
        xm = int(np.where(mask)[0].mean())
        col = mask[max(0,xm-2):xm+3,:,:]
        if not col.any(): return None
        zc = np.where(col)[2]
        return (int(zc.max())-int(zc.min())+1)*ISO_MM
    hd = mid_z(disc_mask); hv = mid_z(vert_mask)
    return (hd/hv) if (hd and hv and hv > 0) else None

# ─── Spondylolisthesis ────────────────────────────────────────────────────────

def spondylolisthesis_translation(vert_iso: np.ndarray,
                                  upper_lbl: int,
                                  lower_lbl: int) -> Optional[float]:
    """Sagittal anterior translation between two adjacent vertebrae (mm)."""
    upper = (vert_iso == upper_lbl); lower = (vert_iso == lower_lbl)
    if not upper.any() or not lower.any(): return None

    def ant_y(mask):
        c = np.array(np.where(mask))
        ymin = int(c[1].min()); ymax = int(c[1].max())
        zone = mask[:, ymin:ymin+max(1,(ymax-ymin)//3), :]
        return float(np.where(zone)[1].mean())*ISO_MM if zone.any() else None

    yu = ant_y(upper); yl = ant_y(lower)
    return abs(yu-yl) if (yu is not None and yl is not None) else None

# ─── Ligamentum Flavum proxy ──────────────────────────────────────────────────

def lf_metrics(arcus_mask: Optional[np.ndarray],
               canal_mask: Optional[np.ndarray]) -> dict:
    if arcus_mask is None or not arcus_mask.any(): return {}
    if canal_mask is None or not canal_mask.any(): return {}
    dist, _, _ = min_dist_3d(canal_mask, arcus_mask)
    if not np.isfinite(dist): return {}
    cls = ('Severe — neurogenic claudication risk' if dist > T.LFT_SEVERE_MM else
           'Hypertrophied'                          if dist > T.LFT_NORMAL_MM else
           'Normal')
    return {'lft_proxy_mm': dist, 'lft_class': cls}

# ─── Baastrup (kissing spine) ─────────────────────────────────────────────────

def baastrup_metrics(spinous_mask: Optional[np.ndarray]) -> dict:
    if spinous_mask is None or not spinous_mask.any(): return {}
    labeled, n = cc_label(spinous_mask)
    if n < 2: return {'spinous_count': n}
    comps = []
    for i in range(1, n+1):
        comp = (labeled==i); zc = np.where(comp)[2]
        comps.append((float(zc.mean()), int(zc.min()), int(zc.max())))
    comps.sort(key=lambda t: t[0])
    gaps = [(comps[i+1][1]-comps[i][2])*ISO_MM for i in range(len(comps)-1)]
    min_g = min(gaps) if gaps else float('inf')
    return {
        'spinous_count': n,
        'inter_process_gaps_mm': gaps,
        'min_inter_process_gap_mm': min_g,
        'baastrup_contact': min_g <= T.BAASTRUP_CONTACT_MM,
        'baastrup_risk':    min_g <= T.BAASTRUP_RISK_MM,
    }

# ─── Facet tropism ────────────────────────────────────────────────────────────

def facet_tropism(sal: Optional[np.ndarray],
                  sar: Optional[np.ndarray]) -> dict:
    if sal is None or not sal.any(): return {}
    if sar is None or not sar.any(): return {}

    def angle(mask):
        c = np.array(np.where(mask), dtype=float).T
        if len(c) < 5: return None
        xy = c[:,:2]; xy -= xy.mean(0)
        vals, vecs = np.linalg.eigh(np.cov(xy.T))
        principal = vecs[:, np.argmax(vals)]
        return np.degrees(np.arctan2(principal[1], principal[0])) % 180.0

    al = angle(sal); ar = angle(sar)
    if al is None or ar is None: return {}
    trop = abs(al-ar); trop = min(trop, 180-trop)
    grade = ('Grade 0 (normal asymmetry)' if trop <= T.TROPISM_NORMAL_DEG else
             'Grade 1 (disc prolapse risk)' if trop < T.TROPISM_SEVERE_DEG else
             'Grade 2 (spondylolisthesis risk)')
    return {
        'facet_angle_l_deg': al,
        'facet_angle_r_deg': ar,
        'facet_tropism_deg': trop,
        'facet_tropism_grade': grade,
    }

# ─── Neural foraminal volume proxy ────────────────────────────────────────────

def foraminal_volume(art_mask: Optional[np.ndarray],
                     level: str, side: str) -> dict:
    if art_mask is None or not art_mask.any(): return {}
    c = np.array(np.where(art_mask), dtype=float)
    if c.shape[1] < 5: return {}
    a = (float(c[2].max())-float(c[2].min()))*ISO_MM   # SI height
    b = (float(c[1].max())-float(c[1].min()))*ISO_MM   # AP
    h = (float(c[0].max())-float(c[0].min()))*ISO_MM   # ML depth
    if not (a>0 and b>0 and h>0): return {}
    vol = (np.pi*a*b*h)/4.0
    result = {f'foraminal_vol_{side}_mm3': vol}
    norms = T.FORAMEN_NORMS.get(level, {})
    norm_v = norms.get(side)
    if norm_v:
        pct = (vol/norm_v)*100.0
        result[f'foraminal_norm_pct_{side}'] = pct
        result[f'foraminal_class_{side}'] = (
            'Severe (Lee Grade 3 equiv)'  if pct < 60 else
            'Moderate (Lee Grade 2 equiv)'if pct < 80 else
            'Mild (Lee Grade 1 equiv)'    if pct < 95 else
            'Normal (Lee Grade 0)')
    return result

# ─── Per-level disc source selection ─────────────────────────────────────────

def _get_disc_mask(masks: MaskSet, upper_lbl: int,
                   upper_mask: Optional[np.ndarray],
                   lower_mask: Optional[np.ndarray],
                   level_str: str) -> Tuple[Optional[np.ndarray], str]:
    """Resolve disc mask: VERIDAH(100+X) > TSS disc > SPINEPS merged."""
    veridah_lbl = VD_IVD_BASE + upper_lbl
    if veridah_lbl in masks.vert_labels:
        return (masks.vert_iso == veridah_lbl), f'VERIDAH({veridah_lbl})'

    tss_dlbl = TSS_DISC_MAP.get(level_str)
    if tss_dlbl and masks.tss_iso is not None and tss_dlbl in masks.tss_labels:
        return (masks.tss_iso == tss_dlbl), f'TSS({tss_dlbl})'

    if SP_IVD in masks.sp_labels and upper_mask is not None and lower_mask is not None:
        zr_up = get_z_range(upper_mask); zr_lo = get_z_range(lower_mask)
        if zr_up and zr_lo:
            z_lo = min(zr_up[1], zr_lo[1]); z_hi = max(zr_up[0], zr_lo[0])
            disc = (masks.sp_iso == SP_IVD).copy()
            disc[:,:,:z_lo] = False; disc[:,:,z_hi+1:] = False
            if disc.any():
                return disc, 'SPINEPS-merged'
    return None, 'none'

def _get_endplate_dist(masks: MaskSet, upper_lbl: int, lower_lbl: int,
                       upper_mask: Optional[np.ndarray],
                       lower_mask: Optional[np.ndarray]) -> Tuple[Optional[float], str]:
    """Endplate-to-endplate distance. VERIDAH 200+X > SPINEPS merged ep."""
    ep_up_lbl = VD_EP_BASE + upper_lbl
    ep_lo_lbl = VD_EP_BASE + lower_lbl
    ep_up = (masks.vert_iso == ep_up_lbl) if ep_up_lbl in masks.vert_labels else None
    ep_lo = (masks.vert_iso == ep_lo_lbl) if ep_lo_lbl in masks.vert_labels else None
    src = 'VERIDAH'

    if (ep_up is None or ep_lo is None) and (SP_ENDPLATE in masks.sp_labels):
        if upper_mask is not None and lower_mask is not None:
            zr_up = get_z_range(upper_mask); zr_lo = get_z_range(lower_mask)
            if zr_up and zr_lo:
                ep_g = (masks.sp_iso == SP_ENDPLATE)
                z_top = zr_up[1]; z_bot = zr_lo[0]
                if ep_up is None:
                    tmp = ep_g.copy(); tmp[:,:,:max(0,z_top-3)]=False; tmp[:,:,z_top+4:]=False
                    ep_up = tmp if tmp.any() else None; src = 'SPINEPS-ep62'
                if ep_lo is None:
                    tmp = ep_g.copy(); tmp[:,:,:max(0,z_bot-3)]=False; tmp[:,:,z_bot+4:]=False
                    ep_lo = tmp if tmp.any() else None; src = 'SPINEPS-ep62'

    if ep_up is not None and ep_lo is not None:
        d, _, _ = min_dist_3d(ep_up, ep_lo)
        return (d if np.isfinite(d) else None), src
    return None, 'none'

# ─── Per-level canal AP ───────────────────────────────────────────────────────

def level_canal_ap(masks: MaskSet, level_str: str,
                   active_canal: Optional[np.ndarray]) -> dict:
    if active_canal is None: return {}
    tss_dlbl = TSS_DISC_MAP.get(level_str)
    if not tss_dlbl or masks.tss_iso is None or tss_dlbl not in masks.tss_labels:
        return {}
    disc_zr = get_z_range(masks.tss_iso == tss_dlbl)
    if not disc_zr: return {}
    z_mid = (disc_zr[0]+disc_zr[1])//2
    sl = active_canal[:, :, max(0,z_mid-1):z_mid+2]
    if not sl.any(): return {}
    yc = np.where(sl)[1]; zc = np.where(sl)[2]
    ap = (int(yc.max())-int(yc.min())+1)*ISO_MM
    ml = (int(zc.max())-int(zc.min())+1)*ISO_MM
    dsca = (np.pi/4.0)*ap*ml
    ap_cls, dsca_cls = classify_stenosis(ap, dsca)
    return {'level_ap_mm':ap,'level_dsca_mm2':dsca,
            'level_ap_class':ap_cls,'level_dsca_class':dsca_cls}

# ─── Main morphometric runner ─────────────────────────────────────────────────

def run_all_morphometrics(masks: MaskSet) -> MorphometricResult:
    """
    Run every morphometric module on a MaskSet.
    Returns a fully populated MorphometricResult.
    """
    res = MorphometricResult(study_id=masks.study_id)

    # ── 1. Global canal ───────────────────────────────────────────────────────
    c = canal_metrics_global(masks)
    res.canal_source            = c.get('canal_source')
    res.canal_ap_mm             = c.get('canal_ap_mm')
    res.canal_dsca_mm2          = c.get('canal_dsca_mm2')
    res.canal_ap_class          = c.get('canal_ap_class')
    res.canal_dsca_class        = c.get('canal_dsca_class')
    res.canal_absolute_stenosis = c.get('canal_absolute_stenosis', False)

    # ── 2. Cord (global + full-length profile) ────────────────────────────────
    cm = cord_metrics_global(masks)
    res.cord_source            = cm.get('cord_source')
    res.cord_ap_mm             = cm.get('cord_ap_mm')
    res.cord_ml_mm             = cm.get('cord_ml_mm')
    res.cord_csa_mm2           = cm.get('cord_csa_mm2')
    res.canal_csa_mm2          = cm.get('canal_csa_mm2')
    res.mscc_proxy             = cm.get('mscc_proxy')
    res.canal_occupation_ratio = cm.get('canal_occupation_ratio')
    res.cord_compression_profile = cm.get('cord_compression_profile')

    # ── 3. Ligamentum flavum ──────────────────────────────────────────────────
    arcus = (masks.sp_iso == SP_ARCUS)  if SP_ARCUS  in masks.sp_labels else None
    canal_m = ((masks.tss_iso == TSS_CANAL) if (masks.tss_iso is not None and
                TSS_CANAL in masks.tss_labels)
               else ((masks.sp_iso == SP_CANAL) if SP_CANAL in masks.sp_labels else None))
    lf = lf_metrics(arcus, canal_m)
    res.lft_proxy_mm = lf.get('lft_proxy_mm')
    res.lft_class    = lf.get('lft_class')

    # ── 4. Baastrup ───────────────────────────────────────────────────────────
    spinous = (masks.sp_iso == SP_SPINOUS) if SP_SPINOUS in masks.sp_labels else None
    b = baastrup_metrics(spinous)
    res.spinous_count             = b.get('spinous_count')
    res.min_inter_process_gap_mm  = b.get('min_inter_process_gap_mm')
    res.inter_process_gaps_mm     = b.get('inter_process_gaps_mm')
    res.baastrup_contact          = b.get('baastrup_contact', False)
    res.baastrup_risk             = b.get('baastrup_risk', False)

    # ── 5. Facet tropism ──────────────────────────────────────────────────────
    sal = (masks.sp_iso == SP_SAL) if SP_SAL in masks.sp_labels else None
    sar = (masks.sp_iso == SP_SAR) if SP_SAR in masks.sp_labels else None
    ft = facet_tropism(sal, sar)
    res.facet_angle_l_deg  = ft.get('facet_angle_l_deg')
    res.facet_angle_r_deg  = ft.get('facet_angle_r_deg')
    res.facet_tropism_deg  = ft.get('facet_tropism_deg')
    res.facet_tropism_grade= ft.get('facet_tropism_grade')

    # ── 6. Per-level ──────────────────────────────────────────────────────────
    corpus = (masks.sp_iso == SP_CORPUS) if SP_CORPUS in masks.sp_labels else None
    inf_art_l = (masks.sp_iso == SP_IAL) if SP_IAL in masks.sp_labels else None
    inf_art_r = (masks.sp_iso == SP_IAR) if SP_IAR in masks.sp_labels else None

    for upper_lbl, lower_lbl, up_name, lo_name in LUMBAR_PAIRS:
        level_str  = f'{up_name}_{lo_name}'
        level_disp = f'{up_name}-{lo_name}'
        lm = LevelMetrics(level=level_str, level_display=level_disp)

        upper_m = (masks.vert_iso == upper_lbl) if upper_lbl in masks.vert_labels else None
        lower_m = (masks.vert_iso == lower_lbl) if lower_lbl in masks.vert_labels else None

        # Disc
        disc_m, disc_src = _get_disc_mask(masks, upper_lbl, upper_m, lower_m, level_str)
        lm.disc_source = disc_src

        if disc_m is not None:
            dhi = disc_height_index_farfan(disc_m, upper_m, lower_m)
            lm.dhi_pct = dhi
            if dhi is not None:
                lm.dhi_grade = ('Severe (>50% loss)' if dhi < T.DHI_SEVERE_PCT else
                                'Moderate'           if dhi < T.DHI_MODERATE_PCT else
                                'Mild'               if dhi < T.DHI_MILD_PCT else 'Normal')
            if upper_m is not None:
                lm.dhi_method2 = disc_height_method2(disc_m, upper_m)

        # Endplate distance
        ep_d, ep_src = _get_endplate_dist(masks, upper_lbl, lower_lbl, upper_m, lower_m)
        lm.endplate_dist_mm = ep_d
        lm.endplate_source  = ep_src

        # Level canal AP
        lac = level_canal_ap(masks, level_str, canal_m)
        lm.level_ap_mm      = lac.get('level_ap_mm')
        lm.level_dsca_mm2   = lac.get('level_dsca_mm2')
        lm.level_ap_class   = lac.get('level_ap_class')
        lm.level_dsca_class = lac.get('level_dsca_class')
        shape = CANAL_SHAPE.get(lo_name)
        if shape: lm.canal_shape = f'{shape[0]} ({shape[1]})'

        # Vertebral heights — prefer corpus border slice
        vert_src = upper_m
        if corpus is not None and upper_m is not None:
            zr = get_z_range(upper_m)
            if zr:
                cb = corpus.copy()
                cb[:,:,:zr[0]]=False; cb[:,:,zr[1]+1:]=False
                if cb.any(): vert_src = cb
        h = vertebral_heights(vert_src)
        if h:
            lm.ha_mm = h.get('Ha'); lm.hm_mm = h.get('Hm'); lm.hp_mm = h.get('Hp')
            r = height_ratios(h)
            lm.compression_hm_ha = r.get('Compression_Hm_Ha')
            lm.compression_hm_hp = r.get('Compression_Hm_Hp')
            lm.wedge_ha_hp       = r.get('Wedge_Ha_Hp')
            lm.crush_hp_ha       = r.get('Crush_Hp_Ha')
            lm.genant_grade      = r.get('Genant_Grade')
            lm.genant_label      = r.get('Genant_Label')

        # Spondylolisthesis
        trans = spondylolisthesis_translation(masks.vert_iso, upper_lbl, lower_lbl)
        lm.sagittal_translation_mm = trans
        if trans is not None:
            lm.spondylolisthesis = (
                f'POSITIVE ({trans:.1f}mm ≥ {T.SPONDYLO_MM}mm)'
                if trans >= T.SPONDYLO_MM
                else f'Negative ({trans:.1f}mm < {T.SPONDYLO_MM}mm)')

        # Foraminal
        fvl = foraminal_volume(sal,      level_str, 'L')
        fvr = foraminal_volume(sar,      level_str, 'R')
        lm.foraminal_vol_L_mm3  = fvl.get(f'foraminal_vol_L_mm3')
        lm.foraminal_vol_R_mm3  = fvr.get(f'foraminal_vol_R_mm3')
        lm.foraminal_class_L    = fvl.get(f'foraminal_class_L')
        lm.foraminal_class_R    = fvr.get(f'foraminal_class_R')
        lm.foraminal_norm_pct_L = fvl.get(f'foraminal_norm_pct_L')
        lm.foraminal_norm_pct_R = fvr.get(f'foraminal_norm_pct_R')

        res.levels.append(lm)

    logger.info(f"  [{masks.study_id}] morphometrics done — "
                f"{len([k for k,v in res.to_dict().items() if v is not None])} values")
    return res
