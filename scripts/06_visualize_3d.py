#!/usr/bin/env python3
"""
06_visualize_3d.py — LSTV-Focused Interactive 3D Spine Viewer (v5)
===================================================================
Renders 3D interactive HTML for LSTV cases.
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
import plotly.graph_objects as go
from scipy.ndimage import binary_fill_holes, gaussian_filter, label as cc_label, zoom as ndizoom
from skimage.measure import marching_cubes

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lstv_engine import (
    ISO_MM, TP_HEIGHT_MM, CONTACT_DIST_MM, EXPECTED_LUMBAR,
    TSS_SACRUM, TSS_LUMBAR, TSS_THORACIC, TSS_DISCS, TSS_CORD, TSS_CANAL,
    SP_TP_L, SP_TP_R, SP_SACRUM, SP_ARCUS, SP_SPINOUS, SP_CORPUS,
    SP_CORD, SP_CANAL, SP_SAL, SP_SAR,
    VD_L1, VD_L2, VD_L3, VD_L4, VD_L5, VD_L6, VD_SAC,
    VD_IVD_BASE, VD_EP_BASE,
    VERIDAH_NAMES, TV_SHAPE_LUMBAR, TV_SHAPE_SACRAL,
    DHI_REDUCED_PCT, DHI_MODERATE_PCT, DHI_MILD_PCT,
    compute_lstv_pathology_score,
)

# ── Vertebral angle thresholds (Seilanian Toosi et al. 2025) ──────────────────
# Defined locally — not yet in lstv_engine.py
DELTA_ANGLE_TYPE2_THRESHOLD = 8.5
C_ANGLE_LSTV_THRESHOLD      = 35.5
A_ANGLE_NORMAL_MEDIAN       = 41.0
D_ANGLE_NORMAL_MEDIAN       = 13.5

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-7s  %(message)s')
logger = logging.getLogger(__name__)

# ── Focused view label sets ────────────────────────────────────────────────────
FOCUSED_VERIDAH_LABELS  = {23, 24, 25, 26}
FOCUSED_VERIDAH_IVD     = {23, 24, 25}
FOCUSED_TSS_LABELS      = {45, 50, 95, 100}
FOCUSED_SPINE_LABELS    = {SP_SACRUM, SP_TP_L, SP_TP_R}

PHENOTYPE_CONFIG = {
    'sacralization': {
        'color': '#ff2222', 'bg': '#3a0000', 'border': '#ff4444',
        'label': 'SACRALIZATION', 'emoji': '🔴',
    },
    'lumbarization': {
        'color': '#ff8c00', 'bg': '#2a1800', 'border': '#ffaa33',
        'label': 'LUMBARIZATION', 'emoji': '🟠',
    },
    'transitional_indeterminate': {
        'color': '#ffe033', 'bg': '#2a2200', 'border': '#ffe066',
        'label': 'TRANSITIONAL (INDETERMINATE)', 'emoji': '🟡',
    },
    'transitional': {
        'color': '#ffe033', 'bg': '#2a2200', 'border': '#ffe066',
        'label': 'TRANSITIONAL (INDETERMINATE)', 'emoji': '🟡',
    },
    'normal': {
        'color': '#2dc653', 'bg': '#001a06', 'border': '#44ff77',
        'label': 'NORMAL VARIANT', 'emoji': '🟢',
    },
}

SPINE_LABELS: List[Tuple] = [
    (SP_SACRUM, 'Sacrum (spine)',    '#ff8c00', 0.80, True,  1.5),
    (SP_ARCUS,  'Arcus Vertebrae',   '#7744bb', 0.60, True,  1.5),
    (SP_SPINOUS,'Spinous Processes', '#d4b830', 0.65, True,  1.5),
    (SP_TP_L,   'TP Left',           '#ff3333', 0.95, False, 0.8),
    (SP_TP_R,   'TP Right',          '#00ccff', 0.95, False, 0.8),
    (SP_SAL,    'Sup Articular L',   '#55aa88', 0.60, True,  1.5),
    (SP_SAR,    'Sup Articular R',   '#338866', 0.60, True,  1.5),
    (SP_CORPUS, 'Corpus Border',     '#5588bb', 0.50, True,  1.5),
    (SP_CORD,   'Spinal Cord',       '#ffe066', 0.72, False, 1.0),
    (SP_CANAL,  'Spinal Canal',      '#00ffb3', 0.25, False, 0.8),
]

VERIDAH_COLOURS: Dict[int, Tuple[str, float]] = {
    20: ('#aabbcc', 0.42), 21: ('#99aabb', 0.42), 22: ('#2288cc', 0.48),
    23: ('#2266aa', 0.55), 24: ('#1e6fa8', 0.68), 25: ('#33aaff', 0.72),
    26: ('#ff8c00', 0.78),
}
VERIDAH_IVD_COLOURS: Dict[int, str] = {
    20: '#ffe28a', 21: '#ffd060', 22: '#ffb830',
    23: '#ff9900', 24: '#ff7700', 25: '#ff5500',
}

TSS_RENDER: List[Tuple] = [
    (50,  'TSS Sacrum',     '#ff8c00', 0.72),
    (41,  'TSS L1',         '#aabbcc', 0.35),
    (42,  'TSS L2',         '#99aabb', 0.35),
    (43,  'TSS L3',         '#2288cc', 0.40),
    (44,  'TSS L4',         '#2266aa', 0.48),
    (45,  'TSS L5',         '#1e6fa8', 0.62),
    (95,  'TSS disc L4-L5', '#ff9900', 0.55),
    (100, 'TSS disc L5-S1', '#ff5500', 0.58),
]


# ── NIfTI helpers ──────────────────────────────────────────────────────────────

def _load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Cannot reduce {path.name} to 3D")
    return data, nii


def _voxmm(nii: nib.Nifti1Image) -> np.ndarray:
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))


def _resample(vol: np.ndarray, vox_mm: np.ndarray) -> np.ndarray:
    return ndizoom(vol.astype(np.int32), (vox_mm / ISO_MM).tolist(),
                   order=0, mode='nearest', prefilter=False).astype(np.int32)


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    c = np.array(np.where(mask))
    return c.mean(axis=1) * ISO_MM if c.size else None


def _z_range(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if not mask.any(): return None
    zc = np.where(mask)[2]
    return int(zc.min()), int(zc.max())


def _inferiormost_cc(mask: np.ndarray,
                     sac: Optional[np.ndarray] = None) -> np.ndarray:
    if not mask.any(): return np.zeros_like(mask, bool)
    lab, n = cc_label(mask)
    if n == 1: return mask.astype(bool)
    sac_zmin = None
    if sac is not None and sac.any():
        sac_zmin = int(np.where(sac)[2].min())
    comps = []
    for i in range(1, n + 1):
        c = (lab == i); zc = np.where(c)[2]
        comps.append((float(zc.mean()), int(zc.max()), c))
    comps.sort(key=lambda t: t[0])
    if sac_zmin is not None:
        cands = [c for _, zm, c in comps if zm < sac_zmin]
        if cands: return cands[0].astype(bool)
    return comps[0][2].astype(bool)


def _tp_height_mm(tp: np.ndarray) -> float:
    if not tp.any(): return 0.0
    zc = np.where(tp)[2]
    return float((int(zc.max()) - int(zc.min()) + 1) * ISO_MM)


def _min_dist(a: np.ndarray, b: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    from scipy.ndimage import distance_transform_edt
    if not a.any() or not b.any(): return float('inf'), None, None
    dt   = distance_transform_edt(~b) * ISO_MM
    di   = np.where(a, dt, np.inf)
    flat = int(np.argmin(di))
    va   = np.array(np.unravel_index(flat, a.shape), dtype=float)
    dist = float(dt[tuple(va.astype(int))])
    cb   = np.array(np.where(b), dtype=float)
    d2   = ((cb.T - va) ** 2).sum(axis=1)
    vb   = cb[:, int(np.argmin(d2))]
    return dist, va * ISO_MM, vb * ISO_MM


def _isolate_z(mask: np.ndarray, z_lo: int, z_hi: int, margin: int = 15) -> np.ndarray:
    out = np.zeros_like(mask)
    lo  = max(0, z_lo - margin)
    hi  = min(mask.shape[2] - 1, z_hi + margin)
    out[:, :, lo:hi + 1] = mask[:, :, lo:hi + 1]
    return out


# ── Bounding box wireframe trace ───────────────────────────────────────────────

def bbox_wireframe(mask: np.ndarray, origin_mm: np.ndarray,
                   colour: str, name: str,
                   dash: str = 'dash', width: int = 4,
                   margin_vox: int = 2) -> Optional[go.Scatter3d]:
    if not mask.any():
        return None
    coords = np.array(np.where(mask))
    x0 = max(0, coords[0].min() - margin_vox) * ISO_MM - origin_mm[0]
    x1 = (coords[0].max() + margin_vox) * ISO_MM - origin_mm[0]
    y0 = max(0, coords[1].min() - margin_vox) * ISO_MM - origin_mm[1]
    y1 = (coords[1].max() + margin_vox) * ISO_MM - origin_mm[1]
    z0 = max(0, coords[2].min() - margin_vox) * ISO_MM - origin_mm[2]
    z1 = (coords[2].max() + margin_vox) * ISO_MM - origin_mm[2]

    corners = [
        (x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
        (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1),
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        p0, p1 = corners[a], corners[b]
        xs += [p0[0], p1[0], None]; ys += [p0[1], p1[1], None]; zs += [p0[2], p1[2], None]

    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        line=dict(color=colour, width=width, dash=dash),
        name=name, showlegend=True, hoverinfo='name',
    )


# ── Marching cubes → Mesh3d ───────────────────────────────────────────────────

def mask_to_mesh3d(iso_mask: np.ndarray,
                   origin_mm: np.ndarray,
                   name: str, colour: str, opacity: float,
                   smooth: float = 1.5, fill: bool = True) -> Optional[go.Mesh3d]:
    if not iso_mask.any(): return None
    m   = binary_fill_holes(iso_mask) if fill else iso_mask.copy()
    vol = gaussian_filter(m.astype(np.float32), sigma=smooth)
    vol = np.pad(vol, 1, mode='constant', constant_values=0)
    if vol.max() <= 0.5 or vol.min() >= 0.5: return None
    try:
        verts, faces, _, _ = marching_cubes(vol, level=0.5,
                                             spacing=(ISO_MM, ISO_MM, ISO_MM))
    except Exception as exc:
        logger.warning(f"  MC failed '{name}': {exc}"); return None
    verts -= ISO_MM
    verts -= origin_mm[np.newaxis, :]
    return go.Mesh3d(
        x=verts[:, 0].tolist(), y=verts[:, 1].tolist(), z=verts[:, 2].tolist(),
        i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
        color=colour, opacity=opacity, name=name,
        showlegend=True, flatshading=False,
        lighting=dict(ambient=0.40, diffuse=0.75, specular=0.28,
                      roughness=0.50, fresnel=0.18),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo='name', showscale=False,
    )


# ── 3D annotation traces ───────────────────────────────────────────────────────

def _line(p0, p1, colour, name, width=6, dash='solid') -> go.Scatter3d:
    return go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode='lines', line=dict(color=colour, width=width, dash=dash),
        name=name, showlegend=True, hoverinfo='name')


def _marker(pos, text, colour, size=12, sym='circle') -> go.Scatter3d:
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=size, color=colour, symbol=sym,
                    line=dict(color='white', width=1)),
        text=[text], textposition='top center',
        textfont=dict(size=12, color=colour),
        name=text, showlegend=False, hoverinfo='text')


def midpt(a, b) -> np.ndarray:
    return (np.asarray(a) + np.asarray(b)) / 2.0


def tp_ruler_traces(tp_iso: np.ndarray, origin_mm: np.ndarray,
                    colour: str, side: str, span_mm: float) -> List:
    if not tp_iso.any(): return []
    zc = np.where(tp_iso)[2]
    z_lo, z_hi = int(zc.min()), int(zc.max())
    xc = np.where(tp_iso)[0]; yc = np.where(tp_iso)[1]
    x_c = int(np.median(xc)); y_c = int(np.median(yc))
    p_lo = np.array([x_c, y_c, z_lo], float) * ISO_MM - origin_mm
    p_hi = np.array([x_c, y_c, z_hi], float) * ISO_MM - origin_mm
    mid  = midpt(p_lo, p_hi)
    flag = f'✗ ≥{TP_HEIGHT_MM:.0f}mm → Type I' if span_mm >= TP_HEIGHT_MM else '✓ <19mm'
    clr  = '#ff4444' if span_mm >= TP_HEIGHT_MM else '#44ff88'
    traces = [_line(p_lo, p_hi, colour, f'TP ruler {side}', width=8)]
    traces.append(_marker(mid, f'{side} TP: {span_mm:.1f}mm  {flag}', clr, size=11, sym='diamond'))
    off = np.array([4., 0., 0.])
    for pt in (p_lo, p_hi):
        traces.append(_line(pt - off, pt + off, colour, f'tick {side}', width=3))
    return traces


def gap_ruler_traces(tp: np.ndarray, sac: np.ndarray,
                     origin_mm: np.ndarray, colour: str,
                     side: str, dist_mm: float) -> List:
    if not tp.any() or not sac.any(): return []
    _, pt_a, pt_b = _min_dist(tp, sac)
    if pt_a is None: return []
    p_a = pt_a - origin_mm; p_b = pt_b - origin_mm
    mid = midpt(p_a, p_b)
    contact = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    lbl = f'CONTACT {dist_mm:.1f}mm' if contact else f'Gap {dist_mm:.1f}mm ✓'
    clr = '#ff2222' if contact else '#44ff88'
    return [
        _line(p_a, p_b, colour, f'gap ruler {side}', width=5,
              dash='dot' if contact else 'dash'),
        _marker(mid, f'{side}: {lbl}', clr, size=9, sym='square'),
    ]


def tv_plane_traces(vert_iso: np.ndarray, tv_label: int,
                    origin_mm: np.ndarray, tv_name: str, phenotype: str) -> List:
    mask = (vert_iso == tv_label)
    if not mask.any(): return []
    zc    = np.where(mask)[2]
    z_mid = int((zc.min() + zc.max()) // 2)
    xs    = np.linspace(0, vert_iso.shape[0] - 1, 14)
    ys    = np.linspace(0, vert_iso.shape[1] - 1, 14)
    xg, yg = np.meshgrid(xs, ys)
    zg    = np.full_like(xg, z_mid)
    cfg   = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    col   = cfg['color']

    def _h2r(h):
        h = h.lstrip('#')
        return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

    plane = go.Surface(
        x=xg * ISO_MM - origin_mm[0],
        y=yg * ISO_MM - origin_mm[1],
        z=zg * ISO_MM - origin_mm[2],
        colorscale=[[0, f"rgba({_h2r(col)},0.14)"],
                    [1, f"rgba({_h2r(col)},0.14)"]],
        showscale=False, opacity=0.22,
        name=f'TV plane ({tv_name})', showlegend=True, hoverinfo='name',
    )
    ctr = _centroid(mask)
    if ctr is not None:
        return [plane, _marker(ctr - origin_mm, f'TV: {tv_name}', col, size=16, sym='cross')]
    return [plane]


def tv_body_annotation_traces(vert_iso: np.ndarray, tv_label: int,
                               origin_mm: np.ndarray, shape: dict,
                               phenotype: str) -> List:
    mask = (vert_iso == tv_label)
    if not mask.any() or not shape: return []
    cfg    = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    colour = cfg['color']
    coords = np.array(np.where(mask))
    y_min  = int(coords[1].min()); y_max = int(coords[1].max())
    z_min  = int(coords[2].min()); z_max = int(coords[2].max())
    x_mid  = int(coords[0].mean())
    p_ant  = np.array([x_mid, y_min, (z_min + z_max) // 2], float) * ISO_MM - origin_mm
    p_post = np.array([x_mid, y_max, (z_min + z_max) // 2], float) * ISO_MM - origin_mm
    p_sup  = np.array([x_mid, (y_min + y_max) // 2, z_max], float) * ISO_MM - origin_mm
    p_inf  = np.array([x_mid, (y_min + y_max) // 2, z_min], float) * ISO_MM - origin_mm
    h_ap   = shape.get('h_ap_ratio', 0)
    sc     = shape.get('shape_class', 'unknown')
    return [
        _line(p_ant, p_post, '#8888aa', 'TV AP depth ruler', width=3, dash='dot'),
        _line(p_sup, p_inf,  colour,    'TV SI height ruler', width=4),
        _marker(midpt(p_sup, p_inf), f'TV H/AP={h_ap:.2f} ({sc})', colour, size=10, sym='diamond-open'),
    ]


def delta_angle_ruler_traces(vert_iso: np.ndarray, tv_label: int,
                              origin_mm: np.ndarray,
                              delta_angle: Optional[float],
                              delta_flag: bool) -> List:
    """3D ruler between TV and TV-1 superior surfaces labelled with delta-angle."""
    if delta_angle is None:
        return []
    tv_mask  = (vert_iso == tv_label)
    tv1_mask = (vert_iso == tv_label - 1)
    if not tv_mask.any() or not tv1_mask.any():
        return []

    def _sup_centroid(mask):
        zc    = np.where(mask)[2]
        z_top = int(zc.max())
        z_band = max(z_top - 3, int(zc.min()))
        slab  = mask.copy(); slab[:, :, :z_band] = False
        if not slab.any(): slab = mask
        c = np.array(np.where(slab))
        return c.mean(axis=1) * ISO_MM - origin_mm

    p_tv  = _sup_centroid(tv_mask)
    p_tv1 = _sup_centroid(tv1_mask)

    colour   = '#ff2222' if delta_flag else ('#ff8800' if delta_angle < 15.0 else '#44ff88')
    flag_str = f'⚠ ≤{DELTA_ANGLE_TYPE2_THRESHOLD}° → Castellvi 2!' if delta_flag else ''
    label    = f'δ={delta_angle:.1f}°  {flag_str}'
    mid      = midpt(p_tv, p_tv1)

    return [
        _line(p_tv, p_tv1, colour, f'delta-angle ruler ({delta_angle:.1f}°)', width=6, dash='dash'),
        _marker(mid, label, colour, size=13, sym='diamond'),
    ]


# ── Clinical narrative ─────────────────────────────────────────────────────────

def _generate_clinical_narrative(result: dict, morpho: dict) -> str:
    phenotype    = morpho.get('lstv_phenotype', 'normal')
    confidence   = morpho.get('phenotype_confidence', '')
    castellvi    = result.get('castellvi_type') or ''
    tv_name      = morpho.get('tv_name', 'L5')
    lumbar_count = morpho.get('lumbar_count_consensus', 5) or 5
    tv_shape     = morpho.get('tv_shape') or {}
    disc_below   = morpho.get('disc_below') or {}
    disc_above   = morpho.get('disc_above') or {}
    score        = result.get('pathology_score', 0)
    lft          = result.get('left') or {}
    rgt          = result.get('right') or {}
    va           = morpho.get('vertebral_angles') or {}

    h_ap      = tv_shape.get('h_ap_ratio')
    shape_c   = tv_shape.get('shape_class', '')
    dhi_b     = disc_below.get('dhi_pct')
    dhi_a     = disc_above.get('dhi_pct')
    dhi_b_lvl = disc_below.get('level', 'below TV')
    dhi_a_lvl = disc_above.get('level', 'above TV')
    l_h = lft.get('tp_height_mm', 0); r_h = rgt.get('tp_height_mm', 0)
    l_d = lft.get('dist_mm', float('inf')); r_d = rgt.get('dist_mm', float('inf'))
    has_ct  = bool(castellvi and castellvi not in ('None', 'N/A'))
    norm_r  = tv_shape.get('norm_ratio')

    delta_angle  = va.get('delta_angle')
    c_angle      = va.get('c_angle')
    a_angle      = va.get('a_angle')
    d_angle      = va.get('d_angle')
    delta_flag   = va.get('delta_le8p5', False)
    c_flag       = va.get('c_le35p5', False)
    disc_pattern = va.get('disc_pattern_lstv', False)
    a_increased  = va.get('a_increased', False)
    d_decreased  = va.get('d_decreased', False)

    paras = []

    conf_str = f' ({confidence} confidence)' if confidence else ''
    if phenotype == 'lumbarization':
        p = (f'<b>Lumbarization</b> identified at lumbosacral junction{conf_str}, '
             f'pathology score {score:.0f}. Six lumbar vertebrae present — '
             f'{tv_name} is the extra mobile segment (Hughes &amp; Saifuddin 2006).')
    elif phenotype == 'sacralization':
        p = (f'<b>Sacralization</b> identified at lumbosacral junction{conf_str}, '
             f'pathology score {score:.0f}. {tv_name} shows progressive sacral incorporation.')
        if lumbar_count == 4:
            p += f' Only {lumbar_count} mobile lumbar segments confirmed.'
    elif phenotype == 'transitional_indeterminate':
        p = (f'<b>Transitional morphology</b> — indeterminate phenotype '
             f'(score {score:.0f}). Castellvi TP morphology present without full primary criteria.')
    else:
        p = (f'<b>No significant LSTV</b> (score {score:.0f}). '
             f'Five lumbar vertebrae, normal lumbosacral morphology.')
    paras.append(p)

    if delta_angle is not None or c_angle is not None:
        angle_parts = []
        if delta_angle is not None:
            clr = 'color:#ff2222' if delta_flag else ('color:#ff8800' if delta_angle < 15 else '')
            flag_txt = (f' <b style="{clr}">⚠ ≤{DELTA_ANGLE_TYPE2_THRESHOLD}° — '
                        f'predicts Type 2 (sens 92.3%, spec 87.9%)</b>') if delta_flag else ''
            angle_parts.append(f'<b>δ={delta_angle:.1f}°</b>{flag_txt}')
        if c_angle is not None:
            clr_c   = 'color:#ff8800' if c_flag else ''
            flag_c  = (f' <b style="{clr_c}">⚠ ≤{C_ANGLE_LSTV_THRESHOLD}° (sens 72.2%)</b>') if c_flag else ''
            angle_parts.append(f'C={c_angle:.1f}°{flag_c}')
        if a_angle is not None:
            angle_parts.append(f'A={a_angle:.1f}°{"↑ (OR 1.14)" if a_increased else ""}')
        if d_angle is not None:
            angle_parts.append(f'D={d_angle:.1f}°{"↓ (OR 0.72)" if d_decreased else ""}')
        p = ('<b>Angles (Seilanian Toosi 2025)</b>: ' + ' | '.join(angle_parts) + '.')
        if disc_pattern:
            p += (' <b style="color:#ffe033">Disc pattern: L4-L5 dehydrated + L5-S1 preserved '
                  '(OR 19.9, p&lt;0.001).</b>')
        paras.append(p)

    if has_ct:
        sides = []
        if l_h >= TP_HEIGHT_MM or l_d <= CONTACT_DIST_MM:
            sides.append(f'left (h={l_h:.1f}mm, gap={l_d:.1f}mm)')
        if r_h >= TP_HEIGHT_MM or r_d <= CONTACT_DIST_MM:
            sides.append(f'right (h={r_h:.1f}mm, gap={r_d:.1f}mm)')
        sides_str = ' and '.join(sides) if sides else 'bilateral'
        type_desc = {
            'IV':  'mixed (Type II unilateral, Type III contralateral)',
            'III': 'complete osseous TP-sacral fusion',
            'II':  'diarthrodial pseudo-articulation',
            'I':   'dysplastic TP ≥19mm without sacral contact',
        }
        ct_key = next((k for k in ('IV','III','II','I') if k in castellvi), '')
        paras.append(f'<b>Castellvi {castellvi.replace("Type ","")}</b> — {sides_str}: '
                     f'{type_desc.get(ct_key, "TP enlargement")} (Castellvi 1984).')

    if dhi_b is not None:
        if dhi_b < DHI_REDUCED_PCT:
            grade_str = f'severely reduced (DHI={dhi_b:.0f}%)'
        elif dhi_b < DHI_MODERATE_PCT:
            grade_str = f'moderately reduced (DHI={dhi_b:.0f}%)'
        elif dhi_b < DHI_MILD_PCT:
            grade_str = f'mildly reduced (DHI={dhi_b:.0f}%)'
        else:
            grade_str = f'preserved (DHI={dhi_b:.0f}%)'
        p = f'Disc <b>{dhi_b_lvl}</b>: {grade_str}.'
        if dhi_a and dhi_a >= DHI_MILD_PCT:
            p += f' Disc above ({dhi_a_lvl}, DHI={dhi_a:.0f}%) preserved.'
        paras.append(p)

    if h_ap and shape_c:
        shape_str = {'lumbar-like': f'lumbar-like (H/AP={h_ap:.2f})',
                     'transitional': f'transitional (H/AP={h_ap:.2f})',
                     'sacral-like': f'sacral-like (H/AP={h_ap:.2f})'}.get(shape_c, f'H/AP={h_ap:.2f}')
        p = f'TV <b>{tv_name}</b> body: {shape_str} (Nardo 2012).'
        if norm_r:
            p += f' TV/L4={norm_r:.2f}.'
        paras.append(p)

    return '\n'.join('<div class="narr-para">' + p + '</div>' for p in paras)


# ── Angle panel ────────────────────────────────────────────────────────────────

def _build_angle_panel(va: dict) -> str:
    if not va:
        return ''

    def row(k, v, cls=''):
        cls_attr = f' {cls}' if cls else ''
        return (f'<div class="pr"><span class="pk">{k}</span>'
                f'<span class="pv{cls_attr}">{v}</span></div>')

    def alert_row(msg, colour='#ff8800', bg='#1a1000'):
        return (f'<div class="angle-alert" style="color:{colour};background:{bg};'
                f'border-left-color:{colour}">{msg}</div>')

    lines = ['<div class="ps">🔭 Vertebral Angles (Seilanian Toosi 2025)</div>']

    delta      = va.get('delta_angle')
    c_ang      = va.get('c_angle')
    a_ang      = va.get('a_angle')
    b_ang      = va.get('b_angle')
    d_ang      = va.get('d_angle')
    d1         = va.get('d1_angle')
    delta_flag = va.get('delta_le8p5', False)
    c_flag     = va.get('c_le35p5', False)
    a_increased= va.get('a_increased', False)
    d_decreased= va.get('d_decreased', False)
    disc_pat   = va.get('disc_pattern_lstv', False)

    if delta is not None:
        d_cls = 'cr' if delta_flag else ('wn' if delta < 15 else 'ok')
        lines.append(row('δ (D−D1)', f'{delta:.1f}°{"  ⚠" if delta_flag else ""}', d_cls))
    if c_ang is not None:
        c_cls = 'cr' if c_flag else ('wn' if c_ang < 38 else 'ok')
        lines.append(row('C-angle', f'{c_ang:.1f}°{"  ⚠" if c_flag else ""}', c_cls))
    if a_ang is not None:
        lines.append(row('A-angle', f'{a_ang:.1f}°{"↑" if a_increased else ""}',
                         'wn' if a_increased else 'ok'))
    if b_ang is not None:
        lines.append(row('B-angle', f'{b_ang:.1f}°'))
    if d_ang is not None:
        lines.append(row('D-angle', f'{d_ang:.1f}°{"↓" if d_decreased else ""}',
                         'wn' if d_decreased else 'ok'))
    if d1 is not None:
        lines.append(row('D1-angle', f'{d1:.1f}°'))

    if delta_flag:
        lines.append(alert_row(
            f'⚠ δ ≤ {DELTA_ANGLE_TYPE2_THRESHOLD}° → Type 2 LSTV  Sn 92.3% Sp 87.9%',
            '#ff3333', '#2a0000'))
    if c_flag and not delta_flag:
        lines.append(alert_row(
            f'⚠ C ≤ {C_ANGLE_LSTV_THRESHOLD}° → any LSTV  Sn 72.2% Sp 57.6%',
            '#ff8800', '#1e1000'))
    if disc_pat:
        lines.append(alert_row(
            '⚠ L4-L5 dehy + L5-S1 preserved  OR 19.9 p&lt;0.001',
            '#ffe033', '#1a1800'))

    lines.append(
        '<div class="pc" style="color:#445566;font-size:.58rem">'
        'Ref: Seilanian Toosi F et al. Arch Bone Jt Surg. 2025;13(5):271-280</div>')
    return '\n'.join(lines)


# ── Metrics HTML panel ─────────────────────────────────────────────────────────

def build_metrics_panel(result: dict) -> str:
    morpho       = result.get('lstv_morphometrics') or {}
    castellvi    = result.get('castellvi_type') or 'None'
    phenotype    = morpho.get('lstv_phenotype', 'normal')
    confidence   = morpho.get('phenotype_confidence', '')
    rationale    = morpho.get('phenotype_rationale', '')
    criteria     = morpho.get('phenotype_criteria', [])
    primary      = morpho.get('primary_criteria_met', [])
    lumbar_count = morpho.get('lumbar_count_consensus', '?')
    tv_name      = morpho.get('tv_name', '?')
    lc_anomaly   = morpho.get('lumbar_count_anomaly', False)
    tv_shape     = morpho.get('tv_shape') or {}
    disc_above   = morpho.get('disc_above') or {}
    disc_below   = morpho.get('disc_below') or {}
    rib          = morpho.get('rib_anomaly') or {}
    score        = result.get('pathology_score', 0)
    xval         = result.get('cross_validation') or {}
    lft          = result.get('left')   or {}
    rgt          = result.get('right')  or {}
    lstv_reasons = result.get('lstv_reason', [])
    probs        = morpho.get('probabilities') or {}
    rad_ev       = morpho.get('radiologic_evidence') or []
    surg         = morpho.get('surgical_relevance') or {}
    rdr          = morpho.get('relative_disc_ratio')
    rdr_note     = morpho.get('relative_disc_note', '')
    grad_note    = tv_shape.get('gradient_note', '')
    caudal_grad  = tv_shape.get('caudal_gradient')
    va           = morpho.get('vertebral_angles') or {}

    cfg   = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    p_col = cfg['color']; p_bg = cfg['bg']; p_bdr = cfg['border']
    p_lbl = cfg['label']; p_emj = cfg['emoji']

    def sect(t): return f'<div class="ps">{t}</div>'
    def row(k, v, cls=''):
        cls_attr = f' {cls}' if cls else ''
        return (f'<div class="pr"><span class="pk">{k}</span>'
                f'<span class="pv{cls_attr}">{v}</span></div>')
    def crit(txt): return f'<div class="pc">• {txt}</div>'

    def prob_bar(label, pct, color):
        w = min(float(pct), 100)
        return (f'<div class="pb-row">'
                f'<span class="pb-label">{label}</span>'
                f'<div class="pb-track"><div class="pb-fill" style="width:{w:.0f}%;background:{color}"></div></div>'
                f'<span class="pb-val">{pct:.0f}%</span></div>')

    def risk_chip(level):
        colours = {'critical':('#ff2222','#3a0000'),'high':('#ff6633','#2a1000'),
                   'moderate':('#ff8800','#1e1000'),'low-moderate':('#ffe033','#1a1800'),
                   'low':('#2dc653','#001a06')}
        fg, bg = colours.get(level, ('#aaa','#111'))
        return (f'<span class="risk-chip" style="color:{fg};background:{bg};'
                f'border-color:{fg}">{level.upper()}</span>')

    def strength_tag(s):
        s = (s or '').lower()
        if s == 'primary':   return '<span class="ev-tag ev-primary">PRIMARY</span>'
        if s == 'secondary': return '<span class="ev-tag ev-secondary">2°</span>'
        if s == 'against':   return '<span class="ev-tag ev-against">AGAINST</span>'
        return '<span class="ev-tag ev-support">SUPPORT</span>'

    def dir_col(d):
        d = (d or '').lower()
        if 'sacral' in d: return '#ff6633'
        if 'lumbar' in d: return '#ffaa33'
        if 'normal' in d: return '#2dc653'
        return '#aaaacc'

    def _n(x): return (x * 100 if (x or 0) <= 1 else x) if x is not None else 0

    lines = []

    lines.append(f'<div class="pstatus" style="background:{p_bg};border:2px solid {p_bdr};color:{p_col}">'
                 f'{p_emj} {p_lbl}</div>')
    lines.append(f'<div class="pconf" style="color:{p_col}">Confidence: {confidence.upper() if confidence else "—"}</div>')

    p_sac  = _n(probs.get('p_sacralization'))
    p_lumb = _n(probs.get('p_lumbarization'))
    p_norm = _n(probs.get('p_normal'))
    dom    = probs.get('dominant_class', '—')
    conf_p = _n(probs.get('confidence_pct'))
    n_crit = probs.get('n_criteria', 0)
    cal_n  = probs.get('calibration_note', '')

    if probs:
        lines.append(sect('Bayesian Probability Model'))
        lines.append(prob_bar('Sacralization', p_sac,  '#ff4444'))
        lines.append(prob_bar('Lumbarization', p_lumb, '#ff8c00'))
        lines.append(prob_bar('Normal',        p_norm, '#2dc653'))
        dom_cls = 'cr' if dom == 'sacralization' else 'wn' if dom == 'lumbarization' else 'ok'
        lines.append(row('Dominant', dom, dom_cls))
        lines.append(row('Confidence', f'{conf_p:.0f}%', 'cr' if conf_p > 85 else 'wn' if conf_p > 65 else 'ok'))
        lines.append(row('Criteria', str(n_crit)))
        if cal_n:
            lines.append(f'<div class="pc" style="color:#5566aa;font-style:italic">{cal_n}</div>')

    angle_panel = _build_angle_panel(va)
    if angle_panel:
        lines.append(angle_panel)

    narrative_html = _generate_clinical_narrative(result, morpho)
    if narrative_html:
        lines.append(sect('Clinical Summary'))
        lines.append(narrative_html)

    if surg:
        wl_risk  = surg.get('wrong_level_risk', '')
        wl_pct   = _n(surg.get('wrong_level_risk_pct'))
        bert_pct = _n(surg.get('bertolotti_probability'))
        nerve_amb= surg.get('nerve_root_ambiguity', False)
        flags    = surg.get('surgical_flags') or []
        approach = surg.get('approach_considerations') or []
        count_rec= surg.get('recommended_counting_method', '')
        ionm_note= surg.get('intraop_neuromonitoring_note', '')

        lines.append(sect('⚕ Surgical Relevance'))
        lines.append(f'<div class="pr"><span class="pk">Wrong-level risk</span>'
                     f'<span class="pv">{risk_chip(wl_risk)}</span></div>')
        if wl_pct:
            lines.append(prob_bar('Level-error probability', wl_pct, '#ff3333'))
        lines.append(row('Nerve root ambiguity',
                         '⚠ YES' if nerve_amb else 'No', 'wn' if nerve_amb else 'ok'))
        lines.append(prob_bar("Bertolotti P", bert_pct, '#ff6633' if bert_pct >= 50 else '#ff8800'))
        if flags:
            lines.append(sect('Intraoperative Flags'))
            for f in flags:
                lines.append(f'<div class="surg-flag">⚠ {f}</div>')
        if count_rec:
            lines.append(f'<div class="surg-note">{count_rec}</div>')
        if approach:
            for a in approach:
                lines.append(f'<div class="surg-note">• {a}</div>')
        if ionm_note:
            lines.append(f'<div class="surg-note">{ionm_note}</div>')

    lines.append(sect('LSTV Detection Basis'))
    if lstv_reasons:
        for r in lstv_reasons:
            lines.append(f'<div class="pc">▶ {r}</div>')
    else:
        lines.append('<div class="pc" style="color:#2dc653">No LSTV criteria met</div>')

    lines.append(sect('Pathology Score'))
    sc_cls = 'cr' if score >= 8 else 'wn' if score >= 3 else 'ok'
    lines.append(row('Score', f'{score:.0f}', sc_cls))

    if rad_ev:
        lines.append(sect('Radiologic Evidence'))
        lines.append('<div class="ev-table">')
        for group_dir, group_label in [
            ('sacralization','Sacralization'), ('lumbarization','Lumbarization'),
            ('normal','Against LSTV'), ('supporting','Supporting'),
        ]:
            group_items = [e for e in rad_ev if (e.get('direction') or '').lower() == group_dir]
            if not group_items: continue
            col = dir_col(group_dir)
            lines.append(f'<div class="ev-group-hdr" style="color:{col}">{group_label}</div>')
            for ev in group_items:
                lr_sac  = ev.get('lr_sac', 0) or 0
                lr_lumb = ev.get('lr_lumb', 0) or 0
                lr_str  = ''
                if lr_sac:  lr_str += f' LR(sac)={lr_sac:+.1f}'
                if lr_lumb: lr_str += f' LR(lum)={lr_lumb:+.1f}'
                lines.append(
                    f'<div class="ev-row">{strength_tag(ev.get("strength",""))}'
                    f'<div class="ev-body">'
                    f'<div class="ev-name">{ev.get("name","")}</div>'
                    f'<div class="ev-finding">{ev.get("finding","")}</div>'
                    f'<div class="ev-meta">{ev.get("citation","")}{lr_str}</div>'
                    f'</div></div>')
        lines.append('</div>')

    lines.append(sect('Castellvi Classification'))
    ct_cls = ('cr' if castellvi and any(x in castellvi for x in ('III','IV'))
              else 'wn' if castellvi and any(x in castellvi for x in ('I','II'))
              else 'ok')
    lines.append(row('Type', castellvi, ct_cls))
    l_h = lft.get('tp_height_mm', 0); r_h = rgt.get('tp_height_mm', 0)
    l_d = lft.get('dist_mm', float('inf')); r_d = rgt.get('dist_mm', float('inf'))
    lines.append(row('Left',  f'{lft.get("classification","—")} | h={l_h:.1f}mm | d={l_d:.1f}mm',
                     'cr' if (l_h >= TP_HEIGHT_MM or l_d <= CONTACT_DIST_MM) else 'ok'))
    lines.append(row('Right', f'{rgt.get("classification","—")} | h={r_h:.1f}mm | d={r_d:.1f}mm',
                     'cr' if (r_h >= TP_HEIGHT_MM or r_d <= CONTACT_DIST_MM) else 'ok'))

    lines.append(sect('Lumbar Vertebrae'))
    lines.append(row('Count (consensus)', f'{lumbar_count}{"  ⚠" if lc_anomaly else ""}',
                     'cr' if lc_anomaly else 'ok'))
    lines.append(row('TSS',    str(morpho.get('lumbar_count_tss', '—'))))
    lines.append(row('VERIDAH',str(morpho.get('lumbar_count_veridah', '—'))))
    if morpho.get('has_l6'):
        lines.append(row('L6 present', 'YES — LUMBARIZATION signal', 'wn'))

    lines.append(sect('Transitional Vertebra'))
    lines.append(row('TV', tv_name, 'wn'))
    h_ap = tv_shape.get('h_ap_ratio'); shpc = tv_shape.get('shape_class','—')
    shp_cls = ('cr' if shpc == 'sacral-like' else 'wn' if shpc == 'transitional' else 'ok')
    if h_ap:
        lines.append(row('Body H/AP', f'{h_ap:.2f} ({shpc})', shp_cls))
    nr = tv_shape.get('norm_ratio')
    if nr:
        lines.append(row('TV/L4 H:AP', f'{nr:.2f}', 'wn' if nr < 0.80 else 'ok'))
    if caudal_grad is not None:
        lines.append(row('H/AP gradient', f'{caudal_grad:+.3f}/level',
                         'cr' if caudal_grad < -0.04 else 'ok'))
    if grad_note:
        lines.append(f'<div class="pc">{grad_note}</div>')

    lines.append(sect('Adjacent Disc Heights (DHI)'))
    for disc, label in ((disc_above,'Above TV'), (disc_below,'Below TV')):
        dhi = disc.get('dhi_pct'); lvl = disc.get('level','—')
        if dhi is not None:
            cls = 'cr' if dhi < DHI_REDUCED_PCT else 'wn' if dhi < DHI_MODERATE_PCT else 'ok'
            lines.append(row(f'{label} ({lvl})', f'{dhi:.0f}% [{disc.get("grade","?")}]', cls))
        elif disc.get('is_absent'):
            lines.append(row(f'{label} ({lvl})', 'ABSENT', 'cr'))
        else:
            lines.append(row(f'{label} ({lvl})', 'Not detected', 'pm'))
    if rdr is not None:
        lines.append(row('Disc ratio', f'{rdr:.2f} (Farshad-Amacker 2014)',
                         'cr' if rdr < 0.50 else 'wn' if rdr < 0.65 else 'ok'))
    if rdr_note:
        lines.append(f'<div class="pc">{rdr_note}</div>')

    lines.append(sect('Rib / Thoracic Count'))
    thr_count = rib.get('thoracic_count')
    if thr_count is not None:
        lines.append(row('Thoracic', f'{thr_count} (exp {rib.get("expected_thoracic",12)})',
                         'cr' if rib.get('count_anomaly') else 'ok'))
    if rib.get('lumbar_rib_l1'):
        lines.append(row('Lumbar rib L1', f'⚠ {rib.get("lumbar_rib_l1_h_mm",0):.1f}mm', 'cr'))

    if primary:
        lines.append(sect('Primary Criteria'))
        for p in primary:
            lines.append(f'<div class="pc">✓ {p}</div>')

    if criteria:
        lines.append(sect('Classification Evidence'))
        for c in criteria[:7]:
            lines.append(crit(c))

    if rationale:
        lines.append(sect('Rationale'))
        lines.append(f'<div class="preason">{rationale}</div>')

    xval_warns = xval.get('warnings', [])
    if xval_warns:
        lines.append(sect('⚠ QC Warnings'))
        for w in xval_warns: lines.append(f'<div class="pwarning">{w}</div>')
    else:
        sd = xval.get('sacrum_dice')
        if sd is not None:
            lines.append(sect('QC (Cross-Validation)'))
            lines.append(row('Sacrum Dice', f'{sd:.3f}', 'ok' if sd >= 0.30 else 'cr'))

    return '\n'.join(lines)


# ── Main figure builder ────────────────────────────────────────────────────────

def build_3d_figure(study_id: str,
                    spineps_dir: Path,
                    totalspine_dir: Path,
                    result: dict,
                    smooth: float = 2.0,
                    show_tss: bool = True):
    seg_dir   = spineps_dir / 'segmentations' / study_id
    sp_path   = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path  = totalspine_dir / study_id / 'sagittal' / f"{study_id}_sagittal_labeled.nii.gz"

    def _load(p, tag):
        if not p.exists(): logger.warning(f"  Missing: {p.name}"); return None, None
        try: return _load_canonical(p)
        except Exception as exc: logger.warning(f"  {tag}: {exc}"); return None, None

    sp_arr,  nii_ref  = _load(sp_path,   'seg-spine_msk')
    vert_arr, _       = _load(vert_path, 'seg-vert_msk')
    tss_arr,  tss_nii = _load(tss_path,  'TSS sagittal')

    if sp_arr is None or vert_arr is None:
        logger.error(f"[{study_id}] Missing required masks"); return None

    vox_mm   = _voxmm(nii_ref)
    sp_iso   = _resample(sp_arr.astype(np.int32),   vox_mm)
    vert_iso = _resample(vert_arr.astype(np.int32), vox_mm)
    tss_iso  = (_resample(tss_arr.astype(np.int32), _voxmm(tss_nii))
                if tss_arr is not None and tss_nii is not None else None)

    sp_labels   = frozenset(np.unique(sp_iso).tolist())   - {0}
    vert_labels = frozenset(np.unique(vert_iso).tolist()) - {0}
    tss_labels  = (frozenset(np.unique(tss_iso).tolist()) - {0}
                   if tss_iso is not None else frozenset())

    morpho    = result.get('lstv_morphometrics') or {}
    phenotype = morpho.get('lstv_phenotype', 'normal')
    tv_name   = morpho.get('tv_name') or result.get('details', {}).get('tv_name', 'L5')
    cfg       = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    tv_label  = morpho.get('tv_label_veridah') or result.get('details', {}).get('tv_label')
    va        = morpho.get('vertebral_angles') or {}

    sac_iso = np.zeros(sp_iso.shape, bool)
    if tss_iso is not None and TSS_SACRUM in tss_labels:
        sac_iso = (tss_iso == TSS_SACRUM)
    elif SP_SACRUM in sp_labels:
        sac_iso = (sp_iso == SP_SACRUM)

    col_mask  = (vert_iso > 0)
    origin_mm = (_centroid(col_mask)
                 if col_mask.any()
                 else np.array(sp_iso.shape, float) / 2.0 * ISO_MM)

    tv_z = _z_range(vert_iso == tv_label) if tv_label else None

    details      = result.get('details', {})
    tp_corrected = bool(details.get('tp_concordance_corrected', False))
    if tp_corrected and details.get('corrected_tv_z_range'):
        tv_z_for_tp = tuple(details['corrected_tv_z_range'])
    else:
        tv_z_for_tp = tv_z

    def _get_tp(tp_lbl):
        if tp_lbl not in sp_labels: return np.zeros(sp_iso.shape, bool)
        if tv_z_for_tp:
            isolated = _isolate_z(sp_iso == tp_lbl, tv_z_for_tp[0], tv_z_for_tp[1])
            tp       = _inferiormost_cc(isolated, sac_iso if sac_iso.any() else None)
            if not tp.any(): tp = (sp_iso == tp_lbl)
        else:
            tp = (sp_iso == tp_lbl)
        return tp

    tp_L = _get_tp(SP_TP_L); tp_R = _get_tp(SP_TP_R)
    span_L = _tp_height_mm(tp_L); span_R = _tp_height_mm(tp_R)
    dist_L = _min_dist(tp_L, sac_iso)[0]; dist_R = _min_dist(tp_R, sac_iso)[0]
    castellvi = result.get('castellvi_type') or 'None'

    traces: List  = []
    groups:  List[str] = []

    def _add(t, group='focused'):
        if t is not None: traces.append(t); groups.append(group)
    def _add_all(lst, group='focused'):
        for t in lst: _add(t, group)

    for lbl, name, col, op, fh, mx_s in SPINE_LABELS:
        if lbl not in sp_labels: continue
        mask = (tp_L if lbl == SP_TP_L else tp_R if lbl == SP_TP_R else (sp_iso == lbl))
        if not mask.any(): continue
        grp  = 'focused' if lbl in FOCUSED_SPINE_LABELS else 'full'
        _add(mask_to_mesh3d(mask, origin_mm, name, col, op, min(smooth, mx_s), fh), grp)

    for lbl, (col, op) in sorted(VERIDAH_COLOURS.items()):
        if lbl not in vert_labels: continue
        eff_col = cfg['color'] if lbl == tv_label else col
        eff_op  = 0.82 if lbl == tv_label else op
        grp = 'focused' if lbl in FOCUSED_VERIDAH_LABELS else 'full'
        _add(mask_to_mesh3d(vert_iso == lbl, origin_mm,
                            VERIDAH_NAMES.get(lbl, str(lbl)), eff_col, eff_op, smooth, True), grp)

    for base, col in VERIDAH_IVD_COLOURS.items():
        ivd_lbl = VD_IVD_BASE + base
        if ivd_lbl not in vert_labels: continue
        grp = 'focused' if base in FOCUSED_VERIDAH_IVD else 'full'
        _add(mask_to_mesh3d(vert_iso == ivd_lbl, origin_mm,
                            f'IVD {VERIDAH_NAMES.get(base,str(base))}', col, 0.58, smooth, True), grp)

    if show_tss and tss_iso is not None:
        for lbl, name, col, op in TSS_RENDER:
            if lbl not in tss_labels: continue
            grp = 'focused' if lbl in FOCUSED_TSS_LABELS else 'full'
            _add(mask_to_mesh3d(tss_iso == lbl, origin_mm, name, col, op,
                                0.8 if lbl in (TSS_CORD, TSS_CANAL) else smooth,
                                lbl not in (TSS_CORD, TSS_CANAL)), grp)

    if tv_label and tv_label in vert_labels:
        _add_all(tv_plane_traces(vert_iso, tv_label, origin_mm, tv_name, phenotype))
    tv_shape_dict = morpho.get('tv_shape') or {}
    if tv_label and tv_label in vert_labels and tv_shape_dict:
        _add_all(tv_body_annotation_traces(vert_iso, tv_label, origin_mm,
                                            tv_shape_dict, phenotype))
    _add_all(tp_ruler_traces(tp_L, origin_mm, '#ff3333', 'Left',  span_L))
    _add_all(tp_ruler_traces(tp_R, origin_mm, '#00ccff', 'Right', span_R))
    _add_all(gap_ruler_traces(tp_L, sac_iso, origin_mm, '#ff8800', 'Left',  dist_L))
    _add_all(gap_ruler_traces(tp_R, sac_iso, origin_mm, '#00aaff', 'Right', dist_R))

    if tv_label and tv_label in vert_labels:
        _add_all(delta_angle_ruler_traces(vert_iso, tv_label, origin_mm,
                                          va.get('delta_angle'), va.get('delta_le8p5', False)))

    ct_has = bool(castellvi and castellvi not in ('None','N/A'))
    if ct_has:
        ct_col = '#ff2222' if any(x in castellvi for x in ('III','IV')) else '#ff8800'
        if tp_L.any() and (span_L >= TP_HEIGHT_MM or dist_L <= CONTACT_DIST_MM):
            _add(bbox_wireframe(tp_L, origin_mm, ct_col, f'⚠ Castellvi L ({castellvi})', width=5))
        if tp_R.any() and (span_R >= TP_HEIGHT_MM or dist_R <= CONTACT_DIST_MM):
            _add(bbox_wireframe(tp_R, origin_mm, ct_col, f'⚠ Castellvi R ({castellvi})', width=5))

    if va.get('delta_le8p5') and tv_label and tv_label in vert_labels:
        tv_mask  = (vert_iso == tv_label)
        tv1_mask = (vert_iso == tv_label - 1)
        if tv_mask.any() and tv1_mask.any():
            _add(bbox_wireframe(tv_mask | tv1_mask, origin_mm, '#ff2222',
                                f'⚠ δ ≤ {DELTA_ANGLE_TYPE2_THRESHOLD}°', dash='dot', width=4, margin_vox=4))

    lumbar_count = morpho.get('lumbar_count_consensus', 5) or 5
    if lumbar_count == 6 and VD_L6 in vert_labels:
        l6_mask = (vert_iso == VD_L6)
        if l6_mask.any():
            _add(bbox_wireframe(l6_mask, origin_mm, '#ff8c00',
                                '⚠ L6 — LUMBARIZATION', dash='dash', width=5, margin_vox=3))

    if lumbar_count == 4:
        l4_mask = None
        if tss_iso is not None and 44 in tss_labels:
            l4_mask = (tss_iso == 44)
        elif VD_L4 in vert_labels:
            l4_mask = (vert_iso == VD_L4)
        if l4_mask is not None and l4_mask.any() and sac_iso.any():
            l4_z_inf  = int(np.where(l4_mask)[2].min())
            sac_z_sup = int(np.where(sac_iso)[2].max())
            if l4_z_inf > sac_z_sup + 8:
                l4_coords = np.where(l4_mask)
                pseudo = np.zeros(sp_iso.shape, bool)
                pseudo[int(l4_coords[0].min()):int(l4_coords[0].max())+1,
                       int(l4_coords[1].min()):int(l4_coords[1].max())+1,
                       sac_z_sup:l4_z_inf+1] = True
                _add(bbox_wireframe(pseudo, origin_mm, '#ff2222',
                                    '⚠ Est. L5 zone — fused (SACRALIZATION)',
                                    dash='dot', width=4, margin_vox=0))

    if not any(isinstance(tr, go.Mesh3d) for tr in traces):
        logger.error(f"[{study_id}] Zero meshes"); return None

    focused_vis      = [True if g == 'focused' else 'legendonly' for g in groups]
    full_vis         = [True] * len(groups)
    focused_vis_json = json.dumps(focused_vis)
    full_vis_json    = json.dumps(full_vis)

    score     = result.get('pathology_score', 0)
    lstv_flag = '⚠ LSTV' if result.get('lstv_detected') else '✓ Normal'
    delta_val = va.get('delta_angle')
    delta_hdr = f'  δ={delta_val:.1f}°{"⚠" if va.get("delta_le8p5") else ""}' if delta_val else ''

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"<b>{study_id}</b>  ·  "
                  f"<span style='color:{cfg['color']}'>{cfg['label']}</span>  ·  "
                  f"Castellvi <b>{castellvi}</b>  ·  TV: <b>{tv_name}</b>  ·  "
                  f"Lumbar: <b>{lumbar_count}</b>  ·  {lstv_flag}  ·  Score: <b>{score:.0f}</b>"
                  + delta_hdr
                  + (' [TP-FIXED]' if tp_corrected else '')),
            font=dict(size=12, color='#e8e8f0'), x=0.01),
        paper_bgcolor='#0a0a18', plot_bgcolor='#0a0a18',
        scene=dict(
            bgcolor='#0a0a18',
            xaxis=_axis('X'), yaxis=_axis('Y'), zaxis=_axis('Z (SI)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=0.2, z=0.3), up=dict(x=0, y=0, z=1))),
        legend=dict(font=dict(color='#e8e8f0', size=9),
                    bgcolor='rgba(10,10,24,0.88)', bordercolor='#222244', borderwidth=1,
                    x=0.01, y=0.97, itemsizing='constant'),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return (fig, castellvi, tv_name,
            result.get('left',{}).get('classification','Normal'),
            result.get('right',{}).get('classification','Normal'),
            span_L, span_R, dist_L, dist_R, morpho, cfg,
            focused_vis_json, full_vis_json)


def _axis(title: str) -> dict:
    return dict(title=title, showgrid=True, gridcolor='#1a1a3a',
                showbackground=True, backgroundcolor='#0a0a18',
                tickfont=dict(color='#6666aa'), titlefont=dict(color='#6666aa'),
                zeroline=False)


# ── HTML template ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LSTV 3D — {study_id}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0a0a18;--sf:#11112a;--bd:#222244;--tx:#e8e8f0;--mu:#6677bb;--base:13px}}
html,body{{background:var(--bg);color:var(--tx);font-family:'JetBrains Mono',monospace;
           font-size:var(--base);height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{display:flex;align-items:center;flex-wrap:wrap;gap:7px;padding:6px 12px;
        border-bottom:1px solid var(--bd);background:var(--sf);flex-shrink:0}}
h1{{font-family:'Syne',sans-serif;font-size:.85rem;font-weight:800}}
.b{{display:inline-block;padding:3px 9px;border-radius:14px;font-size:.72rem;font-weight:600;white-space:nowrap}}
.bs{{background:#222244;color:var(--mu)}}
.bc{{background:{status_bg};color:{status_color};border:1px solid {status_border};font-size:.80rem;font-weight:700;padding:4px 12px;border-radius:4px}}
.bct{{background:#1a2a3a;color:#5599cc}}.bch{{background:#2a1a0a;color:#cc8833}}
.bln{{background:#0a2a0a;color:#44cc66}}.ble{{background:#2a0a0a;color:#ff3333}}
.bnl{{background:#0a1a0a;color:#55cc77;font-weight:600}}
.bsc{{background:#1a1a3a;color:#aaaaff;font-weight:700}}
.bcr{{background:#221010;color:#ff6633;border:1px solid #ff4400;font-weight:700}}
.bdelta{{background:{delta_bg};color:{delta_color};border:1px solid {delta_border};font-weight:700}}
.tb{{display:flex;gap:5px;align-items:center;margin-left:auto;flex-wrap:wrap}}
.tb span{{font-size:.65rem;color:var(--mu);text-transform:uppercase;letter-spacing:.04em}}
button{{background:var(--bg);border:1px solid var(--bd);color:var(--tx);
        font-family:inherit;font-size:.68rem;padding:3px 9px;border-radius:4px;cursor:pointer;transition:background .15s}}
button:hover{{background:var(--bd)}}
button.on{{background:#3366ff;border-color:#3366ff;color:#fff}}
button.on-focus{{background:#ff8c00;border-color:#ff8c00;color:#000;font-weight:700}}
.mt{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:4px 12px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.70rem}}
.m{{display:flex;align-items:center;gap:4px;color:var(--mu)}}
.v{{color:var(--tx);font-weight:600}}
.ok{{color:#2dc653!important}}.wn{{color:#ff8800!important}}.cr{{color:#ff3333!important}}
.note{{font-size:.60rem;color:#3a3a5a;margin-left:auto}}
.main-row{{display:flex;flex:1;min-height:0;overflow:hidden}}
#pl{{flex:1;min-width:0;min-height:0;overflow:hidden}}
#pl .js-plotly-plot,#pl .plot-container{{height:100%!important}}
#mp{{width:360px;flex-shrink:0;background:rgba(10,10,24,0.96);
     border-left:1px solid var(--bd);overflow-y:auto;overflow-x:hidden;
     padding:0 0 16px 0;scrollbar-width:thin;scrollbar-color:#222244 var(--bg)}}
#mp::-webkit-scrollbar{{width:4px}}
#mp::-webkit-scrollbar-thumb{{background:#222244;border-radius:2px}}
.pb-row{{display:flex;align-items:center;gap:5px;padding:2px 10px}}
.pb-label{{color:#8899bb;font-size:.63rem;width:76px;flex-shrink:0;text-align:right}}
.pb-track{{flex:1;height:6px;background:#111130;border-radius:3px;overflow:hidden}}
.pb-fill{{height:100%;border-radius:3px}}
.pb-val{{color:var(--tx);font-size:.66rem;font-weight:600;width:30px;text-align:right}}
.risk-chip{{display:inline-block;padding:1px 7px;border-radius:3px;font-size:.66rem;font-weight:700;border:1px solid}}
.ev-table{{padding:2px 8px}}
.ev-group-hdr{{font-size:.60rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;padding:4px 2px 1px;margin-top:4px;border-bottom:1px solid #161630}}
.ev-row{{display:flex;align-items:flex-start;gap:5px;padding:3px 2px;border-bottom:1px solid #0d0d22}}
.ev-tag{{flex-shrink:0;font-size:.55rem;font-weight:700;padding:1px 4px;border-radius:2px;letter-spacing:.04em;margin-top:1px}}
.ev-primary{{background:#3a1000;color:#ff6633;border:1px solid #ff4400}}
.ev-secondary{{background:#2a1a00;color:#ff9900;border:1px solid #aa6600}}
.ev-against{{background:#001a00;color:#44cc66;border:1px solid #226622}}
.ev-support{{background:#1a1a3a;color:#8899cc;border:1px solid #334488}}
.ev-body{{flex:1;min-width:0}}
.ev-name{{color:#ccd4e8;font-size:.66rem;font-weight:600;line-height:1.3}}
.ev-finding{{color:#8899aa;font-size:.62rem;line-height:1.4;margin-top:1px}}
.ev-meta{{color:#445566;font-size:.58rem;margin-top:1px;font-style:italic}}
.angle-alert{{margin:3px 8px;padding:4px 8px;border-left:3px solid;border-radius:0 4px 4px 0;font-size:.65rem;line-height:1.55;font-weight:600}}
.pstatus{{text-align:center;font-family:'Syne',sans-serif;font-size:.92rem;font-weight:800;
  letter-spacing:.05em;text-transform:uppercase;padding:10px 12px;margin-bottom:2px}}
.pconf{{text-align:center;font-size:.66rem;font-weight:600;padding:2px 12px 6px;opacity:.88}}
.ps{{font-family:'Syne',sans-serif;font-size:.65rem;font-weight:700;color:#6677bb;
     text-transform:uppercase;letter-spacing:.07em;padding:7px 10px 3px;margin-top:5px;
     border-top:1px solid #161630}}
.ps:first-child{{border-top:none;margin-top:0}}
.pr{{display:flex;justify-content:space-between;align-items:baseline;padding:2px 10px;gap:6px}}
.pk{{color:#6677bb;white-space:nowrap;flex-shrink:0;max-width:52%;font-size:.68rem}}
.pv{{text-align:right;color:var(--tx);font-weight:600;word-break:break-word;font-size:.70rem}}
.pv.ok{{color:#2dc653}}.pv.wn{{color:#ff8800}}.pv.cr{{color:#ff3333}}.pv.pm{{color:#6677aa;font-weight:400}}
.pc{{padding:3px 12px;color:#99aabb;font-size:.66rem;line-height:1.5}}
.preason{{padding:4px 10px 5px;color:#aabbcc;font-size:.64rem;line-height:1.55;
          border-left:3px solid #3344aa;margin:3px 8px;background:rgba(30,40,80,.25);border-radius:0 4px 4px 0}}
.pwarning{{padding:3px 10px;color:#ffaa33;font-size:.65rem}}
.narr-para{{padding:4px 10px 5px;color:#c8d8e8;font-size:.68rem;line-height:1.6;margin:2px 0;border-left:2px solid rgba(100,120,200,.35)}}
.narr-para b{{color:#e8eeff}}
.surg-flag{{padding:3px 10px;color:#ffbb44;font-size:.63rem;line-height:1.5;border-left:2px solid #ff6633;margin:1px 8px;background:rgba(40,20,0,.3)}}
.surg-note{{padding:3px 10px;color:#99aacc;font-size:.63rem;line-height:1.55;margin:1px 0}}
.lg{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:3px 12px;border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.67rem}}
.li{{display:flex;align-items:center;gap:4px;color:var(--mu)}}
.sw{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
</style></head><body>
<header>
  <h1>LSTV 3D v5</h1>
  <span class="b bs">{study_id}</span>
  <span class="b bc">{status_emoji}&nbsp;{status_label}</span>
  <span class="b bct">Castellvi: {castellvi}</span>
  <span class="b bch">TV: {tv_name}</span>
  <span class="b {lc_badge}">Lumbar: {lumbar_count}</span>
  <span class="b bdelta">δ={delta_disp}</span>
  {rib_badge}{tp_corrected_badge}{lstv_badge}
  <span class="b bsc">Score: {score:.0f}</span>
  {surgical_risk_badge}{prob_badge}
  <div class="tb">
    <span>View</span>
    <button onclick="setFocused()" id="b-focused" class="on-focus">🎯 Focused</button>
    <button onclick="setFull()" id="b-full">🌐 Full</button>
    &nbsp;<span>Cam</span>
    <button onclick="sv('oblique')" id="b-oblique" class="on">Oblique</button>
    <button onclick="sv('lat')" id="b-lat">Lat</button>
    <button onclick="sv('post')" id="b-post">Post</button>
    <button onclick="sv('ant')" id="b-ant">Ant</button>
    <button onclick="sv('axial')" id="b-axial">Axial</button>
  </div>
</header>
<div class="mt">
  <div class="m">TP-L <span class="v {tpl_c}">{span_L:.1f}mm</span></div>
  <div class="m">TP-R <span class="v {tpr_c}">{span_R:.1f}mm</span></div>
  <div class="m">Gap-L <span class="v {gl_c}">{gap_L}</span></div>
  <div class="m">Gap-R <span class="v {gr_c}">{gap_R}</span></div>
  <div class="m">δ <span class="v {delta_c}">{delta_mt}</span></div>
  <div class="m">C <span class="v {c_c}">{c_mt}</span></div>
  <div class="m">DHI-below <span class="v {dhi_c}">{dhi_disp}</span></div>
  <div class="m">P(sac) <span class="v {psac_c}">{psac_disp}</span></div>
  <div class="m">WL-risk <span class="v {risk_c}">{risk_disp}</span></div>
  <span class="note">drag=rotate · scroll=zoom · click legend=toggle</span>
</div>
<div class="lg">
  <div class="li"><div class="sw" style="background:#ff3333"></div>TP-Left</div>
  <div class="li"><div class="sw" style="background:#00ccff"></div>TP-Right</div>
  <div class="li"><div class="sw" style="background:#ff8c00"></div>Sacrum</div>
  <div class="li"><div class="sw" style="background:{tv_col}"></div>TV ({tv_name})</div>
  <div class="li"><div class="sw" style="background:#2266aa"></div>L4/L5</div>
  <div class="li"><div class="sw" style="background:{delta_sw_col};border:2px dashed {delta_sw_col}"></div>δ-ruler</div>
</div>
<div class="main-row">
  <div id="pl">{plotly_div}</div>
  <div id="mp">{metrics_panel}</div>
</div>
<script>
const FOCUSED_VIS={focused_vis_json};
const FULL_VIS={full_vis_json};
function getPlot(){{return document.querySelector('#pl .js-plotly-plot');}}
function setFocused(){{
  const pd=getPlot();if(!pd)return;
  Plotly.restyle(pd,{{visible:FOCUSED_VIS}});
  document.getElementById('b-focused').className='on-focus';
  document.getElementById('b-full').className='';
}}
function setFull(){{
  const pd=getPlot();if(!pd)return;
  Plotly.restyle(pd,{{visible:FULL_VIS}});
  document.getElementById('b-full').className='on';
  document.getElementById('b-focused').className='';
}}
const CAM={{
  oblique:{{eye:{{x:1.6,y:0.3,z:0.35}},up:{{x:0,y:0,z:1}}}},
  lat:{{eye:{{x:2.5,y:0.0,z:0.0}},up:{{x:0,y:0,z:1}}}},
  post:{{eye:{{x:0.0,y:2.5,z:0.0}},up:{{x:0,y:0,z:1}}}},
  ant:{{eye:{{x:0.0,y:-2.5,z:0.0}},up:{{x:0,y:0,z:1}}}},
  axial:{{eye:{{x:0.0,y:0.0,z:3.0}},up:{{x:0,y:1,z:0}}}},
}};
function sv(n){{
  const pd=getPlot();if(!pd)return;
  Plotly.relayout(pd,{{'scene.camera.eye':CAM[n].eye,'scene.camera.up':CAM[n].up}});
}}
window.addEventListener('load',()=>{{setTimeout(setFocused,400);}});
window.addEventListener('resize',()=>{{const pd=getPlot();if(pd)Plotly.Plots.resize(pd);}});
</script>
</body></html>"""


def save_html(fig, study_id: str, output_dir: Path,
              castellvi: str, tv_name: str, cls_L: str, cls_R: str,
              span_L: float, span_R: float, dist_L: float, dist_R: float,
              morpho: dict, cfg: dict, result: dict,
              focused_vis_json: str, full_vis_json: str,
              tp_corrected: bool = False) -> Path:
    from plotly.io import to_html

    plotly_div = to_html(fig, full_html=False, include_plotlyjs='cdn',
                          config=dict(responsive=True, displayModeBar=True, displaylogo=False))

    def _f(v): return f'{v:.1f}mm' if (v is not None and np.isfinite(v)) else 'N/A'
    def _gc(v): return 'cr' if (np.isfinite(v) and v <= CONTACT_DIST_MM) else 'ok'
    def _hc(v): return 'cr' if v >= TP_HEIGHT_MM else 'ok'
    def _n(x):  return (x * 100 if (x or 0) <= 1 else x) if x is not None else 0

    phenotype    = morpho.get('lstv_phenotype', 'normal')
    lumbar_count = morpho.get('lumbar_count_consensus', '?')
    lc_anomaly   = morpho.get('lumbar_count_anomaly', False)
    score        = result.get('pathology_score', 0)
    rib          = morpho.get('rib_anomaly') or {}
    tv_shape     = morpho.get('tv_shape') or {}
    disc_below   = morpho.get('disc_below') or {}
    dhi_b        = disc_below.get('dhi_pct')
    va           = morpho.get('vertebral_angles') or {}

    status_cfg = PHENOTYPE_CONFIG.get(phenotype, PHENOTYPE_CONFIG['normal'])
    lc_badge   = 'ble' if lc_anomaly else 'bln'
    rib_badge  = '<span class="b ble">Rib ⚠</span>' if rib.get('any_anomaly') else ''
    tp_corrected_badge = '<span class="b bcr">TP-fixed ⚙</span>' if tp_corrected else ''
    lstv_badge = ('<span class="b ble">⚠ LSTV</span>'
                  if result.get('lstv_detected')
                  else '<span class="b bnl">✓ Normal</span>')
    dhi_c  = ('cr' if dhi_b and dhi_b < DHI_REDUCED_PCT else
               'wn' if dhi_b and dhi_b < DHI_MODERATE_PCT else 'ok')
    dhi_disp = f'{dhi_b:.0f}%' if dhi_b else 'N/A'
    tv_col   = status_cfg['color']

    delta_val  = va.get('delta_angle')
    c_val      = va.get('c_angle')
    delta_flag = va.get('delta_le8p5', False)
    c_flag     = va.get('c_le35p5', False)

    delta_mt = f'{delta_val:.1f}°{"⚠" if delta_flag else ""}' if delta_val is not None else 'N/A'
    c_mt     = f'{c_val:.1f}°{"⚠" if c_flag else ""}' if c_val is not None else 'N/A'
    delta_c  = 'cr' if delta_flag else ('wn' if (delta_val is not None and delta_val < 15) else 'ok')
    c_c      = 'cr' if c_flag else ('wn' if (c_val is not None and c_val < 38) else 'ok')

    if delta_val is not None:
        if delta_flag:
            delta_disp = f'{delta_val:.1f}° ⚠'; delta_bg = '#3a0000'; delta_color = '#ff3333'; delta_border = '#ff4444'
        elif delta_val < 15:
            delta_disp = f'{delta_val:.1f}°'; delta_bg = '#1e1000'; delta_color = '#ff8800'; delta_border = '#ff9900'
        else:
            delta_disp = f'{delta_val:.1f}°'; delta_bg = '#001a06'; delta_color = '#2dc653'; delta_border = '#2dc653'
    else:
        delta_disp = 'N/A'; delta_bg = '#111130'; delta_color = '#6677aa'; delta_border = '#334488'

    delta_sw_col = '#ff2222' if delta_flag else ('#ff8800' if (delta_val is not None and delta_val < 15) else '#44ff88')

    probs  = morpho.get('probabilities') or {}
    surg   = morpho.get('surgical_relevance') or {}
    p_sac_v  = _n(probs.get('p_sacralization'))
    wl_risk  = surg.get('wrong_level_risk', '')
    psac_c   = 'cr' if p_sac_v > 70 else 'wn' if p_sac_v > 40 else 'ok'
    risk_c   = 'cr' if wl_risk in ('critical','high') else 'wn' if wl_risk == 'moderate' else 'ok'
    psac_disp = f'{p_sac_v:.0f}%' if probs else 'N/A'
    risk_disp = wl_risk.upper() if wl_risk else 'N/A'

    _risk_bg = {'critical':('#3a0000','#ff2222'),'high':('#2a1000','#ff6633'),
                'moderate':('#1e1000','#ff8800'),'low-moderate':('#1a1800','#ffe033'),
                'low':('#001a06','#2dc653')}
    surg_rbg, surg_rfg = _risk_bg.get(wl_risk, ('#111','#aaa'))
    surgical_risk_badge = (
        f'<span class="b" style="background:{surg_rbg};color:{surg_rfg};border:1px solid {surg_rfg}">'
        f'⚕ {wl_risk.upper()}</span>') if wl_risk else ''

    dom_class = probs.get('dominant_class', '')
    conf_pct  = _n(probs.get('confidence_pct'))
    _dom_fg   = {'sacralization':'#ff6633','lumbarization':'#ffaa33','normal':'#2dc653'}.get(dom_class,'#aaaacc')
    prob_badge = (f'<span class="b" style="background:#0d0d22;color:{_dom_fg};border:1px solid {_dom_fg}">'
                  f'P({dom_class[:3]})={conf_pct:.0f}%</span>') if dom_class else ''

    html = _HTML_TEMPLATE.format(
        study_id=study_id,
        status_label=status_cfg['label'], status_emoji=status_cfg['emoji'],
        status_color=status_cfg['color'], status_bg=status_cfg['bg'],
        status_border=status_cfg['border'],
        castellvi=castellvi, tv_name=tv_name,
        lumbar_count=lumbar_count, lc_badge=lc_badge,
        rib_badge=rib_badge, tp_corrected_badge=tp_corrected_badge,
        lstv_badge=lstv_badge, score=score,
        span_L=span_L, span_R=span_R,
        tpl_c=_hc(span_L), tpr_c=_hc(span_R),
        gap_L=_f(dist_L), gap_R=_f(dist_R),
        gl_c=_gc(dist_L), gr_c=_gc(dist_R),
        delta_mt=delta_mt, delta_c=delta_c,
        c_mt=c_mt, c_c=c_c,
        dhi_disp=dhi_disp, dhi_c=dhi_c,
        tv_col=tv_col,
        delta_disp=delta_disp, delta_bg=delta_bg,
        delta_color=delta_color, delta_border=delta_border,
        delta_sw_col=delta_sw_col,
        psac_disp=psac_disp, psac_c=psac_c,
        risk_disp=risk_disp, risk_c=risk_c,
        surgical_risk_badge=surgical_risk_badge, prob_badge=prob_badge,
        metrics_panel=build_metrics_panel(result),
        plotly_div=plotly_div,
        focused_vis_json=focused_vis_json, full_vis_json=full_vis_json,
    )

    out_path = output_dir / f"{study_id}_lstv_3d.html"
    out_path.write_text(html, encoding='utf-8')
    logger.info(f"  → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ── Study ranking ──────────────────────────────────────────────────────────────

def rank_studies(results: List[dict],
                 n_pathologic: int,
                 n_normal: int) -> Tuple[List[str], List[str]]:
    scored = sorted(
        ((r['study_id'], r.get('pathology_score') or 0) for r in results),
        key=lambda t: t[1], reverse=True,
    )
    pathologic_ids = {sid for sid, _ in scored[:n_pathologic]}
    pathologic     = [sid for sid, _ in scored[:n_pathologic]]
    result_by_id   = {r['study_id']: r for r in results}

    def _is_normal(sid):
        r = result_by_id.get(sid, {})
        if r.get('lstv_detected'): return False
        cnt = (r.get('lstv_morphometrics') or {}).get('lumbar_count_consensus')
        if cnt is not None and cnt != EXPECTED_LUMBAR: return False
        if (r.get('pathology_score') or 0) > 0: return False
        return True

    normal_pool = [(sid, sc) for sid, sc in reversed(scored)
                   if sid not in pathologic_ids and _is_normal(sid)]
    if len(normal_pool) < n_normal:
        extra = [(sid, sc) for sid, sc in reversed(scored)
                 if sid not in pathologic_ids and sid not in {s for s,_ in normal_pool}]
        normal_pool = (normal_pool + extra)[:n_normal]

    normal = [sid for sid, _ in normal_pool[:n_normal]]
    return pathologic, normal


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description='LSTV 3D spine visualiser (v5)')
    parser.add_argument('--spineps_dir',    required=True)
    parser.add_argument('--totalspine_dir', required=True)
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--lstv_json',      default=None)
    parser.add_argument('--study_id',       default=None)
    parser.add_argument('--all',            action='store_true')
    parser.add_argument('--rank_by',        default='lstv')
    parser.add_argument('--top_n',          type=int, default=5)
    parser.add_argument('--top_normal',     type=int, default=2)
    parser.add_argument('--smooth',         type=float, default=2.0)
    parser.add_argument('--no_tss',         action='store_true')
    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    all_results: List[dict]    = []
    result_by_id: Dict[str, dict] = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as fh: all_results = json.load(fh)
            result_by_id = {str(r['study_id']): r for r in all_results}

    if args.study_id:
        study_ids = [args.study_id]
    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
    elif args.rank_by == 'lstv':
        if not all_results:
            parser.error("--rank_by lstv requires --lstv_json")
        pathologic, normal = rank_studies(all_results, args.top_n, args.top_normal)
        seen = set(pathologic)
        study_ids = pathologic + [s for s in normal if s not in seen]
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]
    else:
        parser.error("--rank_by must be 'lstv' or 'all', or use --study_id / --all")

    logger.info(f"Rendering {len(study_ids)} studies → {output_dir}")

    ok = 0
    for sid in study_ids:
        logger.info(f"\n{'='*60}\n[{sid}]")
        try:
            result = result_by_id.get(sid, {
                'study_id': sid, 'lstv_detected': False, 'lstv_reason': [],
                'castellvi_type': None, 'left': {}, 'right': {},
                'lstv_morphometrics': None, 'pathology_score': 0,
            })
            if result.get('pathology_score') is None:
                result['pathology_score'] = compute_lstv_pathology_score(
                    result, result.get('lstv_morphometrics'))

            out = build_3d_figure(sid, spineps_dir, totalspine_dir, result,
                                   smooth=args.smooth, show_tss=not args.no_tss)
            if out is None: continue

            (fig, castellvi, tv_name, cls_L, cls_R,
             span_L, span_R, dist_L, dist_R, morpho, cfg,
             focused_vis_json, full_vis_json) = out

            tp_corrected = bool(result.get('details', {}).get('tp_concordance_corrected', False))
            save_html(fig, sid, output_dir,
                      castellvi, tv_name, cls_L, cls_R,
                      span_L, span_R, dist_L, dist_R, morpho, cfg, result,
                      focused_vis_json, full_vis_json, tp_corrected=tp_corrected)
            ok += 1

        except Exception as exc:
            logger.error(f"  [{sid}] Failed: {exc}")
            logger.debug(traceback.format_exc())

    logger.info(f"\nDone. {ok}/{len(study_ids)} HTMLs → {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
