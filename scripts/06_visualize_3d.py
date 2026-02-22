#!/usr/bin/env python3
"""
06_visualize_3d.py — Comprehensive Interactive 3D Spine Segmentation Viewer
============================================================================
Renders ALL segmentation mask labels as 3D meshes plus full 3D measurement
annotations matching the LSTV detection pipeline.

Label reference (from official READMEs — verified against live log output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPINEPS seg-spine_msk (subregion / semantic) — ACTUAL labels in output:
  26  = Sacrum
  41  = Arcus_Vertebrae          (vertebral arch)
  42  = Spinosus_Process         (spinous process)
  43  = Costal_Process_Left      ← PRIMARY TP SOURCE (left transverse process)
  44  = Costal_Process_Right     ← PRIMARY TP SOURCE (right transverse process)
  45  = Superior_Articular_Left  (superior articular process)
  46  = Superior_Articular_Right
  47  = Inferior_Articular_Left  (inferior articular process)
  48  = Inferior_Articular_Right
  49  = Vertebra_Corpus_border   (vertebral body border)
  60  = Spinal_Cord
  61  = Spinal_Canal
  62  = Endplate
  100 = Vertebra_Disc            (all discs combined in semantic mask)

SPINEPS seg-vert_msk (VERIDAH instance labels):
  Base labels — individual vertebrae:
    1-7   = C1-C7
    8-19  = T1-T12
    28    = T13
    20    = L1,  21=L2,  22=L3,  23=L4,  24=L5,  25=L6
    26    = Sacrum
  100+X   = IVD below vertebra X  (e.g. 120=IVD below L1, 121=IVD below L2…)
  200+X   = Endplate of vertebra X

TotalSpineSeg step2_output (sagittal) — ACTUAL labels in output:
   1  = spinal_cord
   2  = spinal_canal
  11-17  = vertebrae_C1-C7
  21-32  = vertebrae_T1-T12
  41=L1  42=L2  43=L3  44=L4  45=L5
  50  = sacrum
  63-67  = disc_C2_C3 … disc_C6_C7
  71-82  = disc_C7_T1 … disc_T11_T12
  91  = disc_T12_L1
  92  = disc_L1_L2
  93  = disc_L2_L3
  94  = disc_L3_L4
  95  = disc_L4_L5
  100 = disc_L5_S
  ⚠  TSS 43=L3 vertebra, 44=L4 vertebra — completely different from
     SPINEPS 43=TP-Left, 44=TP-Right (different label namespaces!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import (binary_fill_holes, distance_transform_edt,
                           gaussian_filter, label as cc_label)
from skimage.measure import marching_cubes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# LABEL MAPS — SPINEPS seg-spine_msk  (verified from SPINEPS README + logs)
# ============================================================================
# (label, display_name, hex_colour, opacity, fill_holes)
SPINE_MASK_LABELS = [
    # Sacrum
    ( 26, 'Sacrum (spine 26)',          '#ff8c00', 0.72, True),
    # Vertebral arch
    ( 41, 'Arcus Vertebrae',            '#8855cc', 0.50, True),
    # Spinous process
    ( 42, 'Spinous Process',            '#e8c84a', 0.72, True),
    # Transverse / costal processes — PRIMARY TP SOURCE
    ( 43, 'TP Left (costal 43)',        '#ff3333', 0.92, False),
    ( 44, 'TP Right (costal 44)',       '#00d4ff', 0.92, False),
    # Articular processes
    ( 45, 'Sup Articular Left',         '#66ccaa', 0.60, True),
    ( 46, 'Sup Articular Right',        '#44aa88', 0.60, True),
    ( 47, 'Inf Articular Left',         '#aaddcc', 0.55, True),
    ( 48, 'Inf Articular Right',        '#88ccbb', 0.55, True),
    # Vertebral body border
    ( 49, 'Vertebra Corpus Border',     '#6699cc', 0.35, True),
    # Endplate
    ( 62, 'Endplate',                   '#c8f0c8', 0.45, True),
    # All discs combined (semantic mask — one blob)
    (100, 'IVD (all, spine 100)',       '#ffcc44', 0.55, True),
    # Spinal cord and canal
    ( 60, 'Spinal Cord',               '#ffe066', 0.60, False),
    ( 61, 'Spinal Canal',              '#00ffb3', 0.18, False),
]
SPINE_MASK_BY_LABEL = {lbl: (name, col, op, fh)
                        for lbl, name, col, op, fh in SPINE_MASK_LABELS}

# ============================================================================
# LABEL MAPS — SPINEPS seg-vert_msk  (VERIDAH instance labels)
# ============================================================================
# Cervical (shown at low opacity — usually not relevant for LSTV)
VERIDAH_CERVICAL = {i: (f'C{i}', '#557799', 0.25) for i in range(1, 8)}
# Thoracic
VERIDAH_THORACIC = {i + 7: (f'T{i+1}', '#447766', 0.25) for i in range(12)}
VERIDAH_THORACIC[28] = ('T13', '#447766', 0.25)
# Lumbar + sacrum — higher opacity, full colour
VERIDAH_LUMBAR = {
    20: ('L1',     '#1e6fa8', 0.45),
    21: ('L2',     '#2389cc', 0.45),
    22: ('L3',     '#29a3e8', 0.45),
    23: ('L4',     '#52bef5', 0.48),
    24: ('L5',     '#85d4ff', 0.52),
    25: ('L6',     '#aae3ff', 0.52),
    26: ('Sacrum (vert 26)', '#ff8c00', 0.60),
}
# IVDs (100+X) and endplates (200+X) — rendered from seg-vert
VERIDAH_IVD_BASE   = 100   # 100+vert_label = IVD below that vertebra
VERIDAH_EP_BASE    = 200   # 200+vert_label = endplate

# Preferred TV order (inferiormost first)
LUMBAR_LABELS_ORDERED = [25, 24, 23, 22, 21, 20]
VERIDAH_NAMES = {**{k: v[0] for k, v in VERIDAH_LUMBAR.items()},
                 **{k: v[0] for k, v in VERIDAH_CERVICAL.items()},
                 **{k: v[0] for k, v in VERIDAH_THORACIC.items()}}

# ============================================================================
# LABEL MAPS — TotalSpineSeg step2_output  (verified from TSS README + logs)
# ============================================================================
TSS_SACRUM_LABEL = 50
TSS_L5_LABEL     = 45

TSS_LABELS = [
    # Spinal cord / canal
    ( 1,  'TSS Spinal Cord',    '#ffe066', 0.50),
    ( 2,  'TSS Spinal Canal',   '#00ffb3', 0.15),
    # Thoracic vertebrae that may appear in lumbar-focused scans
    ( 31, 'TSS T11',            '#447766', 0.28),
    ( 32, 'TSS T12',            '#447766', 0.28),
    # Lumbar vertebrae
    ( 41, 'TSS L1',             '#1e6fa8', 0.28),
    ( 42, 'TSS L2',             '#2389cc', 0.28),
    ( 43, 'TSS L3',             '#29a3e8', 0.28),  # ← vertebral body, NOT TP
    ( 44, 'TSS L4',             '#52bef5', 0.28),  # ← vertebral body, NOT TP
    ( 45, 'TSS L5',             '#85d4ff', 0.30),
    ( 50, 'TSS Sacrum',         '#ff8c00', 0.65),
    # Thoracic discs that may appear
    ( 82, 'TSS disc T11-T12',   '#ffe28a', 0.40),
    # Lumbar discs
    ( 91, 'TSS disc T12-L1',    '#ffd060', 0.45),
    ( 92, 'TSS disc L1-L2',     '#ffb830', 0.50),
    ( 93, 'TSS disc L2-L3',     '#ff9900', 0.50),
    ( 94, 'TSS disc L3-L4',     '#ff7700', 0.50),
    ( 95, 'TSS disc L4-L5',     '#ff5500', 0.52),
    (100, 'TSS disc L5-S',      '#ff3300', 0.55),
]
TSS_BY_LABEL = {lbl: (name, col, op) for lbl, name, col, op in TSS_LABELS}

# ============================================================================
# KEY CONSTANTS  (must match detection pipeline)
# ============================================================================
TP_LEFT_LABEL   = 43   # seg-spine_msk Costal_Process_Left
TP_RIGHT_LABEL  = 44   # seg-spine_msk Costal_Process_Right
SPINEPS_SACRUM  = 26   # seg-spine_msk Sacrum fallback
VERIDAH_L5      = 24
VERIDAH_L6      = 25
TP_HEIGHT_MM    = 19.0
CONTACT_DIST_MM = 2.0

IAN_PAN_LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
IAN_PAN_LABELS = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

# Valid Plotly Scatter3d marker symbols
_VALID_SYMBOLS = ['circle', 'circle-open', 'cross', 'diamond',
                  'diamond-open', 'square', 'square-open', 'x']

# ============================================================================
# NIBABEL HELPERS
# ============================================================================

def load_canonical(path: Path):
    nii  = nib.load(str(path))
    nii  = nib.as_closest_canonical(nii)
    data = nii.get_fdata()
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Cannot reduce {path.name} to 3D: {data.shape}")
    return data, nii

def voxel_size_mm(nii):
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))

def vox_to_mm(vertices, vox_mm, origin_mm=None):
    v = vertices * vox_mm[np.newaxis, :]
    if origin_mm is not None:
        v -= origin_mm[np.newaxis, :]
    return v

def centroid_vox(mask):
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return None
    return coords.mean(axis=1)

def centroid_mm_space(mask, vox_mm, origin_mm):
    c = centroid_vox(mask)
    if c is None:
        return None
    return c * vox_mm - origin_mm

def min_dist_3d(mask_a, mask_b, vox_mm):
    if not mask_a.any() or not mask_b.any():
        return float('inf'), None, None
    dt      = distance_transform_edt(~mask_b, sampling=vox_mm)
    dist_at = np.where(mask_a, dt, np.inf)
    flat    = int(np.argmin(dist_at))
    vox_a   = np.array(np.unravel_index(flat, mask_a.shape))
    dist_mm = float(dt[tuple(vox_a)])
    cb      = np.array(np.where(mask_b))
    d2      = ((cb.T * vox_mm - vox_a * vox_mm) ** 2).sum(axis=1)
    vox_b   = cb[:, int(np.argmin(d2))]
    return dist_mm, vox_a, vox_b

def tp_z_extent(tp_mask, vox_mm):
    """Return (span_mm, z_lo_vox, z_hi_vox) for a TP mask."""
    if not tp_mask.any():
        return 0.0, None, None
    zc = np.where(tp_mask)[2]
    z_lo, z_hi = int(zc.min()), int(zc.max())
    return (z_hi - z_lo) * vox_mm[2], z_lo, z_hi

def inferiormost_tp_cc(tp3d, sac_mask=None):
    if not tp3d.any():
        return np.zeros_like(tp3d, dtype=bool)
    labeled, n = cc_label(tp3d)
    if n <= 1:
        return tp3d.astype(bool)
    sac_z_min = None
    if sac_mask is not None and sac_mask.any():
        sac_z_min = int(np.where(sac_mask)[2].min())
    cc_info = []
    for i in range(1, n + 1):
        comp = (labeled == i)
        zc   = np.where(comp)[2]
        cc_info.append((float(zc.mean()), int(zc.max()), comp))
    cc_info.sort(key=lambda t: t[0])
    if sac_z_min is not None:
        cands = [c for _, zm, c in cc_info if zm < sac_z_min]
        if cands:
            return cands[0].astype(bool)
    return cc_info[0][2].astype(bool)

def isolate_tp_at_tv(subreg, tp_label, z_min, z_max):
    tp_full = (subreg == tp_label)
    iso = np.zeros_like(tp_full)
    iso[:, :, z_min:z_max + 1] = tp_full[:, :, z_min:z_max + 1]
    return iso

def get_tv_z_range(vert_data, tv_label):
    mask = (vert_data == tv_label)
    if not mask.any():
        return None
    z = np.where(mask)[2]
    return int(z.min()), int(z.max())

# ============================================================================
# STUDY SELECTION  (identical to 05_visualize_overlay.py)
# ============================================================================

def select_studies(csv_path, top_n, rank_by, valid_ids):
    if not csv_path.exists():
        raise FileNotFoundError(f"Uncertainty CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered to {len(df)} studies ({before - len(df)} excluded)")
    if rank_by not in df.columns:
        raise ValueError(f"Column '{rank_by}' not in CSV. "
                         f"Available: {', '.join(df.columns)}")
    df_sorted  = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top_ids    = df_sorted.head(top_n)['study_id'].tolist()
    bottom_ids = df_sorted.tail(top_n)['study_id'].tolist()
    seen, selected = set(), []
    for sid in top_ids + bottom_ids:
        if sid not in seen:
            selected.append(sid); seen.add(sid)
    logger.info(f"Rank={rank_by}  Top{top_n}:{top_ids}  Bot{top_n}:{bottom_ids}")
    return selected

# ============================================================================
# MARCHING CUBES → PLOTLY MESH
# ============================================================================

def mask_to_mesh3d(mask, vox_mm, name, colour, opacity,
                   step=2, smooth_sigma=1.0,
                   fill_holes=True, origin_mm=None):
    """
    Convert a binary 3D mask to a Plotly Mesh3d via marching cubes.
    Returns None (with warning) if mask is empty or MC fails.
    """
    if not mask.any():
        return None
    if fill_holes:
        mask = binary_fill_holes(mask)
    # After fill_holes the mask might still be all-False for degenerate inputs
    if not mask.any():
        return None
    ms   = mask[::step, ::step, ::step]
    vsub = vox_mm * step
    vol  = (gaussian_filter(ms.astype(float), sigma=smooth_sigma)
            if smooth_sigma > 0 else ms.astype(float))
    vol  = np.pad(vol, 1, mode='constant', constant_values=0)
    # Guard: MC requires data with values both above and below level=0.5
    if vol.max() <= 0.5 or vol.min() >= 0.5:
        logger.debug(f"  Skipping MC for '{name}': vol range "
                     f"[{vol.min():.3f}, {vol.max():.3f}] has no level=0.5 crossing")
        return None
    try:
        verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=(1, 1, 1))
    except Exception as e:
        logger.warning(f"  MC failed for '{name}': {e}")
        return None
    verts -= 1.0
    vm = vox_to_mm(verts, vsub, origin_mm)
    return go.Mesh3d(
        x=vm[:, 0].tolist(), y=vm[:, 1].tolist(), z=vm[:, 2].tolist(),
        i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
        color=colour, opacity=opacity,
        name=name, showlegend=True, flatshading=False,
        lighting=dict(ambient=0.35, diffuse=0.75, specular=0.30,
                      roughness=0.6, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo='name', showscale=False,
    )

# ============================================================================
# 3D ANNOTATION HELPERS
# ============================================================================

def _to_mm(vox, vox_mm, origin_mm):
    return np.array(vox, dtype=float) * vox_mm - origin_mm

def ruler_line(p0, p1, colour, name, width=6, dash='solid'):
    return go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode='lines',
        line=dict(color=colour, width=width, dash=dash),
        name=name, showlegend=True, hoverinfo='name',
    )

def label_point(pos, text, colour, size=10, symbol='circle'):
    """Labelled point — only uses valid Plotly Scatter3d marker symbols."""
    if symbol not in _VALID_SYMBOLS:
        symbol = 'circle'
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=size, color=colour, symbol=symbol,
                    line=dict(color='white', width=1)),
        text=[text], textposition='top center',
        textfont=dict(size=11, color=colour),
        name=text, showlegend=False, hoverinfo='text',
    )

def midpt(a, b):
    return (np.array(a) + np.array(b)) / 2.0

# ── TP height ruler ──────────────────────────────────────────────────────────

def tp_height_ruler_traces(tp_mask, vox_mm, origin_mm, colour, side_name, span_mm):
    """Vertical ruler showing TP height, placed at the widest-span X column."""
    if not tp_mask.any():
        return []
    # Find X column with maximum Z span
    best_x, best_span = tp_mask.shape[0] // 2, 0.0
    for x in range(tp_mask.shape[0]):
        col = tp_mask[x]
        if not col.any():
            continue
        zc = np.where(col.any(axis=0))[0]
        if zc.size < 2:
            continue
        span = (zc.max() - zc.min()) * vox_mm[2]
        if span > best_span:
            best_span, best_x = span, x
    col_best = tp_mask[best_x]
    if not col_best.any():
        return []
    zc  = np.where(col_best.any(axis=0))[0]
    yc  = np.where(col_best.any(axis=1))[0]
    z_lo, z_hi = int(zc.min()), int(zc.max())
    y_c = int(yc.mean()) if yc.size else tp_mask.shape[1] // 2

    p_lo = _to_mm([best_x, y_c, z_lo], vox_mm, origin_mm)
    p_hi = _to_mm([best_x, y_c, z_hi], vox_mm, origin_mm)
    mid  = midpt(p_lo, p_hi)
    flag = '✓' if span_mm < TP_HEIGHT_MM else f'✗ ≥{TP_HEIGHT_MM:.0f}mm'
    lbl  = f'{side_name} TP height: {span_mm:.1f}mm  {flag}'

    traces = [ruler_line(p_lo, p_hi, colour, f'Height ruler {side_name}', width=8)]
    traces.append(label_point(mid, lbl, colour, size=9, symbol='diamond'))
    # Tick caps at each end
    off = np.array([5.0, 0, 0])
    for pt in (p_lo, p_hi):
        traces.append(ruler_line(pt - off, pt + off, colour,
                                  f'Tick {side_name}', width=4))
    return traces

# ── TP–Sacrum gap ruler ──────────────────────────────────────────────────────

def gap_ruler_traces(tp_mask, sac_mask, vox_mm, origin_mm, colour, side_name, dist_mm):
    """Line from closest TP voxel to closest sacrum voxel."""
    if not tp_mask.any() or not sac_mask.any():
        return []
    _, vox_a, vox_b = min_dist_3d(tp_mask, sac_mask, vox_mm)
    if vox_a is None or vox_b is None:
        return []
    p_a = _to_mm(vox_a, vox_mm, origin_mm)
    p_b = _to_mm(vox_b, vox_mm, origin_mm)
    mid = midpt(p_a, p_b)
    contact = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    dash    = 'dot' if contact else 'dash'
    clbl    = (f'CONTACT {dist_mm:.1f}mm → P2' if contact
               else f'Gap: {dist_mm:.1f}mm ✓')
    return [
        ruler_line(p_a, p_b, colour, f'Gap ruler {side_name}', width=5, dash=dash),
        label_point(mid, f'{side_name}: {clbl}', colour, size=7, symbol='square'),
    ]

# ── TV identification plane ──────────────────────────────────────────────────

def tv_plane_traces(vert_mask, tv_label, vox_mm, origin_mm, tv_name):
    """Semi-transparent plane at the TV z-midpoint + labelled centroid point."""
    mask = (vert_mask == tv_label)
    if not mask.any():
        return []
    zc    = np.where(mask)[2]
    z_mid = int((zc.min() + zc.max()) // 2)
    xs    = np.linspace(0, vert_mask.shape[0] - 1, 10)
    ys    = np.linspace(0, vert_mask.shape[1] - 1, 10)
    xg, yg = np.meshgrid(xs, ys)
    zg    = np.full_like(xg, z_mid)
    xm = xg * vox_mm[0] - origin_mm[0]
    ym = yg * vox_mm[1] - origin_mm[1]
    zm = zg * vox_mm[2] - origin_mm[2]
    plane = go.Surface(
        x=xm, y=ym, z=zm,
        colorscale=[[0, 'rgba(0,230,180,0.12)'],
                    [1, 'rgba(0,230,180,0.12)']],
        showscale=False, opacity=0.18,
        name=f'TV plane ({tv_name})', showlegend=True, hoverinfo='name',
    )
    ctr = centroid_mm_space(mask, vox_mm, origin_mm)
    pts = ([label_point(ctr, f'TV: {tv_name}', '#00e6b4', size=14, symbol='cross')]
           if ctr is not None else [])
    return [plane] + pts

# ── Castellvi contact spheres ────────────────────────────────────────────────

def castellvi_contact_traces(tp_L, tp_R, sac_mask, vox_mm, origin_mm,
                              cls_L, cls_R, dist_L, dist_R):
    traces = []
    for tp_mask, side, dist_mm, cls in (
        (tp_L, 'Left',  dist_L, cls_L),
        (tp_R, 'Right', dist_R, cls_R),
    ):
        if not (tp_mask.any() and sac_mask.any()):
            continue
        if not (np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM):
            continue
        _, vox_a, _ = min_dist_3d(tp_mask, sac_mask, vox_mm)
        if vox_a is None:
            continue
        p   = _to_mm(vox_a, vox_mm, origin_mm)
        col = '#ff2222' if 'III' in (cls or '') else '#ff9900'
        traces.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode='markers+text',
            marker=dict(size=20, color=col, opacity=0.85,
                        symbol='circle',
                        line=dict(color='white', width=2)),
            text=[f'{side}: {cls}'],
            textposition='middle right',
            textfont=dict(size=13, color=col),
            name=f'Contact {side} ({cls})',
            showlegend=True, hoverinfo='text',
        ))
    return traces

# ── Ian Pan 3D bar chart ─────────────────────────────────────────────────────

def ian_pan_bar_traces(uncertainty_row, origin_mm, x_offset_mm=50):
    """3D bar chart of per-level Ian Pan confidences, placed beside the spine."""
    if uncertainty_row is None:
        return []
    confs = {lvl: uncertainty_row.get(f'{lvl}_confidence', float('nan'))
             for lvl in IAN_PAN_LEVELS}
    valid_confs = [v for v in confs.values() if not np.isnan(v)]
    if not valid_confs:
        return []
    max_conf = max(valid_confs)
    max_h    = 40.0   # mm at confidence = 1.0
    bar_w    = 5.0
    gap      = 2.0
    traces   = []

    for i, (lvl, lbl) in enumerate(zip(IAN_PAN_LEVELS, IAN_PAN_LABELS)):
        conf = confs[lvl]
        if np.isnan(conf):
            continue
        x0 = origin_mm[0] + x_offset_mm + i * (bar_w + gap)
        x1 = x0 + bar_w
        h  = conf * max_h
        z0 = -max_h / 2
        z1 = z0 + h
        y  = 0.0
        col = '#e63946' if conf == max_conf else '#457b9d'
        # Box mesh (8 vertices, 12 triangles)
        vx = [x0, x1, x1, x0, x0, x1, x1, x0]
        vy = [y-1, y-1, y+1, y+1, y-1, y-1, y+1, y+1]
        vz = [z0, z0, z0, z0, z1, z1, z1, z1]
        fi = [0, 0, 1, 1, 4, 4, 0, 0, 3, 3, 1, 2]
        fj = [1, 3, 2, 5, 5, 7, 4, 3, 7, 2, 5, 6]
        fk = [2, 2, 5, 6, 6, 6, 5, 7, 6, 6, 6, 7]
        traces.append(go.Mesh3d(
            x=vx, y=vy, z=vz, i=fi, j=fj, k=fk,
            color=col, opacity=0.80,
            name=f'Ian {lbl}: {conf:.2f}',
            showlegend=True, flatshading=True, hoverinfo='name',
        ))
        traces.append(go.Scatter3d(
            x=[(x0 + x1) / 2], y=[y], z=[z1 + 3],
            mode='text',
            text=[f'{lbl}<br>{conf:.2f}'],
            textfont=dict(size=9, color=col),
            showlegend=False, hoverinfo='skip',
        ))
    return traces

# ============================================================================
# PER-STUDY 3D BUILDER
# ============================================================================

def build_3d_figure(study_id, spineps_dir, totalspine_dir,
                    step=2, smooth=1.5,
                    lstv_result=None, uncertainty_row=None,
                    show_tss=True):

    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_path = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path  = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path   = (totalspine_dir / study_id / 'sagittal'
                  / f"{study_id}_sagittal_labeled.nii.gz")

    def _load(path, label):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}"); return None, None
        try:
            return load_canonical(path)
        except Exception as e:
            logger.warning(f"  Cannot load {label}: {e}"); return None, None

    sag_sp,   nii_ref = _load(spine_path, 'seg-spine_msk')
    sag_vert, _       = _load(vert_path,  'seg-vert_msk')
    sag_tss,  _       = _load(tss_path,   'TSS sagittal')

    if sag_sp is None or nii_ref is None:
        logger.error(f"  [{study_id}] Missing seg-spine_msk"); return None
    if sag_vert is None:
        logger.error(f"  [{study_id}] Missing seg-vert_msk");  return None

    sag_sp   = sag_sp.astype(int)
    sag_vert = sag_vert.astype(int)
    if sag_tss is not None:
        sag_tss = sag_tss.astype(int)

    vox_mm = voxel_size_mm(nii_ref)
    logger.info(f"  Voxel: {vox_mm}  Shape: {sag_sp.shape}")

    # ── Log all unique labels so we can verify against READMEs ──────────────
    sp_labels   = set(np.unique(sag_sp).tolist())   - {0}
    vert_labels = set(np.unique(sag_vert).tolist())  - {0}
    tss_labels  = (set(np.unique(sag_tss).tolist()) - {0}
                   if sag_tss is not None else set())
    logger.info(f"  seg-spine labels present: {sorted(sp_labels)}")
    logger.info(f"  seg-vert  labels present: {sorted(vert_labels)}")
    if sag_tss is not None:
        logger.info(f"  TSS       labels present: {sorted(tss_labels)}")

    # ── Shared origin (centroid of vertebral column) ─────────────────────────
    col_mask  = sag_vert > 0
    origin_mm = (np.array(np.where(col_mask)).mean(axis=1) * vox_mm
                 if col_mask.any()
                 else np.array(sag_sp.shape) / 2.0 * vox_mm)

    # ── Sacrum: prefer TSS 50, fall back to SPINEPS 26 ──────────────────────
    if sag_tss is not None and (sag_tss == TSS_SACRUM_LABEL).any():
        sac_mask = (sag_tss == TSS_SACRUM_LABEL)
        logger.info("  Sacrum: TSS label 50")
    else:
        sac_mask = (sag_sp == SPINEPS_SACRUM)
        logger.warning("  Sacrum: fallback SPINEPS label 26")

    # ── Transitional vertebra from VERIDAH seg-vert_msk ─────────────────────
    tv_label, tv_name = None, 'N/A'
    for cand in LUMBAR_LABELS_ORDERED:
        if cand in vert_labels:
            tv_label = cand
            tv_name  = VERIDAH_NAMES.get(cand, str(cand))
            break

    z_min_tv = z_max_tv = None
    if tv_label is not None:
        zr = get_tv_z_range(sag_vert, tv_label)
        if zr:
            z_min_tv, z_max_tv = zr
            logger.info(f"  TV: {tv_name} (label {tv_label})  "
                        f"z-range [{z_min_tv}, {z_max_tv}]")

    # ── Isolate TP masks at TV z-range ──────────────────────────────────────
    # If TV z-range found, restrict TP search to that range (+margin);
    # otherwise fall back to the full volume.
    if z_min_tv is not None and (TP_LEFT_LABEL in sp_labels or
                                  TP_RIGHT_LABEL in sp_labels):
        # Add a small margin (±5 slices) in case TV and TP don't perfectly overlap
        margin   = 5
        z_lo_tp  = max(0, z_min_tv - margin)
        z_hi_tp  = min(sag_sp.shape[2] - 1, z_max_tv + margin)
        tp_L_raw = isolate_tp_at_tv(sag_sp, TP_LEFT_LABEL,  z_lo_tp, z_hi_tp)
        tp_R_raw = isolate_tp_at_tv(sag_sp, TP_RIGHT_LABEL, z_lo_tp, z_hi_tp)
        # If isolation returns empty, fall back to full volume
        if not tp_L_raw.any():
            tp_L_raw = (sag_sp == TP_LEFT_LABEL)
            logger.warning("  TP-L isolated mask empty — using full volume")
        if not tp_R_raw.any():
            tp_R_raw = (sag_sp == TP_RIGHT_LABEL)
            logger.warning("  TP-R isolated mask empty — using full volume")
    else:
        tp_L_raw = (sag_sp == TP_LEFT_LABEL)
        tp_R_raw = (sag_sp == TP_RIGHT_LABEL)

    tp_L = inferiormost_tp_cc(tp_L_raw, sac_mask if sac_mask.any() else None)
    tp_R = inferiormost_tp_cc(tp_R_raw, sac_mask if sac_mask.any() else None)

    span_L, _, _ = tp_z_extent(tp_L, vox_mm)
    span_R, _, _ = tp_z_extent(tp_R, vox_mm)
    dist_L, _, _ = min_dist_3d(tp_L, sac_mask, vox_mm)
    dist_R, _, _ = min_dist_3d(tp_R, sac_mask, vox_mm)
    logger.info(f"  TP-L: span={span_L:.1f}mm  gap={dist_L:.1f}mm  "
                f"voxels={tp_L.sum()}")
    logger.info(f"  TP-R: span={span_R:.1f}mm  gap={dist_R:.1f}mm  "
                f"voxels={tp_R.sum()}")

    # ── Classification from detection JSON ──────────────────────────────────
    castellvi = 'N/A'
    cls_L     = 'N/A'
    cls_R     = 'N/A'
    if lstv_result:
        castellvi = lstv_result.get('castellvi_type') or 'None'
        cls_L     = lstv_result.get('left',  {}).get('classification', 'N/A')
        cls_R     = lstv_result.get('right', {}).get('classification', 'N/A')
        det_tv    = lstv_result.get('details', {}).get('tv_name')
        if det_tv:
            tv_name = det_tv

    # ========================================================================
    # BUILD TRACES
    # ========================================================================
    traces = []

    # ── 1. SPINEPS seg-spine_msk — all subregion labels ─────────────────────
    for lbl, name, col, op, fh in SPINE_MASK_LABELS:
        if lbl not in sp_labels:
            continue
        # Use the isolated+pruned TP masks for 43/44
        if lbl == TP_LEFT_LABEL:
            mask = tp_L
        elif lbl == TP_RIGHT_LABEL:
            mask = tp_R
        else:
            mask = (sag_sp == lbl)
        if not mask.any():
            continue
        # Use finer MC step for small/thin structures
        fine = max(1, step - 1) if lbl in (43, 44, 42, 45, 46, 47, 48) else step
        t = mask_to_mesh3d(mask, vox_mm, name, col, op,
                           step=fine, smooth_sigma=smooth,
                           fill_holes=fh, origin_mm=origin_mm)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-spine {lbl:>3}  {name}")

    # ── 2. SPINEPS seg-vert_msk — all VERIDAH instance labels ───────────────
    all_veridah = {**VERIDAH_CERVICAL, **VERIDAH_THORACIC, **VERIDAH_LUMBAR}
    for lbl, (name, col, op) in sorted(all_veridah.items()):
        if lbl not in vert_labels:
            continue
        mask = (sag_vert == lbl)
        t = mask_to_mesh3d(mask, vox_mm, name, col, op,
                           step=step, smooth_sigma=smooth,
                           fill_holes=True, origin_mm=origin_mm)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-vert  {lbl:>3}  {name}")

    # ── 2b. VERIDAH IVD labels (100+X present in seg-vert) ──────────────────
    # IVD labels: 100+base_vert.  E.g. 120=IVD below L1, 121=IVD below L2…
    ivd_colour_by_base = {
        20: '#ffe28a', 21: '#ffd060', 22: '#ffb830',
        23: '#ff9900', 24: '#ff7700', 25: '#ff5500',
    }
    for base_lbl, col in ivd_colour_by_base.items():
        ivd_lbl = VERIDAH_IVD_BASE + base_lbl
        if ivd_lbl not in vert_labels:
            continue
        name = f'IVD below {VERIDAH_NAMES.get(base_lbl, str(base_lbl))}'
        mask = (sag_vert == ivd_lbl)
        t = mask_to_mesh3d(mask, vox_mm, name, col, 0.55,
                           step=step, smooth_sigma=smooth,
                           fill_holes=True, origin_mm=origin_mm)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-vert  {ivd_lbl:>3}  {name}")

    # ── 3. TotalSpineSeg — all present labels ────────────────────────────────
    if show_tss and sag_tss is not None:
        for lbl, name, col, op in TSS_LABELS:
            if lbl not in tss_labels:
                continue
            mask = (sag_tss == lbl)
            # Render TSS at reduced opacity to avoid occluding SPINEPS meshes
            t = mask_to_mesh3d(mask, vox_mm, f'TSS {name}', col, op * 0.50,
                               step=step, smooth_sigma=smooth,
                               fill_holes=True, origin_mm=origin_mm)
            if t:
                traces.append(t)
                logger.info(f"    ✓ TSS       {lbl:>3}  {name}")

    if not any(isinstance(tr, go.Mesh3d) for tr in traces):
        logger.error(f"  [{study_id}] No mesh traces generated — check label maps")
        return None

    # ── 4. TV identification plane ───────────────────────────────────────────
    if tv_label is not None:
        traces += tv_plane_traces(sag_vert, tv_label, vox_mm, origin_mm, tv_name)

    # ── 5. TP height rulers ──────────────────────────────────────────────────
    traces += tp_height_ruler_traces(tp_L, vox_mm, origin_mm,
                                      '#ff3333', 'Left',  span_L)
    traces += tp_height_ruler_traces(tp_R, vox_mm, origin_mm,
                                      '#00d4ff', 'Right', span_R)

    # ── 6. TP–Sacrum gap rulers ──────────────────────────────────────────────
    traces += gap_ruler_traces(tp_L, sac_mask, vox_mm, origin_mm,
                                '#ff8800', 'Left',  dist_L)
    traces += gap_ruler_traces(tp_R, sac_mask, vox_mm, origin_mm,
                                '#00aaff', 'Right', dist_R)

    # ── 7. Castellvi contact spheres ─────────────────────────────────────────
    traces += castellvi_contact_traces(tp_L, tp_R, sac_mask,
                                        vox_mm, origin_mm,
                                        cls_L, cls_R, dist_L, dist_R)

    # ── 8. Ian Pan confidence bars ───────────────────────────────────────────
    traces += ian_pan_bar_traces(uncertainty_row, origin_mm, x_offset_mm=50)

    # ── Summary annotation box ───────────────────────────────────────────────
    def _fmt(v):
        return f'{v:.1f} mm' if np.isfinite(v) else 'N/A'
    summary = [
        f"TV:           {tv_name}",
        f"TP-L height:  {_fmt(span_L)}  {'✗ Type I' if span_L >= TP_HEIGHT_MM else '✓'}",
        f"TP-R height:  {_fmt(span_R)}  {'✗ Type I' if span_R >= TP_HEIGHT_MM else '✓'}",
        f"Gap L:        {_fmt(dist_L)}  {'← CONTACT' if np.isfinite(dist_L) and dist_L <= CONTACT_DIST_MM else ''}",
        f"Gap R:        {_fmt(dist_R)}  {'← CONTACT' if np.isfinite(dist_R) and dist_R <= CONTACT_DIST_MM else ''}",
        f"Class Left:   {cls_L}",
        f"Class Right:  {cls_R}",
        f"Castellvi:    {castellvi}",
    ]
    if tp_L.any():
        summary.append(f"TP-L voxels:  {tp_L.sum()}  "
                       f"vol={tp_L.sum() * vox_mm.prod() / 1000:.2f} cm³")
    if tp_R.any():
        summary.append(f"TP-R voxels:  {tp_R.sum()}  "
                       f"vol={tp_R.sum() * vox_mm.prod() / 1000:.2f} cm³")
    if sac_mask.any():
        summary.append(f"Sacrum vol:   "
                       f"{sac_mask.sum() * vox_mm.prod() / 1000:.1f} cm³")
    if uncertainty_row:
        for lvl, lbl in zip(IAN_PAN_LEVELS, IAN_PAN_LABELS):
            v = uncertainty_row.get(f'{lvl}_confidence', float('nan'))
            if not np.isnan(v):
                summary.append(f"Ian {lbl}:  {v:.3f}")

    title_str = (
        f"<b>Study {study_id}</b>"
        f"  ·  Castellvi: <b>{castellvi}</b>"
        f"  ·  TV: <b>{tv_name}</b>"
        f"  ·  L: <b>{cls_L}</b>"
        f"  ·  R: <b>{cls_R}</b>"
    )

    annotations = [
        dict(
            text=('Controls: Left-drag=rotate · Scroll=zoom · '
                  'Legend=toggle · Dbl-click=isolate'),
            xref='paper', yref='paper',
            x=0.5, y=-0.01, xanchor='center', yanchor='top',
            showarrow=False, font=dict(size=10, color='#8888aa'), align='center',
        ),
        dict(
            text='<b>LSTV Measurements</b><br>' + '<br>'.join(summary),
            xref='paper', yref='paper',
            x=0.99, y=0.98, xanchor='right', yanchor='top',
            showarrow=False,
            font=dict(size=11, color='#e8e8f0', family='monospace'),
            bgcolor='rgba(13,13,26,0.88)',
            bordercolor='#2a2a4a', borderwidth=1, align='left',
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title_str, font=dict(size=14, color='#e8e8f0'), x=0.01),
        paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a',
        scene=dict(
            bgcolor='#0d0d1a',
            xaxis=dict(title='X (mm)', showgrid=True, gridcolor='#1a1a3e',
                       showbackground=True, backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'), zeroline=False),
            yaxis=dict(title='Y (mm)', showgrid=True, gridcolor='#1a1a3e',
                       showbackground=True, backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'), zeroline=False),
            zaxis=dict(title='Z (mm)', showgrid=True, gridcolor='#1a1a3e',
                       showbackground=True, backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'), zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=0.0, z=0.3),
                        up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(
            font=dict(color='#e8e8f0', size=10),
            bgcolor='rgba(13,13,26,0.85)',
            bordercolor='#2a2a4a', borderwidth=1,
            x=0.01, y=0.98, itemsizing='constant',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        annotations=annotations,
    )

    return fig, castellvi, tv_name, cls_L, cls_R, span_L, span_R, dist_L, dist_R

# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>3D Spine — {study_id}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@500;700&display=swap');
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:#0d0d1a; --surface:#13132a; --border:#2a2a4a;
      --text:#e8e8f0; --muted:#6666aa; --blue:#3a86ff;
    }}
    html, body {{
      background:var(--bg); color:var(--text);
      font-family:'JetBrains Mono',monospace;
      height:100vh; display:flex; flex-direction:column; overflow:hidden;
    }}
    header {{
      display:flex; align-items:center; gap:10px; flex-wrap:wrap;
      padding:7px 14px; border-bottom:1px solid var(--border);
      background:var(--surface); flex-shrink:0;
    }}
    header h1 {{
      font-family:'Syne',sans-serif; font-size:.90rem; font-weight:700;
      color:var(--text); letter-spacing:.04em; white-space:nowrap;
    }}
    .badge {{
      display:inline-block; padding:2px 8px; border-radius:20px;
      font-size:.67rem; font-weight:600; letter-spacing:.05em;
    }}
    .badge-study      {{ background:#2a2a4a; color:var(--muted); }}
    .badge-castellvi  {{ background:#ff8c00; color:#0d0d1a; }}
    .badge-tv         {{ background:#1e6fa8; color:#fff; }}
    .badge-left       {{ background:#ff3333; color:#fff; }}
    .badge-right      {{ background:#007799; color:#fff; }}
    .badge-ian        {{ background:#1a3a2a; color:#2dc653; border:1px solid #2dc653; }}
    .toolbar {{
      display:flex; gap:5px; align-items:center; margin-left:auto;
    }}
    .toolbar span {{
      font-size:.62rem; color:var(--muted);
      text-transform:uppercase; letter-spacing:.08em;
    }}
    button {{
      background:var(--bg); border:1px solid var(--border); color:var(--text);
      font-family:'JetBrains Mono',monospace; font-size:.67rem;
      padding:3px 9px; border-radius:4px; cursor:pointer; transition:all .15s;
    }}
    button:hover {{ background:var(--border); }}
    button.active {{ background:var(--blue); border-color:var(--blue); color:#fff; }}
    .metrics-strip {{
      display:flex; gap:20px; flex-wrap:wrap; align-items:center;
      padding:4px 14px; background:var(--bg);
      border-bottom:1px solid var(--border); flex-shrink:0;
      font-size:.66rem;
    }}
    .metrics-strip .m {{ display:flex; align-items:center; gap:4px; color:var(--muted); }}
    .metrics-strip .v {{ color:var(--text); font-weight:600; }}
    .ok   {{ color:#2dc653 !important; }}
    .warn {{ color:#ff8800 !important; }}
    .crit {{ color:#ff3333 !important; }}
    .legend-strip {{
      display:flex; gap:12px; flex-wrap:wrap; align-items:center;
      padding:4px 14px; background:var(--bg);
      border-bottom:1px solid var(--border); flex-shrink:0; font-size:.64rem;
    }}
    .legend-strip .item {{ display:flex; align-items:center; gap:3px; color:var(--muted); }}
    .swatch {{ width:10px; height:10px; border-radius:2px; flex-shrink:0; }}
    #plot-container {{ flex:1; min-height:0; }}
    #plot-container .js-plotly-plot,
    #plot-container .plot-container {{ height:100% !important; }}
  </style>
</head>
<body>
  <header>
    <h1>3D SPINE — LSTV</h1>
    <span class="badge badge-study">{study_id}</span>
    <span class="badge badge-castellvi">Castellvi: {castellvi}</span>
    <span class="badge badge-tv">TV: {tv_name}</span>
    <span class="badge badge-left">L: {cls_L}</span>
    <span class="badge badge-right">R: {cls_R}</span>
    {ian_badge}
    <div class="toolbar">
      <span>View</span>
      <button onclick="setView('oblique')"   id="btn-oblique"   class="active">Oblique</button>
      <button onclick="setView('lateral')"   id="btn-lateral">Lateral</button>
      <button onclick="setView('posterior')" id="btn-posterior">Post</button>
      <button onclick="setView('anterior')"  id="btn-anterior">Ant</button>
      <button onclick="setView('axial')"     id="btn-axial">Axial</button>
    </div>
  </header>

  <div class="metrics-strip">
    <div class="m">TP-L height <span class="v {tp_l_cls}">{span_L}</span></div>
    <div class="m">TP-R height <span class="v {tp_r_cls}">{span_R}</span></div>
    <div class="m">Gap L <span class="v {gap_l_cls}">{gap_L}</span></div>
    <div class="m">Gap R <span class="v {gap_r_cls}">{gap_R}</span></div>
    <div class="m">Left <span class="v">{cls_L}</span></div>
    <div class="m">Right <span class="v">{cls_R}</span></div>
    {ian_metrics}
    <div style="margin-left:auto;color:#333355;font-size:.60rem">
      Left-drag=rotate · Scroll=zoom · Legend click=toggle · Dbl-click=isolate
    </div>
  </div>

  <div class="legend-strip">
    <div class="item"><div class="swatch" style="background:#ff3333"></div>TP Left (43)</div>
    <div class="item"><div class="swatch" style="background:#00d4ff"></div>TP Right (44)</div>
    <div class="item"><div class="swatch" style="background:#ff8c00"></div>Sacrum</div>
    <div class="item"><div class="swatch" style="background:#8855cc"></div>Arcus</div>
    <div class="item"><div class="swatch" style="background:#e8c84a"></div>Spinous</div>
    <div class="item"><div class="swatch" style="background:#66ccaa"></div>Articular</div>
    <div class="item"><div class="swatch" style="background:#6699cc;opacity:.5"></div>Corpus</div>
    <div class="item"><div class="swatch" style="background:#ffcc44"></div>IVD (spine)</div>
    <div class="item"><div class="swatch" style="background:#ffe28a"></div>IVD (vert)</div>
    <div class="item"><div class="swatch" style="background:#1e6fa8;opacity:.6"></div>L1-L6</div>
    <div class="item"><div class="swatch" style="background:#00ffb3;opacity:.4"></div>Canal</div>
    <div class="item"><div class="swatch" style="background:#ffe066;opacity:.7"></div>Cord</div>
    <div class="item"><div class="swatch" style="background:#00e6b4;opacity:.3"></div>TV plane</div>
  </div>

  <div id="plot-container">{plotly_div}</div>

  <script>
    const VIEWS = {{
      oblique:   {{ eye:{{x:1.6,y:0.8,z:0.4}},  up:{{x:0,y:0,z:1}} }},
      lateral:   {{ eye:{{x:2.2,y:0.0,z:0.0}},  up:{{x:0,y:0,z:1}} }},
      posterior: {{ eye:{{x:0.0,y:2.2,z:0.0}},  up:{{x:0,y:0,z:1}} }},
      anterior:  {{ eye:{{x:0.0,y:-2.2,z:0.0}}, up:{{x:0,y:0,z:1}} }},
      axial:     {{ eye:{{x:0.0,y:0.0,z:2.8}},  up:{{x:0,y:1,z:0}} }},
    }};
    function setView(name) {{
      const pd = document.querySelector('#plot-container .js-plotly-plot');
      if (!pd) return;
      const v = VIEWS[name];
      Plotly.relayout(pd, {{'scene.camera.eye':v.eye,'scene.camera.up':v.up}});
      document.querySelectorAll('.toolbar button')
              .forEach(b => b.classList.remove('active'));
      const btn = document.getElementById('btn-'+name);
      if (btn) btn.classList.add('active');
    }}
    window.addEventListener('resize', () => {{
      const pd = document.querySelector('#plot-container .js-plotly-plot');
      if (pd) Plotly.Plots.resize(pd);
    }});
  </script>
</body>
</html>"""

# ============================================================================
# SAVE HTML
# ============================================================================

def save_html(fig, study_id, output_dir, castellvi, tv_name, cls_L, cls_R,
              span_L, span_R, dist_L, dist_R, uncertainty_row):
    from plotly.io import to_html

    plotly_div = to_html(
        fig, full_html=False, include_plotlyjs='cdn',
        config=dict(responsive=True, displayModeBar=True,
                    modeBarButtonsToRemove=['toImage'], displaylogo=False),
    )

    def _fmt(v):  return f'{v:.1f} mm' if np.isfinite(v) else 'N/A'
    def _hcls(v): return 'warn' if v >= TP_HEIGHT_MM else 'ok'
    def _gcls(v): return 'crit' if (np.isfinite(v) and v <= CONTACT_DIST_MM) else 'ok'

    ian_badge = ''
    ian_metrics = ''
    if uncertainty_row:
        conf = uncertainty_row.get('l5_s1_confidence', float('nan'))
        if not np.isnan(conf):
            ian_badge = (f'<span class="badge badge-ian">'
                         f'Ian L5-S1: {conf:.3f}</span>')
        ian_metrics = ''.join(
            f'<div class="m">{lbl} <span class="v">'
            f'{uncertainty_row.get(f"{lvl}_confidence", float("nan")):.2f}'
            f'</span></div>'
            for lvl, lbl in zip(IAN_PAN_LEVELS, IAN_PAN_LABELS)
            if not np.isnan(uncertainty_row.get(f'{lvl}_confidence', float('nan')))
        )

    html = HTML_TEMPLATE.format(
        study_id   = study_id,
        castellvi  = castellvi or 'N/A',
        tv_name    = tv_name   or 'N/A',
        cls_L      = cls_L     or 'N/A',
        cls_R      = cls_R     or 'N/A',
        ian_badge  = ian_badge,
        ian_metrics= ian_metrics,
        span_L     = _fmt(span_L), tp_l_cls = _hcls(span_L),
        span_R     = _fmt(span_R), tp_r_cls = _hcls(span_R),
        gap_L      = _fmt(dist_L), gap_l_cls = _gcls(dist_L),
        gap_R      = _fmt(dist_R), gap_r_cls = _gcls(dist_R),
        plotly_div = plotly_div,
    )

    out_path = output_dir / f"{study_id}_3d_spine.html"
    out_path.write_text(html, encoding='utf-8')
    logger.info(f"  Saved → {out_path}  "
                f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive 3D Spine Viewer — '
                    'correct SPINEPS/TSS/VERIDAH labels + 3D LSTV measurements')

    parser.add_argument('--spineps_dir',    required=True)
    parser.add_argument('--totalspine_dir', required=True)
    parser.add_argument('--output_dir',     required=True)

    # Study selection — mirrors 05_visualize_overlay.py
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--study_id', default=None)
    group.add_argument('--all',      action='store_true')

    parser.add_argument('--uncertainty_csv', default=None)
    parser.add_argument('--valid_ids',       default=None)
    parser.add_argument('--top_n',    type=int,   default=None)
    parser.add_argument('--rank_by',  default='l5_s1_confidence')
    parser.add_argument('--lstv_json', default=None)

    parser.add_argument('--step',     type=int,   default=2)
    parser.add_argument('--smooth',   type=float, default=1.5)
    parser.add_argument('--no_tss',   action='store_true',
                        help='Skip TotalSpineSeg label rendering')

    args = parser.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    results_by_id = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                results_by_id = {str(r['study_id']): r for r in json.load(f)}
            logger.info(f"Loaded {len(results_by_id)} LSTV results")

    uncertainty_by_id = {}
    csv_path = Path(args.uncertainty_csv) if args.uncertainty_csv else None
    if csv_path and csv_path.exists():
        df_unc = pd.read_csv(csv_path)
        df_unc['study_id'] = df_unc['study_id'].astype(str)
        uncertainty_by_id  = {r['study_id']: r
                               for r in df_unc.to_dict('records')}
        logger.info(f"Loaded uncertainty for {len(uncertainty_by_id)} studies")

    if args.study_id:
        study_ids = [args.study_id]
        logger.info(f"Single-study mode: {args.study_id}")
    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")
    else:
        if not args.uncertainty_csv or args.top_n is None:
            parser.error("--uncertainty_csv and --top_n required unless "
                         "--all or --study_id")
        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
        study_ids = select_studies(csv_path, args.top_n,
                                   args.rank_by, valid_ids)
        study_ids = [s for s in study_ids if (seg_root / s).is_dir()]
        logger.info(f"Selective mode: {len(study_ids)} studies")

    ok = 0
    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            out = build_3d_figure(
                study_id        = sid,
                spineps_dir     = spineps_dir,
                totalspine_dir  = totalspine_dir,
                step            = args.step,
                smooth          = args.smooth,
                lstv_result     = results_by_id.get(sid),
                uncertainty_row = uncertainty_by_id.get(sid),
                show_tss        = not args.no_tss,
            )
            if out is None:
                continue
            fig, castellvi, tv_name, cls_L, cls_R, \
                span_L, span_R, dist_L, dist_R = out

            save_html(fig, sid, output_dir, castellvi, tv_name, cls_L, cls_R,
                      span_L, span_R, dist_L, dist_R,
                      uncertainty_by_id.get(sid))
            ok += 1

        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"\nDone. {ok}/{len(study_ids)} HTMLs → {output_dir}")


if __name__ == '__main__':
    main()
