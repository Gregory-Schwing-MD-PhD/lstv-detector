#!/usr/bin/env python3
"""
06_visualize_3d.py  —  Interactive 3D Spine Viewer (Morphometrics-JSON Edition)
===================================================================================
Reads pre-computed morphometrics from 05_morphometrics.py JSON output.
NO re-computation of measurements; this script only does:
  1. NIfTI load + isotropic resample (for mesh geometry)
  2. Marching-cubes surface meshing
  3. 3D annotation placement (fixed: offset from volume, not at origin)
  4. HTML assembly

STUDY SELECTION MODES
---------------------
  --study_id ID               single study
  --all                       every study with SPINEPS segmentation
  --rank_by morpho            rank by pathology burden score computed from
                               --morphometrics_json  (preferred for pipeline use)
  --rank_by <csv_column>      rank by a column in --uncertainty_csv
                               (legacy mode — requires --uncertainty_csv + --top_n)

  When using --rank_by morpho:
    --top_n N           renders N most-pathologic studies  (default 5)
    --top_normal N      renders N most-normal studies       (default 1)
    --morphometrics_json PATH   required

  Pathology scoring weights (see pathology_score.py):
    Canal absolute stenosis +3 | relative +1
    Cord severe MSCC +4 | moderate +3 | mild +1
    DHI severe per level +2 | moderate +1
    Spondylolisthesis per level +2
    Vertebral wedge fracture +2 | intervention threshold +3
    Baastrup contact +2 | risk +1
    Facet tropism grade 2 +2 | grade 1 +1
    LFT severe +2 | hypertrophy +1
    Castellvi III/IV +2 | I/II +1

KEY CHANGE from v1:
  All morphometric values come from --morphometrics_json (output of 05_morphometrics.py).
  If JSON is absent, falls back to inline engine computation so the script is still
  independently usable.

ANNOTATION FIX:
  All morphometric text is now rendered in an HTML panel on the right side of the page
  (matching the style of the legend on the left), completely outside the 3D scene.
  The Ian-Pan confidence bar chart has been removed entirely.

Label reference — see 04_detect_lstv.py and morphometrics_engine.py for full tables.
"""

import argparse, json, logging, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import (binary_fill_holes, gaussian_filter,
                            label as cc_label, zoom as ndizoom)
from skimage.measure import marching_cubes

# Engine — import for fallback computation and constants only
from morphometrics_engine import (
    T, ISO_MM, TP_HEIGHT_MM, CONTACT_DIST_MM,
    SP_TP_L, SP_TP_R, SP_SACRUM, SP_CORD, SP_CANAL,
    SP_ARCUS, SP_SPINOUS, SP_SAL, SP_SAR, SP_IAL, SP_IAR,
    SP_CORPUS, SP_ENDPLATE, SP_IVD,
    VD_L1, VD_L2, VD_L3, VD_L4, VD_L5, VD_L6, VD_SAC,
    VD_IVD_BASE, VD_EP_BASE,
    TSS_CORD, TSS_CANAL, TSS_SACRUM,
    VERIDAH_LUMBAR_LABELS, VERIDAH_NAMES,
    LUMBAR_PAIRS, CANAL_SHAPE,
    load_study_masks, run_all_morphometrics,
)

from pathology_score import compute_pathology_score, select_studies_by_morpho

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Annotation geometry ──────────────────────────────────────────────────────

BBOX_HALF = 16

IAN_PAN_LEVELS = ['l1_l2','l2_l3','l3_l4','l4_l5','l5_s1']
IAN_PAN_LABELS = ['L1-L2','L2-L3','L3-L4','L4-L5','L5-S1']
_VALID_SYM     = {'circle','circle-open','cross','diamond',
                  'diamond-open','square','square-open','x'}

# ─── Label colour tables ──────────────────────────────────────────────────────

# (lbl, name, colour, opacity, fill_holes, max_smooth_sigma)
SPINE_LABELS = [
    ( 26, 'Sacrum (spine)',        '#ff8c00', 0.72, True,  1.5),
    ( 41, 'Arcus Vertebrae',       '#8855cc', 0.55, True,  1.5),
    ( 42, 'Spinous Process',       '#e8c84a', 0.75, True,  1.5),
    ( 43, 'TP Left (costal)',      '#ff3333', 0.95, False, 1.0),
    ( 44, 'TP Right (costal)',     '#00d4ff', 0.95, False, 1.0),
    ( 45, 'Sup Articular Left',    '#66ccaa', 0.65, True,  1.5),
    ( 46, 'Sup Articular Right',   '#44aa88', 0.65, True,  1.5),
    ( 47, 'Inf Articular Left',    '#aaddcc', 0.60, True,  1.5),
    ( 48, 'Inf Articular Right',   '#88ccbb', 0.60, True,  1.5),
    ( 49, 'Corpus Border',         '#6699cc', 0.40, True,  1.5),
    ( 60, 'Spinal Cord',           '#ffe066', 0.65, False, 1.2),
    ( 61, 'Spinal Canal',          '#00ffb3', 0.18, False, 1.0),
    ( 62, 'Endplate (merged)',      '#ff6b6b', 0.80, False, 0.6),
    (100, 'IVD (spine, all)',       '#ffcc44', 0.55, True,  1.5),
]

VERIDAH_CERVICAL = {i: (f'C{i}','#557799',0.20) for i in range(1,8)}
VERIDAH_THORACIC = {i+7: (f'T{i+1}','#447766',0.20) for i in range(12)}
VERIDAH_THORACIC[28] = ('T13','#447766',0.20)
VERIDAH_LUMBAR_MAP = {
    20:('L1','#1e6fa8',0.48), 21:('L2','#2389cc',0.48),
    22:('L3','#29a3e8',0.48), 23:('L4','#52bef5',0.50),
    24:('L5','#85d4ff',0.52), 25:('L6','#aae3ff',0.52),
    26:('Sacrum (vert 26)','#ff8c00',0.62),
}
VERIDAH_IVD_COLOURS = {20:'#ffe28a',21:'#ffd060',22:'#ffb830',
                        23:'#ff9900',24:'#ff7700',25:'#ff5500'}
VERIDAH_EP_COLOUR = '#ff8888'

TSS_LABELS = [
    (1,'TSS Cord','#ffe066',0.50),(2,'TSS Canal','#00ffb3',0.14),
    (11,'TSS C1','#88aabb',0.18),(12,'TSS C2','#88aabb',0.18),
    (13,'TSS C3','#88aabb',0.18),(14,'TSS C4','#88aabb',0.18),
    (15,'TSS C5','#88aabb',0.18),(16,'TSS C6','#88aabb',0.18),
    (17,'TSS C7','#88aabb',0.18),
    (21,'TSS T1','#447766',0.18),(22,'TSS T2','#447766',0.18),
    (23,'TSS T3','#447766',0.18),(24,'TSS T4','#447766',0.18),
    (25,'TSS T5','#447766',0.18),(26,'TSS T6','#447766',0.18),
    (27,'TSS T7','#447766',0.18),(28,'TSS T8','#447766',0.18),
    (29,'TSS T9','#447766',0.18),(30,'TSS T10','#447766',0.18),
    (31,'TSS T11','#447766',0.22),(32,'TSS T12','#447766',0.22),
    (41,'TSS L1','#1e6fa8',0.25),(42,'TSS L2','#2389cc',0.25),
    (43,'TSS L3','#29a3e8',0.25),(44,'TSS L4','#52bef5',0.25),
    (45,'TSS L5','#85d4ff',0.28),(50,'TSS Sacrum','#ff8c00',0.65),
    (63,'TSS disc C2-C3','#d4e8ff',0.35),(64,'TSS disc C3-C4','#d4e8ff',0.35),
    (65,'TSS disc C4-C5','#d4e8ff',0.35),(66,'TSS disc C5-C6','#d4e8ff',0.35),
    (67,'TSS disc C6-C7','#d4e8ff',0.35),(71,'TSS disc C7-T1','#d4e8ff',0.35),
    (72,'TSS disc T1-T2','#ffe8aa',0.28),(73,'TSS disc T2-T3','#ffe8aa',0.28),
    (74,'TSS disc T3-T4','#ffe8aa',0.28),(75,'TSS disc T4-T5','#ffe8aa',0.28),
    (76,'TSS disc T5-T6','#ffe8aa',0.28),(77,'TSS disc T6-T7','#ffe8aa',0.28),
    (78,'TSS disc T7-T8','#ffe8aa',0.28),(79,'TSS disc T8-T9','#ffe8aa',0.28),
    (80,'TSS disc T9-T10','#ffe8aa',0.28),(81,'TSS disc T10-T11','#ffe8aa',0.30),
    (82,'TSS disc T11-T12','#ffe28a',0.40),
    (91,'TSS disc T12-L1','#ffd060',0.45),(92,'TSS disc L1-L2','#ffb830',0.50),
    (93,'TSS disc L2-L3','#ff9900',0.50),(94,'TSS disc L3-L4','#ff7700',0.50),
    (95,'TSS disc L4-L5','#ff5500',0.52),(100,'TSS disc L5-S','#ff3300',0.55),
]

# ─── NIfTI + resample ────────────────────────────────────────────────────────

def _load_canonical(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
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

def _voxmm(nii):
    return np.abs(np.array(nii.header.get_zooms()[:3], dtype=float))

def _resample(vol, vox_mm, target=ISO_MM):
    return ndizoom(vol.astype(np.int32), (vox_mm/target).tolist(),
                   order=0, mode='nearest', prefilter=False).astype(np.int32)

# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _centroid(mask):
    c = np.array(np.where(mask))
    return c.mean(axis=1)*ISO_MM if c.size else None

def _min_dist(a, b):
    from scipy.ndimage import distance_transform_edt
    if not a.any() or not b.any(): return float('inf'), None, None
    dt = distance_transform_edt(~b)*ISO_MM
    di = np.where(a, dt, np.inf)
    flat = int(np.argmin(di))
    va = np.array(np.unravel_index(flat, a.shape), dtype=float)
    dist = float(dt[tuple(va.astype(int))])
    cb = np.array(np.where(b), dtype=float)
    d2 = ((cb.T-va)**2).sum(axis=1)
    vb = cb[:, int(np.argmin(d2))]
    return dist, va*ISO_MM, vb*ISO_MM

def _z_range(mask):
    if not mask.any(): return None
    zc = np.where(mask)[2]
    return int(zc.min()), int(zc.max())

def _tp_h(tp):
    if not tp.any(): return 0.0
    zc = np.where(tp)[2]
    return (int(zc.max())-int(zc.min()))*ISO_MM

def _inferiormost_cc(mask, sac=None):
    if not mask.any(): return np.zeros_like(mask, bool)
    lab, n = cc_label(mask)
    if n == 1: return mask.astype(bool)
    sz = None
    if sac is not None and sac.any(): sz = int(np.where(sac)[2].min())
    comps = []
    for i in range(1,n+1):
        c = (lab==i); zc = np.where(c)[2]
        comps.append((float(zc.mean()), int(zc.max()), c))
    comps.sort(key=lambda t: t[0])
    if sz is not None:
        cands = [c for _,zm,c in comps if zm < sz]
        if cands: return cands[0].astype(bool)
    return comps[0][2].astype(bool)

def _isolate_z(mask, z_lo, z_hi, margin=20):
    out = np.zeros_like(mask)
    lo = max(0, z_lo-margin); hi = min(mask.shape[2]-1, z_hi+margin)
    out[:,:,lo:hi+1] = mask[:,:,lo:hi+1]
    return out

# ─── Marching cubes → Mesh3d ──────────────────────────────────────────────────

def mask_to_mesh3d(iso_mask, origin_mm, name, colour, opacity,
                   smooth_sigma=1.5, fill_holes=True):
    if not iso_mask.any(): return None
    m = binary_fill_holes(iso_mask) if fill_holes else iso_mask.copy()
    if not m.any(): return None
    vol = gaussian_filter(m.astype(np.float32), sigma=smooth_sigma)
    vol = np.pad(vol, 1, mode='constant', constant_values=0)
    if vol.max() <= 0.5 or vol.min() >= 0.5: return None
    try:
        verts, faces, _, _ = marching_cubes(
            vol, level=0.5, spacing=(ISO_MM, ISO_MM, ISO_MM))
    except Exception as e:
        logger.warning(f"  MC failed '{name}': {e}"); return None
    verts -= ISO_MM
    verts -= origin_mm[np.newaxis,:]
    return go.Mesh3d(
        x=verts[:,0].tolist(), y=verts[:,1].tolist(), z=verts[:,2].tolist(),
        i=faces[:,0].tolist(), j=faces[:,1].tolist(), k=faces[:,2].tolist(),
        color=colour, opacity=opacity, name=name,
        showlegend=True, flatshading=False,
        lighting=dict(ambient=0.35, diffuse=0.75, specular=0.30,
                      roughness=0.6, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo='name', showscale=False,
    )

# ─── Annotation primitives ────────────────────────────────────────────────────

def _sym(s): return s if s in _VALID_SYM else 'circle'

def ruler_line(p0, p1, colour, name, width=6, dash='solid'):
    return go.Scatter3d(
        x=[p0[0],p1[0]], y=[p0[1],p1[1]], z=[p0[2],p1[2]],
        mode='lines', line=dict(color=colour, width=width, dash=dash),
        name=name, showlegend=True, hoverinfo='name')

def label_point(pos, text, colour, size=10, symbol='circle', showlegend=False):
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=size, color=colour, symbol=_sym(symbol),
                    line=dict(color='white', width=1)),
        text=[text], textposition='top center',
        textfont=dict(size=10, color=colour),
        name=text, showlegend=showlegend, hoverinfo='text')

def midpt(a, b): return (np.array(a)+np.array(b))/2.0

# ─── Per-study TP annotation traces (3D mesh attached) ───────────────────────

def tp_height_ruler_traces(tp_iso, origin_mm, colour, side, span_mm):
    if not tp_iso.any(): return []
    best_x, best_span = tp_iso.shape[0]//2, 0.0
    for x in range(tp_iso.shape[0]):
        col = tp_iso[x]
        if not col.any(): continue
        zc = np.where(col.any(axis=0))[0]
        if zc.size < 2: continue
        sp = (zc.max()-zc.min())*ISO_MM
        if sp > best_span: best_span, best_x = sp, x
    col = tp_iso[best_x]
    if not col.any(): return []
    zc = np.where(col.any(axis=0))[0]
    yc = np.where(col.any(axis=1))[0]
    z_lo, z_hi = int(zc.min()), int(zc.max())
    y_c = int(yc.mean()) if yc.size else tp_iso.shape[1]//2
    def iv(x,y,z): return np.array([x,y,z],float)*ISO_MM-origin_mm
    p_lo = iv(best_x, y_c, z_lo); p_hi = iv(best_x, y_c, z_hi)
    mid  = midpt(p_lo, p_hi)
    flag = '✓' if span_mm < TP_HEIGHT_MM else f'✗ ≥{TP_HEIGHT_MM:.0f}mm→TypeI'
    traces = [ruler_line(p_lo, p_hi, colour, f'Height ruler {side}', width=8)]
    traces.append(label_point(mid, f'{side} TP: {span_mm:.1f}mm  {flag}',
                              colour, size=9, symbol='diamond'))
    off = np.array([5.,0.,0.])
    for pt in (p_lo, p_hi):
        traces.append(ruler_line(pt-off, pt+off, colour, f'Tick {side}', width=4))
    return traces

def gap_ruler_traces(tp_iso, sac_iso, origin_mm, colour, side, dist_mm):
    if not tp_iso.any() or not sac_iso.any(): return []
    _, pt_a, pt_b = _min_dist(tp_iso, sac_iso)
    if pt_a is None: return []
    p_a = pt_a-origin_mm; p_b = pt_b-origin_mm
    mid = midpt(p_a, p_b)
    contact = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    dash  = 'dot' if contact else 'dash'
    clbl  = (f'CONTACT {dist_mm:.1f}mm→P2' if contact
             else f'Gap: {dist_mm:.1f}mm ✓')
    return [ruler_line(p_a, p_b, colour, f'Gap ruler {side}', width=5, dash=dash),
            label_point(mid, f'{side}: {clbl}', colour, size=7, symbol='square')]

def tv_plane_traces(vert_iso, tv_label, origin_mm, tv_name):
    mask = (vert_iso==tv_label)
    if not mask.any(): return []
    zc   = np.where(mask)[2]
    z_mid= int((zc.min()+zc.max())//2)
    xs   = np.linspace(0, vert_iso.shape[0]-1, 12)
    ys   = np.linspace(0, vert_iso.shape[1]-1, 12)
    xg, yg = np.meshgrid(xs, ys)
    zg   = np.full_like(xg, z_mid)
    plane = go.Surface(
        x=xg*ISO_MM-origin_mm[0], y=yg*ISO_MM-origin_mm[1], z=zg*ISO_MM-origin_mm[2],
        colorscale=[[0,'rgba(0,230,180,0.10)'],[1,'rgba(0,230,180,0.10)']],
        showscale=False, opacity=0.18, name=f'TV plane ({tv_name})',
        showlegend=True, hoverinfo='name')
    ctr = _centroid(mask)
    pts = ([label_point(ctr-origin_mm, f'TV: {tv_name}', '#00e6b4',
                        size=14, symbol='cross')]
           if ctr is not None else [])
    return [plane]+pts

def castellvi_contact_traces(tp_L, tp_R, sac_iso, origin_mm,
                              cls_L, cls_R, dist_L, dist_R):
    traces = []
    for tp, side, dist_mm, cls in (
        (tp_L,'Left',dist_L,cls_L),(tp_R,'Right',dist_R,cls_R)):
        if not (tp.any() and sac_iso.any()): continue
        if not (np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM): continue
        _, pt_a, _ = _min_dist(tp, sac_iso)
        if pt_a is None: continue
        p   = pt_a-origin_mm
        col = '#ff2222' if 'III' in (cls or '') else '#ff9900'
        traces.append(go.Scatter3d(
            x=[p[0]],y=[p[1]],z=[p[2]], mode='markers+text',
            marker=dict(size=20, color=col, opacity=0.85, symbol='circle',
                        line=dict(color='white',width=2)),
            text=[f'{side}: {cls}'], textposition='middle right',
            textfont=dict(size=13, color=col),
            name=f'Contact {side} ({cls})', showlegend=True, hoverinfo='text'))
    return traces

def cord_compression_traces(cord_profile: Optional[dict], cord_iso, origin_mm):
    if cord_profile is None or not cord_iso.any(): return []
    traces = []
    slices = cord_profile.get('slices', [])
    cols   = {'Mild':'#f0a500','Moderate':'#e07800','Severe':'#e02020'}
    for s in slices:
        if not s.get('flagged'): continue
        z_vox = int(s['z_mm'] / ISO_MM)
        if z_vox >= cord_iso.shape[2]: continue
        sl = cord_iso[:,:,z_vox]
        if not sl.any(): continue
        yc = np.where(sl)[0]; xc = np.where(sl)[1]
        x_mm = float(xc.mean()+2)*ISO_MM - origin_mm[0]
        y_mm = float(yc.mean())*ISO_MM   - origin_mm[1]
        z_mm = s['z_mm'] - origin_mm[2]
        col  = cols.get(s['cls'], '#ffaa00')
        traces.append(go.Scatter3d(
            x=[x_mm], y=[y_mm], z=[z_mm], mode='markers',
            marker=dict(size=8, color=col, opacity=0.7,
                        symbol='circle', line=dict(color='white',width=1)),
            name=f"Cord {s['cls']} z={s['z_mm']:.0f}mm MSCC={s['mscc']:.2f}",
            showlegend=True, hoverinfo='name'))
    return traces


# ─── HTML metrics panel builder ───────────────────────────────────────────────

def build_metrics_panel_html(metrics: dict,
                              span_L: float, span_R: float,
                              dist_L: float, dist_R: float,
                              castellvi: str, cls_L: str, cls_R: str,
                              tv_name: str,
                              uncertainty_row: Optional[dict],
                              pathology_score: Optional[float] = None) -> str:
    def row(label: str, value: str, css_cls: str = 'ok') -> str:
        return (f'<div class="pr">'
                f'<span class="pk">{label}</span>'
                f'<span class="pv {css_cls}">{value}</span>'
                f'</div>')

    def section(title: str) -> str:
        return f'<div class="ps">{title}</div>'

    lines = []

    # ── Pathology burden score ─────────────────────────────────────────────────
    if pathology_score is not None:
        score_cls = ('cr' if pathology_score >= 8 else
                     'wn' if pathology_score >= 3 else 'ok')
        lines.append(section('Pathology Burden'))
        lines.append(row('Score', f'{pathology_score:.0f}', score_cls))

    # ── LSTV ──────────────────────────────────────────────────────────────────
    lines.append(section('LSTV'))
    lines.append(row('TV', tv_name or 'N/A'))
    lines.append(row('Castellvi', castellvi or 'N/A'))
    tpl_cls = 'wn' if span_L >= TP_HEIGHT_MM else 'ok'
    tpr_cls = 'wn' if span_R >= TP_HEIGHT_MM else 'ok'
    lines.append(row('TP Left',  f'{span_L:.1f} mm', tpl_cls))
    lines.append(row('TP Right', f'{span_R:.1f} mm', tpr_cls))
    gl_cls = 'cr' if (np.isfinite(dist_L) and dist_L <= CONTACT_DIST_MM) else 'ok'
    gr_cls = 'cr' if (np.isfinite(dist_R) and dist_R <= CONTACT_DIST_MM) else 'ok'
    lines.append(row('Gap Left',  f'{dist_L:.1f} mm' if np.isfinite(dist_L) else 'N/A', gl_cls))
    lines.append(row('Gap Right', f'{dist_R:.1f} mm' if np.isfinite(dist_R) else 'N/A', gr_cls))
    lines.append(row('Class Left',  cls_L or 'N/A'))
    lines.append(row('Class Right', cls_R or 'N/A'))

    # ── Canal ─────────────────────────────────────────────────────────────────
    lines.append(section('Canal Stenosis'))
    ap     = metrics.get('canal_ap_mm')
    ap_cls = metrics.get('canal_ap_class', 'N/A')
    dsca   = metrics.get('canal_dsca_mm2')
    ap_col = ('cr' if 'Absolute' in (ap_cls or '') else
              'wn' if 'Relative' in (ap_cls or '') else 'ok')
    if ap:
        lines.append(row('AP (overall)', f'{ap:.1f} mm — {ap_cls}', ap_col))
    if dsca:
        lines.append(row('DSCA≈', f'{dsca:.0f} mm²', ap_col))

    # ── Per-level canal AP ────────────────────────────────────────────────────
    level_ap_found = False
    for _, _, up_n, lo_n in LUMBAR_PAIRS:
        lk  = f'{up_n}_{lo_n}_level_ap_mm'
        lck = f'{up_n}_{lo_n}_level_ap_class'
        ap_v   = metrics.get(lk)
        ap_c   = metrics.get(lck, 'N/A')
        if ap_v is None: continue
        if not level_ap_found:
            lines.append(section('Level AP'))
            level_ap_found = True
        col = ('cr' if 'Absolute' in (ap_c or '') else
               'wn' if 'Relative' in (ap_c or '') else 'ok')
        lines.append(row(f'{up_n}-{lo_n}', f'{ap_v:.1f} mm ({ap_c})', col))

    # ── Cord compression ──────────────────────────────────────────────────────
    lines.append(section('Cord Compression'))
    cp = metrics.get('cord_compression_profile')
    if cp:
        mc  = cp.get('max_mscc', 0)
        cc  = cp.get('classification', 'N/A')
        fc  = cp.get('flagged_count', 0)
        col = ('cr' if cc == 'Severe' else
               'wn' if cc in ('Moderate', 'Mild') else 'ok')
        lines.append(row('Max MSCC', f'{mc:.2f} — {cc}', col))
        lines.append(row('Flagged slices', str(fc), col if fc > 0 else 'ok'))
    else:
        lines.append(row('Profile', 'N/A', 'pm'))

    # ── Disc DHI ──────────────────────────────────────────────────────────────
    lines.append(section('Disc DHI'))
    for _, _, up_n, lo_n in LUMBAR_PAIRS:
        dhi = metrics.get(f'{up_n}_{lo_n}_dhi_pct')
        if dhi is None: continue
        col = ('cr' if dhi < T.DHI_SEVERE_PCT else
               'wn' if dhi < T.DHI_MODERATE_PCT else 'ok')
        lines.append(row(f'{up_n}-{lo_n}', f'{dhi:.0f}%', col))

    # ── Spondylolisthesis ─────────────────────────────────────────────────────
    lines.append(section('Spondylolisthesis'))
    for _, _, up_n, lo_n in LUMBAR_PAIRS:
        trans = metrics.get(f'{up_n}_{lo_n}_sagittal_translation_mm')
        if trans is None: continue
        col  = 'cr' if trans >= T.SPONDYLO_MM else 'ok'
        flag = ' ✗' if trans >= T.SPONDYLO_MM else ''
        lines.append(row(f'{up_n}-{lo_n}', f'{trans:.1f} mm{flag}', col))

    # ── Ligamentum flavum ─────────────────────────────────────────────────────
    lines.append(section('Lig. Flavum'))
    lft     = metrics.get('lft_proxy_mm')
    lft_cls = metrics.get('lft_class', 'N/A')
    if lft is not None:
        col = ('cr' if lft > T.LFT_SEVERE_MM else
               'wn' if lft > T.LFT_NORMAL_MM else 'ok')
        lines.append(row('LFT proxy', f'{lft:.1f} mm — {lft_cls}', col))
    else:
        lines.append(row('LFT proxy', 'N/A', 'pm'))

    # ── Baastrup ──────────────────────────────────────────────────────────────
    lines.append(section('Baastrup'))
    gap     = metrics.get('min_inter_process_gap_mm')
    contact = metrics.get('baastrup_contact', False)
    risk    = metrics.get('baastrup_risk', False)
    if gap is not None:
        col = 'cr' if contact else 'wn' if risk else 'ok'
        lbl = 'CONTACT' if contact else ('Risk zone' if risk else 'Normal')
        lines.append(row('Min gap', f'{gap:.1f} mm — {lbl}', col))
    else:
        lines.append(row('Min gap', 'N/A', 'pm'))

    # ── Facet tropism ─────────────────────────────────────────────────────────
    lines.append(section('Facet Tropism'))
    trop     = metrics.get('facet_tropism_deg')
    ft_grade = metrics.get('facet_tropism_grade', 'N/A')
    fa_l     = metrics.get('facet_angle_l_deg')
    fa_r     = metrics.get('facet_angle_r_deg')
    if trop is not None:
        col = ('cr' if trop >= T.TROPISM_SEVERE_DEG else
               'wn' if trop >= T.TROPISM_NORMAL_DEG else 'ok')
        grade_short = ft_grade.split('(')[0].strip() if ft_grade else 'N/A'
        lines.append(row('Asymmetry', f'{trop:.1f}° — {grade_short}', col))
    if fa_l is not None:
        lines.append(row('Angle Left',  f'{fa_l:.1f}°'))
    if fa_r is not None:
        lines.append(row('Angle Right', f'{fa_r:.1f}°'))

    # ── Foraminal ─────────────────────────────────────────────────────────────
    lines.append(section('Foraminal'))
    for _, _, up_n, lo_n in LUMBAR_PAIRS:
        fl  = metrics.get(f'{up_n}_{lo_n}_foraminal_class_L')
        fr  = metrics.get(f'{up_n}_{lo_n}_foraminal_class_R')
        if fl or fr:
            left_col  = 'cr' if fl and 'Grade 3' in fl else 'wn' if fl and ('Grade 1' in fl or 'Grade 2' in fl) else 'ok'
            right_col = 'cr' if fr and 'Grade 3' in fr else 'wn' if fr and ('Grade 1' in fr or 'Grade 2' in fr) else 'ok'
            left_lbl  = (fl or 'N/A').replace('Normal (','').replace(')','')
            right_lbl = (fr or 'N/A').replace('Normal (','').replace(')','')
            lines.append(row(f'{up_n}-{lo_n} L', left_lbl,  left_col))
            lines.append(row(f'{up_n}-{lo_n} R', right_lbl, right_col))

    # ── Uncertainty (Ian-Pan) ─────────────────────────────────────────────────
    if uncertainty_row:
        has_any = any(
            not np.isnan(uncertainty_row.get(f'{lvl}_confidence', float('nan')))
            for lvl in IAN_PAN_LEVELS)
        if has_any:
            lines.append(section('Ian-Pan Confidence'))
            for lvl, lbl in zip(IAN_PAN_LEVELS, IAN_PAN_LABELS):
                v = uncertainty_row.get(f'{lvl}_confidence', float('nan'))
                if not np.isnan(v):
                    lines.append(row(lbl, f'{v:.3f}', 'pm'))

    return '\n'.join(lines)


# ─── Main per-study builder ───────────────────────────────────────────────────

def build_3d_figure(study_id: str, spineps_dir: Path, totalspine_dir: Path,
                    smooth: float = 1.5, show_tss: bool = True,
                    lstv_result: Optional[dict] = None,
                    morphometrics: Optional[dict] = None,
                    uncertainty_row: Optional[dict] = None):
    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_path = seg_dir / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path  = seg_dir / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path   = (totalspine_dir / study_id / 'sagittal'
                  / f"{study_id}_sagittal_labeled.nii.gz")

    def _load(p, tag):
        if not p.exists(): logger.warning(f"  Missing: {p.name}"); return None,None
        try: return _load_canonical(p)
        except Exception as e: logger.warning(f"  {tag}: {e}"); return None,None

    sag_sp,  nii_ref = _load(spine_path, 'seg-spine_msk')
    sag_vert, _      = _load(vert_path,  'seg-vert_msk')
    sag_tss,  _      = _load(tss_path,   'TSS sagittal')
    if sag_sp is None or sag_vert is None:
        logger.error(f"[{study_id}] Missing required masks"); return None

    vox_mm   = _voxmm(nii_ref)
    sp_iso   = _resample(sag_sp.astype(np.int32), vox_mm)
    vert_iso = _resample(sag_vert.astype(np.int32), vox_mm)
    tss_iso  = (_resample(sag_tss.astype(np.int32), _voxmm(_load(tss_path,'tss')[1] or nii_ref))
                if sag_tss is not None else None)

    sp_labels   = set(np.unique(sp_iso).tolist())-{0}
    vert_labels = set(np.unique(vert_iso).tolist())-{0}
    tss_labels  = (set(np.unique(tss_iso).tolist())-{0} if tss_iso is not None else set())

    if morphometrics is None:
        logger.info(f"  [{study_id}] No pre-computed morphometrics — computing inline")
        from morphometrics_engine import MaskSet as _MS
        ms = _MS(study_id=study_id, sp_iso=sp_iso, vert_iso=vert_iso,
                 tss_iso=tss_iso, sp_labels=sp_labels,
                 vert_labels=vert_labels, tss_labels=tss_labels)
        res = run_all_morphometrics(ms)
        metrics = res.to_dict()
        if res.cord_compression_profile:
            metrics['cord_compression_profile'] = res.cord_compression_profile
    else:
        metrics = morphometrics

    col_mask  = vert_iso > 0
    origin_mm = (_centroid(col_mask)
                 if col_mask.any()
                 else np.array(sp_iso.shape,float)/2.0*ISO_MM)

    if tss_iso is not None and (tss_iso==TSS_SACRUM).any():
        sac_iso = (tss_iso==TSS_SACRUM)
    elif (sp_iso==SP_SACRUM).any():
        sac_iso = (sp_iso==SP_SACRUM)
    else:
        sac_iso = np.zeros(sp_iso.shape, bool)

    tv_label, tv_name = None, 'N/A'
    for cand in VERIDAH_LUMBAR_LABELS:
        if cand in vert_labels:
            tv_label = cand
            tv_name  = VERIDAH_NAMES.get(cand, str(cand))
            break
    if lstv_result:
        det_tv = lstv_result.get('details',{}).get('tv_name')
        if det_tv: tv_name = det_tv

    tp_L_full = (sp_iso==SP_TP_L); tp_R_full = (sp_iso==SP_TP_R)
    if tv_label is not None:
        tv_zr = _z_range(vert_iso==tv_label)
        if tv_zr:
            tp_L_iso = _isolate_z(tp_L_full, *tv_zr)
            tp_R_iso = _isolate_z(tp_R_full, *tv_zr)
            if not tp_L_iso.any(): tp_L_iso = tp_L_full
            if not tp_R_iso.any(): tp_R_iso = tp_R_full
        else:
            tp_L_iso = tp_L_full; tp_R_iso = tp_R_full
    else:
        tp_L_iso = tp_L_full; tp_R_iso = tp_R_full

    tp_L = _inferiormost_cc(tp_L_iso, sac_iso if sac_iso.any() else None)
    tp_R = _inferiormost_cc(tp_R_iso, sac_iso if sac_iso.any() else None)
    span_L = _tp_h(tp_L); span_R = _tp_h(tp_R)
    dist_L = _min_dist(tp_L, sac_iso)[0]; dist_R = _min_dist(tp_R, sac_iso)[0]

    castellvi = 'N/A'; cls_L = 'N/A'; cls_R = 'N/A'
    if lstv_result:
        castellvi = lstv_result.get('castellvi_type') or 'None'
        cls_L     = lstv_result.get('left',  {}).get('classification','N/A')
        cls_R     = lstv_result.get('right', {}).get('classification','N/A')

    traces = []

    for lbl, name, col, op, fh, max_sig in SPINE_LABELS:
        if lbl not in sp_labels: continue
        mask = (tp_L if lbl==SP_TP_L else tp_R if lbl==SP_TP_R else (sp_iso==lbl))
        if not mask.any(): continue
        eff_sig = min(smooth, max_sig) if max_sig < 1.0 else smooth
        t = mask_to_mesh3d(mask, origin_mm, name, col, op, eff_sig, fh)
        if t: traces.append(t)

    all_vd = {**VERIDAH_CERVICAL, **VERIDAH_THORACIC, **VERIDAH_LUMBAR_MAP}
    for lbl,(name,col,op) in sorted(all_vd.items()):
        if lbl not in vert_labels: continue
        t = mask_to_mesh3d(vert_iso==lbl, origin_mm, name, col, op, smooth, True)
        if t: traces.append(t)

    for base, col in VERIDAH_IVD_COLOURS.items():
        ivd_lbl = VD_IVD_BASE+base
        if ivd_lbl not in vert_labels: continue
        name = f'IVD below {VERIDAH_NAMES.get(base,str(base))}'
        t = mask_to_mesh3d(vert_iso==ivd_lbl, origin_mm, name, col, 0.55, smooth, True)
        if t: traces.append(t)

    for base in VERIDAH_IVD_COLOURS:
        ep_lbl = VD_EP_BASE+base
        if ep_lbl not in vert_labels: continue
        name = f'Endplate {VERIDAH_NAMES.get(base,str(base))}'
        t = mask_to_mesh3d(vert_iso==ep_lbl, origin_mm, name,
                           VERIDAH_EP_COLOUR, 0.75, 0.6, False)
        if t: traces.append(t)

    if show_tss and tss_iso is not None:
        for lbl, name, col, op in TSS_LABELS:
            if lbl not in tss_labels: continue
            is_thin = lbl in (1,2)
            t = mask_to_mesh3d(tss_iso==lbl, origin_mm, name, col, op,
                               0.8 if is_thin else smooth, not is_thin)
            if t: traces.append(t)

    if not any(isinstance(tr, go.Mesh3d) for tr in traces):
        logger.error(f"[{study_id}] Zero meshes"); return None

    if tv_label is not None:
        traces += tv_plane_traces(vert_iso, tv_label, origin_mm, tv_name)
    traces += tp_height_ruler_traces(tp_L, origin_mm, '#ff3333', 'Left',  span_L)
    traces += tp_height_ruler_traces(tp_R, origin_mm, '#00d4ff', 'Right', span_R)
    traces += gap_ruler_traces(tp_L, sac_iso, origin_mm, '#ff8800', 'Left',  dist_L)
    traces += gap_ruler_traces(tp_R, sac_iso, origin_mm, '#00aaff', 'Right', dist_R)
    traces += castellvi_contact_traces(tp_L,tp_R,sac_iso,origin_mm,
                                       cls_L,cls_R,dist_L,dist_R)

    cord_iso = (sp_iso==SP_CORD) if SP_CORD in sp_labels else \
               ((tss_iso==TSS_CORD) if (tss_iso is not None and TSS_CORD in tss_labels)
                else np.zeros(sp_iso.shape, bool))
    cp = metrics.get('cord_compression_profile')
    if cord_iso.any():
        traces += cord_compression_traces(cp, cord_iso, origin_mm)

    ap_mm       = metrics.get('canal_ap_mm')
    ap_cls_name = metrics.get('canal_ap_class','N/A')
    trop        = metrics.get('facet_tropism_deg')
    cord_cls    = (cp.get('classification','N/A') if cp else 'N/A')

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"<b>{study_id}</b> · Castellvi: <b>{castellvi}</b>"
                  f" · TV: <b>{tv_name}</b>"
                  f" · L:<b>{cls_L}</b> R:<b>{cls_R}</b>"
                  f" · Canal:<b>{ap_cls_name}</b>"
                  f" · Cord:<b>{cord_cls}</b>"
                  f" · FT:<b>{trop:.1f}°</b>" if trop else
                  f"<b>{study_id}</b> · Castellvi: <b>{castellvi}</b>"),
            font=dict(size=12, color='#e8e8f0'), x=0.01),
        paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a',
        scene=dict(
            bgcolor='#0d0d1a',
            xaxis=dict(title='X',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            yaxis=dict(title='Y',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            zaxis=dict(title='Z (SI)',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6,y=0.0,z=0.3),up=dict(x=0,y=0,z=1))),
        legend=dict(font=dict(color='#e8e8f0',size=9),
                    bgcolor='rgba(13,13,26,0.85)',
                    bordercolor='#2a2a4a', borderwidth=1,
                    x=0.01, y=0.98, itemsizing='constant'),
        margin=dict(l=0,r=0,t=40,b=0),
    )
    return fig, castellvi, tv_name, cls_L, cls_R, span_L, span_R, dist_L, dist_R, metrics


# ─── HTML template ────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Spine — {study_id}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0d0d1a;--sf:#13132a;--bd:#2a2a4a;--tx:#e8e8f0;--mu:#6666aa}}
html,body{{background:var(--bg);color:var(--tx);font-family:'JetBrains Mono',monospace;
           height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:5px 10px;
        border-bottom:1px solid var(--bd);background:var(--sf);flex-shrink:0}}
h1{{font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700}}
.b{{display:inline-block;padding:2px 7px;border-radius:16px;
    font-size:.62rem;font-weight:600}}
.bs{{background:#2a2a4a;color:var(--mu)}} .bc{{background:#ff8c00;color:#0d0d1a}}
.bt{{background:#1e6fa8;color:#fff}} .bL{{background:#cc2222;color:#fff}}
.bR{{background:#006688;color:#fff}} .bi{{background:#1a3a2a;color:#2dc653;border:1px solid #2dc653}}
.bw{{background:#553300;color:#ffaa44;border:1px solid #ffaa44}}
.be{{background:#330011;color:#ff4466;border:1px solid #ff4466}}
.bc2{{background:#1a1a3a;color:#8888ff;border:1px solid #4444aa}}
.bscore{{background:#1a1a3a;color:#ccccff;border:1px solid #5555cc;font-weight:700}}
.tb{{display:flex;gap:5px;align-items:center;margin-left:auto}}
.tb span{{font-size:.57rem;color:var(--mu);text-transform:uppercase}}
button{{background:var(--bg);border:1px solid var(--bd);color:var(--tx);
        font-family:inherit;font-size:.62rem;padding:2px 8px;border-radius:4px;cursor:pointer}}
button:hover{{background:var(--bd)}} button.on{{background:#3a86ff;border-color:#3a86ff;color:#fff}}
.mt{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:3px 10px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.62rem}}
.m{{display:flex;align-items:center;gap:3px;color:var(--mu)}}
.v{{color:var(--tx);font-weight:600}} .ok{{color:#2dc653!important}}
.wn{{color:#ff8800!important}} .cr{{color:#ff3333!important}}
.main-row{{display:flex;flex:1;min-height:0;overflow:hidden}}
.lg{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:3px 10px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.60rem}}
.li{{display:flex;align-items:center;gap:3px;color:var(--mu)}}
.sw{{width:9px;height:9px;border-radius:2px;flex-shrink:0}}
.note{{font-size:.55rem;color:#444;margin-left:auto}}
#pl{{flex:1;min-width:0;min-height:0;overflow:hidden}}
#pl .js-plotly-plot,#pl .plot-container{{height:100%!important}}
#metrics-panel{{
  width:210px;flex-shrink:0;
  background:rgba(13,13,26,0.92);
  border-left:1px solid var(--bd);
  overflow-y:auto;overflow-x:hidden;
  padding:6px 0 12px 0;
  font-size:.60rem;
  scrollbar-width:thin;scrollbar-color:#2a2a4a var(--bg);
}}
#metrics-panel::-webkit-scrollbar{{width:4px}}
#metrics-panel::-webkit-scrollbar-track{{background:var(--bg)}}
#metrics-panel::-webkit-scrollbar-thumb{{background:#2a2a4a;border-radius:2px}}
.ps{{
  font-family:'Syne',sans-serif;font-size:.57rem;font-weight:700;
  color:#6666aa;text-transform:uppercase;letter-spacing:.06em;
  padding:5px 8px 2px 8px;margin-top:4px;
  border-top:1px solid #1a1a3a;
}}
.ps:first-child{{border-top:none;margin-top:0}}
.pr{{display:flex;justify-content:space-between;align-items:baseline;
     padding:1px 8px;gap:4px}}
.pk{{color:#6666aa;white-space:nowrap;flex-shrink:0}}
.pv{{text-align:right;color:var(--tx);font-weight:600;word-break:break-word}}
.pv.ok{{color:#2dc653}} .pv.wn{{color:#ff8800}} .pv.cr{{color:#ff3333}}
.pv.pm{{color:#8888aa;font-weight:400}}
</style></head><body>
<header><h1>3D SPINE</h1>
  <span class="b bs">{study_id}</span>
  <span class="b bc">Castellvi:{castellvi}</span>
  <span class="b bt">TV:{tv_name}</span>
  <span class="b bL">L:{cls_L}</span>
  <span class="b bR">R:{cls_R}</span>
  <span class="b {canal_badge}">Canal:{canal_class}</span>
  <span class="b {cord_badge}">Cord:{cord_label}</span>
  <span class="b {ft_badge}">FT:{ft_label}</span>
  <span class="b {bstp_badge}">Baastrup:{bstp_label}</span>
  <span class="b bscore">Score:{pathology_score}</span>
  {ian_badge}
  <div class="tb"><span>View</span>
    <button onclick="sv('oblique')"   id="b-oblique"   class="on">Oblique</button>
    <button onclick="sv('lateral')"   id="b-lateral">Lat</button>
    <button onclick="sv('posterior')" id="b-posterior">Post</button>
    <button onclick="sv('anterior')"  id="b-anterior">Ant</button>
    <button onclick="sv('axial')"     id="b-axial">Axial</button>
  </div></header>
<div class="mt">
  <div class="m">TP-L <span class="v {tpl_c}">{span_L}</span></div>
  <div class="m">TP-R <span class="v {tpr_c}">{span_R}</span></div>
  <div class="m">Gap-L <span class="v {gl_c}">{gap_L}</span></div>
  <div class="m">Gap-R <span class="v {gr_c}">{gap_R}</span></div>
  <div class="m">AP <span class="v {ap_c}">{ap_disp}</span></div>
  <div class="m">MSCC <span class="v {mscc_c}">{mscc_disp}</span></div>
  <div class="m">LFT <span class="v {lft_c}">{lft_disp}</span></div>
  <div class="m">FT <span class="v {ft_c}">{ft_disp}</span></div>
  {dhi_row}
  <div class="note">drag=rotate · scroll=zoom · legend=toggle</div>
</div>
<div class="lg">
  <div class="li"><div class="sw" style="background:#ff3333"></div>TP-L</div>
  <div class="li"><div class="sw" style="background:#00d4ff"></div>TP-R</div>
  <div class="li"><div class="sw" style="background:#ff8c00"></div>Sacrum</div>
  <div class="li"><div class="sw" style="background:#8855cc"></div>Arcus</div>
  <div class="li"><div class="sw" style="background:#e8c84a"></div>Spinous</div>
  <div class="li"><div class="sw" style="background:#ffe066;opacity:.8"></div>Cord</div>
  <div class="li"><div class="sw" style="background:#00ffb3;opacity:.5"></div>Canal</div>
  <div class="li"><div class="sw" style="background:#ff6b6b"></div>Endplate(merged)</div>
  <div class="li"><div class="sw" style="background:#ff8888"></div>Endplate(per-vert)</div>
  <div class="li"><div class="sw" style="background:#e02020"></div>Cord severe</div>
  <div class="li"><div class="sw" style="background:#e07800"></div>Cord moderate</div>
  <div class="li"><div class="sw" style="background:#f0a500"></div>Cord mild</div>
  <div class="li"><div class="sw" style="background:#00e6b4;opacity:.4"></div>TV plane</div>
</div>
<div class="main-row">
  <div id="pl">{plotly_div}</div>
  <div id="metrics-panel">
    {metrics_panel_html}
  </div>
</div>
<script>
const V={{
  oblique: {{eye:{{x:1.6,y:0.8,z:0.4}},up:{{x:0,y:0,z:1}}}},
  lateral: {{eye:{{x:2.4,y:0.0,z:0.0}},up:{{x:0,y:0,z:1}}}},
  posterior:{{eye:{{x:0.0,y:2.4,z:0.0}},up:{{x:0,y:0,z:1}}}},
  anterior: {{eye:{{x:0.0,y:-2.4,z:0.0}},up:{{x:0,y:0,z:1}}}},
  axial:   {{eye:{{x:0.0,y:0.0,z:3.0}},up:{{x:0,y:1,z:0}}}},
}};
function sv(n){{
  const pd=document.querySelector('#pl .js-plotly-plot');
  if(!pd)return;
  Plotly.relayout(pd,{{'scene.camera.eye':V[n].eye,'scene.camera.up':V[n].up}});
  document.querySelectorAll('.tb button').forEach(b=>b.classList.remove('on'));
  const b=document.getElementById('b-'+n); if(b)b.classList.add('on');
}}
window.addEventListener('resize',()=>{{
  const pd=document.querySelector('#pl .js-plotly-plot');
  if(pd)Plotly.Plots.resize(pd);
}});
</script></body></html>"""


def save_html(fig, study_id: str, output_dir: Path,
              castellvi: str, tv_name: str, cls_L: str, cls_R: str,
              span_L: float, span_R: float, dist_L: float, dist_R: float,
              metrics: dict, uncertainty_row: Optional[dict],
              pathology_score: Optional[float] = None) -> Path:
    from plotly.io import to_html
    plotly_div = to_html(fig, full_html=False, include_plotlyjs='cdn',
                         config=dict(responsive=True, displayModeBar=True,
                                     displaylogo=False))
    def _f(v): return f'{v:.1f} mm' if (v is not None and np.isfinite(v)) else 'N/A'
    def _f2(v): return f'{v:.2f}' if (v is not None and np.isfinite(v)) else 'N/A'
    def _hc(v): return 'wn' if v >= TP_HEIGHT_MM else 'ok'
    def _gc(v): return 'cr' if (np.isfinite(v) and v <= CONTACT_DIST_MM) else 'ok'

    ap    = metrics.get('canal_ap_mm')
    ap_cls= metrics.get('canal_ap_class','N/A')
    mscc  = (metrics.get('cord_compression_profile',{}).get('max_mscc')
             if metrics.get('cord_compression_profile')
             else metrics.get('mscc_proxy'))
    cord_cls = (metrics.get('cord_compression_profile',{}).get('classification','N/A')
                if metrics.get('cord_compression_profile') else 'N/A')
    lft   = metrics.get('lft_proxy_mm')
    lft_cls= metrics.get('lft_class','N/A')
    trop  = metrics.get('facet_tropism_deg')
    ft_grade= metrics.get('facet_tropism_grade','N/A')

    def _bcls(cls):
        return ('be' if 'Absolute' in (cls or '') or 'Severe' in (cls or '') or 'Grade 2' in (cls or '') else
                'bw' if 'Relative' in (cls or '') or 'Moderate' in (cls or '') or 'Grade 1' in (cls or '') else 'bi')
    def _col(v, crit_fn, warn_fn=None):
        if v is None: return 'ok'
        if crit_fn(v): return 'cr'
        if warn_fn and warn_fn(v): return 'wn'
        return 'ok'

    dhi_row = ''
    for _,_,up_n,lo_n in LUMBAR_PAIRS:
        d = metrics.get(f'{up_n}_{lo_n}_dhi_pct')
        if d is not None:
            col = 'cr' if d<T.DHI_SEVERE_PCT else 'wn' if d<T.DHI_MODERATE_PCT else 'ok'
            dhi_row += f'<div class="m">DHI {up_n}-{lo_n} <span class="v {col}">{d:.0f}%</span></div>'

    ian_badge = ''
    if uncertainty_row:
        c = uncertainty_row.get('l5_s1_confidence', float('nan'))
        if not np.isnan(c):
            ian_badge = f'<span class="b bc2">Ian L5-S1:{c:.3f}</span>'

    score_disp = f'{pathology_score:.0f}' if pathology_score is not None else 'N/A'

    metrics_panel_html = build_metrics_panel_html(
        metrics, span_L, span_R, dist_L, dist_R,
        castellvi, cls_L, cls_R, tv_name, uncertainty_row, pathology_score)

    html = _HTML.format(
        study_id=study_id,
        castellvi=castellvi or 'N/A',
        tv_name=tv_name or 'N/A',
        cls_L=cls_L or 'N/A', cls_R=cls_R or 'N/A',
        canal_badge=_bcls(ap_cls), canal_class=ap_cls,
        cord_badge=_bcls(cord_cls), cord_label=cord_cls,
        ft_badge=_bcls(ft_grade), ft_label=f'{trop:.1f}°' if trop else 'N/A',
        bstp_badge=('be' if metrics.get('baastrup_contact') else
                    'bw' if metrics.get('baastrup_risk') else 'bi'),
        bstp_label=('CONTACT' if metrics.get('baastrup_contact') else
                    'Risk' if metrics.get('baastrup_risk') else 'None'),
        pathology_score=score_disp,
        ian_badge=ian_badge,
        span_L=_f(span_L), tpl_c=_hc(span_L),
        span_R=_f(span_R), tpr_c=_hc(span_R),
        gap_L=_f(dist_L),  gl_c=_gc(dist_L),
        gap_R=_f(dist_R),  gr_c=_gc(dist_R),
        ap_disp=f'{ap:.1f}mm ({ap_cls})' if ap else 'N/A',
        ap_c=_col(ap, lambda v: v<T.AP_ABSOLUTE_MM, lambda v: v<T.AP_NORMAL_MM),
        mscc_disp=_f2(mscc),
        mscc_c=_col(mscc, lambda v: v>=T.MSCC_MODERATE, lambda v: v>=T.MSCC_MILD) if mscc else 'ok',
        lft_disp=_f(lft), lft_c=_col(lft, lambda v: v>T.LFT_SEVERE_MM,
                                       lambda v: v>T.LFT_NORMAL_MM) if lft else 'ok',
        ft_disp=f'{trop:.1f}°' if trop else 'N/A',
        ft_c=_col(trop, lambda v: v>=T.TROPISM_SEVERE_DEG, lambda v: v>=T.TROPISM_NORMAL_DEG) if trop else 'ok',
        dhi_row=dhi_row,
        metrics_panel_html=metrics_panel_html,
        plotly_div=plotly_div,
    )
    out = output_dir / f"{study_id}_3d_spine.html"
    out.write_text(html, encoding='utf-8')
    logger.info(f"  → {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return out


# ─── Study selection ──────────────────────────────────────────────────────────

def select_studies_legacy(csv_path: Path, top_n: int, rank_by: str, valid_ids) -> list:
    """Original uncertainty-CSV-based selection (kept for backwards compatibility)."""
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids is not None: df = df[df['study_id'].isin(valid_ids)]
    df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top = df.head(top_n)['study_id'].tolist()
    bot = df.tail(top_n)['study_id'].tolist()
    seen, sel = set(), []
    for sid in top+bot:
        if sid not in seen: sel.append(sid); seen.add(sid)
    return sel


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='3D spine visualiser — reads morphometrics JSON for selection + annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Study selection modes:
  --study_id ID               single study
  --all                       all studies with SPINEPS segmentation
  --rank_by morpho            rank by pathology burden score (requires --morphometrics_json)
  --rank_by <csv_col>         rank by uncertainty CSV column (legacy, requires --uncertainty_csv)

Morpho-mode examples:
  # 5 most pathologic + 1 most normal (default)
  --rank_by morpho --morphometrics_json results/morphometrics/morphometrics_all.json

  # 10 most pathologic + 2 most normal
  --rank_by morpho --top_n 10 --top_normal 2 --morphometrics_json ...

  # Specific studies (ignores all selection flags)
  --study_id 12345
""")
    ap.add_argument('--spineps_dir',    required=True)
    ap.add_argument('--totalspine_dir', required=True)
    ap.add_argument('--output_dir',     required=True)

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--study_id', default=None,
        help='Single study ID to render')
    grp.add_argument('--all', action='store_true',
        help='Render every study with SPINEPS segmentation')

    ap.add_argument('--rank_by', default='morpho',
        help='morpho = pathology score from JSON; or a column name in --uncertainty_csv')
    ap.add_argument('--top_n', type=int, default=5,
        help='Number of most-pathologic studies (morpho mode) or top/bottom N (legacy mode)')
    ap.add_argument('--top_normal', type=int, default=1,
        help='Number of most-normal studies to render (morpho mode only)')

    ap.add_argument('--morphometrics_json', default=None,
        help='Path to morphometrics_all.json (required for --rank_by morpho)')
    ap.add_argument('--lstv_json',  default=None,
        help='Path to lstv_results.json for Castellvi annotation')
    ap.add_argument('--uncertainty_csv', default=None,
        help='Path to lstv_uncertainty_metrics.csv (legacy selection mode)')
    ap.add_argument('--valid_ids', default=None,
        help='Path to valid_id.npy (legacy selection mode)')

    ap.add_argument('--smooth', type=float, default=1.5,
        help='Gaussian smoothing sigma for marching cubes (default 1.5)')
    ap.add_argument('--no_tss', action='store_true',
        help='Skip TotalSpineSeg label rendering')
    args = ap.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_root = spineps_dir / 'segmentations'

    # ── Load pre-computed morphometrics ───────────────────────────────────────
    morpho_by_id: Dict[str, dict] = {}
    morpho_all: list = []
    if args.morphometrics_json:
        p = Path(args.morphometrics_json)
        if p.exists():
            with open(p) as f:
                morpho_all = json.load(f)
            morpho_by_id = {str(r.get('study_id','')): r for r in morpho_all
                            if not r.get('error')}
            logger.info(f"Loaded morphometrics for {len(morpho_by_id)} studies")
        else:
            logger.warning(f"morphometrics_json not found: {p}")

    # ── Load LSTV results ─────────────────────────────────────────────────────
    lstv_by_id: Dict[str, dict] = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                lstv_by_id = {str(r['study_id']): r for r in json.load(f)}
            logger.info(f"Loaded {len(lstv_by_id)} LSTV results")

    # ── Load uncertainty (legacy) ─────────────────────────────────────────────
    uncertainty_by_id: Dict[str, dict] = {}
    if args.uncertainty_csv:
        p = Path(args.uncertainty_csv)
        if p.exists():
            df = pd.read_csv(p)
            df['study_id'] = df['study_id'].astype(str)
            uncertainty_by_id = {r['study_id']:r for r in df.to_dict('records')}

    # ── Determine study IDs ───────────────────────────────────────────────────
    score_by_id: Dict[str, float] = {}

    if args.study_id:
        study_ids = [args.study_id]
        logger.info(f"Single-study mode: {args.study_id}")

    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")

    elif args.rank_by == 'morpho':
        if not morpho_all:
            ap.error("--rank_by morpho requires --morphometrics_json to be provided and non-empty")
        pathologic_ids, normal_ids, score_by_id = select_studies_by_morpho(
            morpho_all,
            n_pathologic=args.top_n,
            n_normal=args.top_normal,
            lstv_by_id=lstv_by_id or None,
        )
        # Deduplicate: pathologic first, then normal (if not already included)
        seen = set(pathologic_ids)
        extra_normal = [s for s in normal_ids if s not in seen]
        study_ids = pathologic_ids + extra_normal
        # Filter to studies that actually have segmentation
        study_ids = [s for s in study_ids if (seg_root/s).is_dir()]
        logger.info(
            f"Morpho selection: {len(pathologic_ids)} pathologic + "
            f"{len(extra_normal)} normal = {len(study_ids)} studies")
        if score_by_id:
            logger.info("Pathology scores:")
            for sid in study_ids:
                s = score_by_id.get(sid, 0)
                tag = '★ normal' if sid in normal_ids and sid not in pathologic_ids else ''
                logger.info(f"  {sid}: {s:.0f}  {tag}")

    else:
        # Legacy uncertainty-CSV mode
        if not args.uncertainty_csv or args.top_n is None:
            ap.error("--uncertainty_csv + --top_n required when --rank_by is not 'morpho'")
        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
        study_ids = select_studies_legacy(Path(args.uncertainty_csv), args.top_n,
                                          args.rank_by, valid_ids)
        study_ids = [s for s in study_ids if (seg_root/s).is_dir()]
        logger.info(f"Legacy selection: {len(study_ids)} studies by {args.rank_by}")

    # ── Render ────────────────────────────────────────────────────────────────
    ok = 0
    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            pre_morpho = morpho_by_id.get(sid)
            out = build_3d_figure(
                study_id=sid, spineps_dir=spineps_dir,
                totalspine_dir=totalspine_dir, smooth=args.smooth,
                show_tss=not args.no_tss,
                lstv_result=lstv_by_id.get(sid),
                morphometrics=pre_morpho,
                uncertainty_row=uncertainty_by_id.get(sid))

            if out is None: continue
            (fig, castellvi, tv_name, cls_L, cls_R,
             span_L, span_R, dist_L, dist_R, metrics) = out

            pscore = score_by_id.get(sid)
            if pscore is None and pre_morpho is not None:
                # Compute on the fly if not already scored (e.g. single-study mode)
                m = {**pre_morpho}
                if sid in lstv_by_id:
                    m['castellvi_type'] = lstv_by_id[sid].get('castellvi_type')
                pscore = compute_pathology_score(m)

            save_html(fig, sid, output_dir, castellvi, tv_name, cls_L, cls_R,
                      span_L, span_R, dist_L, dist_R, metrics,
                      uncertainty_by_id.get(sid), pscore)
            ok += 1
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"\nDone. {ok}/{len(study_ids)} HTMLs → {output_dir}")


if __name__ == '__main__':
    main()
