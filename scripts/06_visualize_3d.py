#!/usr/bin/env python3
"""
06_visualize_3d.py  —  Comprehensive Interactive 3D Spine Segmentation Viewer
==============================================================================
ISOTROPIC-FIRST ARCHITECTURE
  Every mask is resampled to 1×1×1 mm³ BEFORE meshing or measurement.
  Root cause of missing TP voxels: sagittal MRI is ~4.88×0.94×0.94 mm
  (only 15 slices).  A TP spanning 1-2 slices collapses to a flat sheet
  under step-subsampling; marching cubes finds no isosurface.
  After zoom to 1mm³ those same slices become ~5 solid voxels → MC works.

SPINEPS seg-spine_msk labels (from README + live logs):
  26=Sacrum  41=Arcus  42=Spinous  43=TP-Left  44=TP-Right
  45=SupArt-L  46=SupArt-R  47=InfArt-L  48=InfArt-R
  49=CorpusBorder  60=Cord  61=Canal  62=Endplate  100=Disc(all)

VERIDAH seg-vert_msk labels:
  1-7=C1-C7  8-19=T1-T12  28=T13  20=L1  21-25=L2-L6  26=Sacrum
  100+X=IVD below X   200+X=Endplate of X

TotalSpineSeg sagittal labels (from README + live logs):
  1=cord  2=canal  31=T11  32=T12  41=L1..45=L5  50=sacrum
  82=disc_T11-T12  91=disc_T12-L1  92-95=disc_L1-L4_L5  100=disc_L5-S
  ⚠  TSS 43=L3 vertebra body  44=L4 vertebra body  (different from SPINEPS!)
"""

import argparse, json, logging, traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import (binary_fill_holes, distance_transform_edt,
                           gaussian_filter, label as cc_label,
                           zoom as ndizoom)
from skimage.measure import marching_cubes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Label maps ────────────────────────────────────────────────────────────────

SPINE_MASK_LABELS = [                              # (lbl, name, colour, op, fill)
    ( 26, 'Sacrum (spine)',          '#ff8c00', 0.72, True),
    ( 41, 'Arcus Vertebrae',         '#8855cc', 0.55, True),
    ( 42, 'Spinous Process',         '#e8c84a', 0.75, True),
    ( 43, 'TP Left  (costal 43)',    '#ff3333', 0.95, False),
    ( 44, 'TP Right (costal 44)',    '#00d4ff', 0.95, False),
    ( 45, 'Sup Articular Left',      '#66ccaa', 0.65, True),
    ( 46, 'Sup Articular Right',     '#44aa88', 0.65, True),
    ( 47, 'Inf Articular Left',      '#aaddcc', 0.60, True),
    ( 48, 'Inf Articular Right',     '#88ccbb', 0.60, True),
    ( 49, 'Vertebra Corpus Border',  '#6699cc', 0.35, True),
    ( 60, 'Spinal Cord',             '#ffe066', 0.65, False),
    ( 61, 'Spinal Canal',            '#00ffb3', 0.18, False),
    ( 62, 'Endplate',                '#c8f0c8', 0.45, True),
    (100, 'IVD (spine, all)',         '#ffcc44', 0.55, True),
]

VERIDAH_CERVICAL = {i: (f'C{i}',    '#557799', 0.20) for i in range(1, 8)}
VERIDAH_THORACIC = {i+7: (f'T{i+1}','#447766', 0.20) for i in range(12)}
VERIDAH_THORACIC[28] = ('T13', '#447766', 0.20)
VERIDAH_LUMBAR = {
    20: ('L1',              '#1e6fa8', 0.48),
    21: ('L2',              '#2389cc', 0.48),
    22: ('L3',              '#29a3e8', 0.48),
    23: ('L4',              '#52bef5', 0.50),
    24: ('L5',              '#85d4ff', 0.52),
    25: ('L6',              '#aae3ff', 0.52),
    26: ('Sacrum (vert 26)','#ff8c00', 0.62),
}
VERIDAH_IVD_BASE    = 100
VERIDAH_IVD_COLOURS = {20:'#ffe28a',21:'#ffd060',22:'#ffb830',
                        23:'#ff9900',24:'#ff7700',25:'#ff5500'}
VERIDAH_NAMES = {**{k:v[0] for k,v in VERIDAH_LUMBAR.items()},
                 **{k:v[0] for k,v in VERIDAH_CERVICAL.items()},
                 **{k:v[0] for k,v in VERIDAH_THORACIC.items()}}
LUMBAR_LABELS_ORDERED = [25, 24, 23, 22, 21, 20]

TSS_SACRUM_LABEL = 50
TSS_LABELS = [
    (  1,'TSS Cord',         '#ffe066',0.50),
    (  2,'TSS Canal',        '#00ffb3',0.14),
    ( 31,'TSS T11',          '#447766',0.25),
    ( 32,'TSS T12',          '#447766',0.25),
    ( 41,'TSS L1',           '#1e6fa8',0.25),
    ( 42,'TSS L2',           '#2389cc',0.25),
    ( 43,'TSS L3',           '#29a3e8',0.25),
    ( 44,'TSS L4',           '#52bef5',0.25),
    ( 45,'TSS L5',           '#85d4ff',0.28),
    ( 50,'TSS Sacrum',       '#ff8c00',0.65),
    ( 82,'TSS disc T11-T12', '#ffe28a',0.40),
    ( 91,'TSS disc T12-L1',  '#ffd060',0.45),
    ( 92,'TSS disc L1-L2',   '#ffb830',0.50),
    ( 93,'TSS disc L2-L3',   '#ff9900',0.50),
    ( 94,'TSS disc L3-L4',   '#ff7700',0.50),
    ( 95,'TSS disc L4-L5',   '#ff5500',0.52),
    (100,'TSS disc L5-S',    '#ff3300',0.55),
]

TP_LEFT_LABEL   = 43
TP_RIGHT_LABEL  = 44
SPINEPS_SACRUM  = 26
TP_HEIGHT_MM    = 19.0
CONTACT_DIST_MM = 2.0
ISO_MM          = 1.0    # ← isotropic target voxel size in mm

IAN_PAN_LEVELS = ['l1_l2','l2_l3','l3_l4','l4_l5','l5_s1']
IAN_PAN_LABELS = ['L1-L2','L2-L3','L3-L4','L4-L5','L5-S1']

_VALID_S3D_SYM = {'circle','circle-open','cross','diamond',
                  'diamond-open','square','square-open','x'}

# ── NIfTI loading ─────────────────────────────────────────────────────────────

def load_canonical(path):
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

# ── Isotropic resampling ──────────────────────────────────────────────────────

def resample_label_vol_to_iso(label_vol, vox_mm, target_mm=ISO_MM):
    """
    Resample integer label volume to isotropic target_mm³ using
    nearest-neighbour interpolation (order=0) so label values are preserved.
    zoom_factors = vox_mm / target_mm, e.g. [4.88, 0.94, 0.94] @ 1mm.
    """
    zoom_factors = (vox_mm / target_mm).tolist()
    return ndizoom(label_vol.astype(np.int32), zoom_factors,
                   order=0, mode='nearest', prefilter=False).astype(np.int32)

# ── Geometry helpers — all in isotropic mm space ──────────────────────────────

def centroid_mm(iso_mask):
    """Centroid of binary mask in absolute iso mm coordinates."""
    coords = np.array(np.where(iso_mask))
    if coords.size == 0:
        return None
    return coords.mean(axis=1) * ISO_MM

def min_dist_mm(mask_a, mask_b):
    """
    Minimum Euclidean distance (mm) between two binary masks in iso space.
    Returns (dist_mm, pt_a_mm, pt_b_mm) in absolute iso mm.
    """
    if not mask_a.any() or not mask_b.any():
        return float('inf'), None, None
    dt      = distance_transform_edt(~mask_b) * ISO_MM
    dist_at = np.where(mask_a, dt, np.inf)
    flat    = int(np.argmin(dist_at))
    vox_a   = np.array(np.unravel_index(flat, mask_a.shape))
    dist_mm = float(dt[tuple(vox_a)])
    cb      = np.array(np.where(mask_b))
    d2      = ((cb.T - vox_a) ** 2).sum(axis=1)
    vox_b   = cb[:, int(np.argmin(d2))]
    return dist_mm, vox_a.astype(float) * ISO_MM, vox_b.astype(float) * ISO_MM

def tp_height_mm(tp_iso):
    """Superior-inferior (Z) extent of TP mask in mm (iso voxels × ISO_MM)."""
    if not tp_iso.any():
        return 0.0
    zc = np.where(tp_iso)[2]
    return (int(zc.max()) - int(zc.min())) * ISO_MM

def inferiormost_cc(mask_iso, sac_iso=None):
    """Keep the inferiormost connected component, excluding sacrum overlap."""
    if not mask_iso.any():
        return np.zeros_like(mask_iso, dtype=bool)
    labeled, n = cc_label(mask_iso)
    if n == 1:
        return mask_iso.astype(bool)
    sac_z_min = None
    if sac_iso is not None and sac_iso.any():
        sac_z_min = int(np.where(sac_iso)[2].min())
    cc_info = []
    for i in range(1, n + 1):
        comp = (labeled == i)
        zc   = np.where(comp)[2]
        cc_info.append((float(zc.mean()), int(zc.max()), comp))
    cc_info.sort(key=lambda t: t[0])
    if sac_z_min is not None:
        cands = [c for _, zmax, c in cc_info if zmax < sac_z_min]
        if cands:
            return cands[0].astype(bool)
    return cc_info[0][2].astype(bool)

def isolate_at_z_range(mask_iso, z_lo, z_hi, margin=20):
    out = np.zeros_like(mask_iso)
    lo2 = max(0, z_lo - margin)
    hi2 = min(mask_iso.shape[2] - 1, z_hi + margin)
    out[:, :, lo2:hi2 + 1] = mask_iso[:, :, lo2:hi2 + 1]
    return out

def get_z_range(iso_mask):
    if not iso_mask.any():
        return None
    zc = np.where(iso_mask)[2]
    return int(zc.min()), int(zc.max())

# ── Marching cubes → Plotly Mesh3d ───────────────────────────────────────────

def mask_to_mesh3d(iso_mask, origin_mm, name, colour, opacity,
                   smooth_sigma=1.5, fill_holes=True):
    """
    iso_mask : binary array in 1mm³ isotropic space.
    origin_mm: centring offset (subtract from MC vertex coordinates).
    Pipeline: fill_holes → Gaussian smooth → pad → MC → centre.
    spacing=ISO_MM so vertices come out directly in mm — no extra scaling.
    """
    if not iso_mask.any():
        return None
    m = binary_fill_holes(iso_mask) if fill_holes else iso_mask.copy()
    if not m.any():
        return None
    vol = gaussian_filter(m.astype(np.float32), sigma=smooth_sigma)
    vol = np.pad(vol, 1, mode='constant', constant_values=0)
    if vol.max() <= 0.5 or vol.min() >= 0.5:
        logger.debug(f"  '{name}': no 0.5 crossing — skipped")
        return None
    try:
        verts, faces, _, _ = marching_cubes(
            vol, level=0.5, spacing=(ISO_MM, ISO_MM, ISO_MM))
    except Exception as e:
        logger.warning(f"  MC failed '{name}': {e}")
        return None
    verts -= ISO_MM          # undo 1-voxel pad
    verts -= origin_mm[np.newaxis, :]
    return go.Mesh3d(
        x=verts[:,0].tolist(), y=verts[:,1].tolist(), z=verts[:,2].tolist(),
        i=faces[:,0].tolist(), j=faces[:,1].tolist(), k=faces[:,2].tolist(),
        color=colour, opacity=opacity, name=name,
        showlegend=True, flatshading=False,
        lighting=dict(ambient=0.35, diffuse=0.75,
                      specular=0.30, roughness=0.6, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo='name', showscale=False,
    )

# ── Annotation helpers ────────────────────────────────────────────────────────

def _sym(s):
    return s if s in _VALID_S3D_SYM else 'circle'

def ruler_line(p0, p1, colour, name, width=6, dash='solid'):
    return go.Scatter3d(
        x=[p0[0],p1[0]], y=[p0[1],p1[1]], z=[p0[2],p1[2]],
        mode='lines', line=dict(color=colour, width=width, dash=dash),
        name=name, showlegend=True, hoverinfo='name')

def label_point(pos, text, colour, size=10, symbol='circle'):
    return go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(size=size, color=colour, symbol=_sym(symbol),
                    line=dict(color='white', width=1)),
        text=[text], textposition='top center',
        textfont=dict(size=11, color=colour),
        name=text, showlegend=False, hoverinfo='text')

def midpt(a, b):
    return (np.array(a) + np.array(b)) / 2.0

def tp_height_ruler_traces(tp_iso, origin_mm, colour, side, span_mm):
    if not tp_iso.any():
        return []
    best_x, best_span = tp_iso.shape[0] // 2, 0.0
    for x in range(tp_iso.shape[0]):
        col = tp_iso[x]
        if not col.any():
            continue
        zc = np.where(col.any(axis=0))[0]
        if zc.size < 2:
            continue
        sp = (zc.max() - zc.min()) * ISO_MM
        if sp > best_span:
            best_span, best_x = sp, x
    col = tp_iso[best_x]
    if not col.any():
        return []
    zc  = np.where(col.any(axis=0))[0]
    yc  = np.where(col.any(axis=1))[0]
    z_lo, z_hi = int(zc.min()), int(zc.max())
    y_c = int(yc.mean()) if yc.size else tp_iso.shape[1] // 2
    def iv(x,y,z): return np.array([x,y,z],float)*ISO_MM - origin_mm
    p_lo = iv(best_x, y_c, z_lo)
    p_hi = iv(best_x, y_c, z_hi)
    mid  = midpt(p_lo, p_hi)
    flag = '✓' if span_mm < TP_HEIGHT_MM else f'✗ ≥{TP_HEIGHT_MM:.0f}mm→TypeI'
    lbl  = f'{side} TP: {span_mm:.1f}mm  {flag}'
    traces = [ruler_line(p_lo, p_hi, colour, f'Height ruler {side}', width=8)]
    traces.append(label_point(mid, lbl, colour, size=9, symbol='diamond'))
    off = np.array([5.,0.,0.])
    for pt in (p_lo, p_hi):
        traces.append(ruler_line(pt-off, pt+off, colour, f'Tick {side}', width=4))
    return traces

def gap_ruler_traces(tp_iso, sac_iso, origin_mm, colour, side, dist_mm):
    if not tp_iso.any() or not sac_iso.any():
        return []
    _, pt_a, pt_b = min_dist_mm(tp_iso, sac_iso)
    if pt_a is None:
        return []
    p_a = pt_a - origin_mm
    p_b = pt_b - origin_mm
    mid = midpt(p_a, p_b)
    contact = np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM
    dash    = 'dot' if contact else 'dash'
    clbl    = (f'CONTACT {dist_mm:.1f}mm→P2' if contact
               else f'Gap: {dist_mm:.1f}mm ✓')
    return [ruler_line(p_a, p_b, colour, f'Gap ruler {side}', width=5, dash=dash),
            label_point(mid, f'{side}: {clbl}', colour, size=7, symbol='square')]

def tv_plane_traces(vert_iso, tv_label, origin_mm, tv_name):
    mask = (vert_iso == tv_label)
    if not mask.any():
        return []
    zc    = np.where(mask)[2]
    z_mid = int((zc.min() + zc.max()) // 2)
    xs = np.linspace(0, vert_iso.shape[0]-1, 12)
    ys = np.linspace(0, vert_iso.shape[1]-1, 12)
    xg, yg = np.meshgrid(xs, ys)
    zg = np.full_like(xg, z_mid)
    xm = xg*ISO_MM - origin_mm[0]
    ym = yg*ISO_MM - origin_mm[1]
    zm = zg*ISO_MM - origin_mm[2]
    plane = go.Surface(
        x=xm, y=ym, z=zm,
        colorscale=[[0,'rgba(0,230,180,0.10)'],[1,'rgba(0,230,180,0.10)']],
        showscale=False, opacity=0.18,
        name=f'TV plane ({tv_name})', showlegend=True, hoverinfo='name')
    ctr = centroid_mm(mask)
    pts = ([label_point(ctr-origin_mm, f'TV: {tv_name}', '#00e6b4',
                        size=14, symbol='cross')]
           if ctr is not None else [])
    return [plane] + pts

def castellvi_contact_traces(tp_L, tp_R, sac_iso, origin_mm,
                              cls_L, cls_R, dist_L, dist_R):
    traces = []
    for tp_iso, side, dist_mm, cls in (
        (tp_L,'Left', dist_L, cls_L),(tp_R,'Right',dist_R,cls_R)):
        if not (tp_iso.any() and sac_iso.any()):
            continue
        if not (np.isfinite(dist_mm) and dist_mm <= CONTACT_DIST_MM):
            continue
        _, pt_a, _ = min_dist_mm(tp_iso, sac_iso)
        if pt_a is None:
            continue
        p   = pt_a - origin_mm
        col = '#ff2222' if 'III' in (cls or '') else '#ff9900'
        traces.append(go.Scatter3d(
            x=[p[0]],y=[p[1]],z=[p[2]],
            mode='markers+text',
            marker=dict(size=20,color=col,opacity=0.85,symbol='circle',
                        line=dict(color='white',width=2)),
            text=[f'{side}: {cls}'], textposition='middle right',
            textfont=dict(size=13,color=col),
            name=f'Contact {side} ({cls})',showlegend=True,hoverinfo='text'))
    return traces

def ian_pan_bar_traces(uncertainty_row, origin_mm, x_offset_mm=55):
    if uncertainty_row is None:
        return []
    confs = {lvl: uncertainty_row.get(f'{lvl}_confidence', float('nan'))
             for lvl in IAN_PAN_LEVELS}
    valid = [v for v in confs.values() if not np.isnan(v)]
    if not valid:
        return []
    max_conf = max(valid)
    max_h=40.0; bar_w=5.0; gap=2.0
    traces = []
    for i,(lvl,lbl) in enumerate(zip(IAN_PAN_LEVELS,IAN_PAN_LABELS)):
        conf = confs[lvl]
        if np.isnan(conf):
            continue
        x0 = origin_mm[0]+x_offset_mm+i*(bar_w+gap); x1=x0+bar_w
        h  = conf*max_h; z0=-max_h/2; z1=z0+h; y=0.0
        col = '#e63946' if conf==max_conf else '#457b9d'
        vx=[x0,x1,x1,x0,x0,x1,x1,x0]
        vy=[y-1,y-1,y+1,y+1,y-1,y-1,y+1,y+1]
        vz=[z0,z0,z0,z0,z1,z1,z1,z1]
        fi=[0,0,1,1,4,4,0,0,3,3,1,2]; fj=[1,3,2,5,5,7,4,3,7,2,5,6]
        fk=[2,2,5,6,6,6,5,7,6,6,6,7]
        traces.append(go.Mesh3d(x=vx,y=vy,z=vz,i=fi,j=fj,k=fk,
                                color=col,opacity=0.80,
                                name=f'Ian {lbl}: {conf:.2f}',
                                showlegend=True,flatshading=True,hoverinfo='name'))
        traces.append(go.Scatter3d(
            x=[(x0+x1)/2],y=[y],z=[z1+3],mode='text',
            text=[f'{lbl}<br>{conf:.2f}'],
            textfont=dict(size=9,color=col),showlegend=False,hoverinfo='skip'))
    return traces

# ── Study selection ───────────────────────────────────────────────────────────

def select_studies(csv_path, top_n, rank_by, valid_ids):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids is not None:
        before = len(df)
        df = df[df['study_id'].isin(valid_ids)]
        logger.info(f"Filtered to {len(df)} ({before-len(df)} excluded)")
    if rank_by not in df.columns:
        raise ValueError(f"Column '{rank_by}' not found.")
    df_s = df.sort_values(rank_by,ascending=False).reset_index(drop=True)
    top  = df_s.head(top_n)['study_id'].tolist()
    bot  = df_s.tail(top_n)['study_id'].tolist()
    seen,sel = set(),[]
    for sid in top+bot:
        if sid not in seen: sel.append(sid); seen.add(sid)
    logger.info(f"Rank={rank_by}  Top{top_n}:{top}  Bot{top_n}:{bot}")
    return sel

# ── Per-study builder ─────────────────────────────────────────────────────────

def build_3d_figure(study_id, spineps_dir, totalspine_dir,
                    smooth=1.5, lstv_result=None,
                    uncertainty_row=None, show_tss=True):

    seg_dir    = spineps_dir / 'segmentations' / study_id
    spine_path = seg_dir  / f"{study_id}_seg-spine_msk.nii.gz"
    vert_path  = seg_dir  / f"{study_id}_seg-vert_msk.nii.gz"
    tss_path   = (totalspine_dir / study_id / 'sagittal'
                  / f"{study_id}_sagittal_labeled.nii.gz")

    def _load(path, tag):
        if not path.exists():
            logger.warning(f"  Missing: {path.name}"); return None,None
        try:    return load_canonical(path)
        except Exception as e:
            logger.warning(f"  Cannot load {tag}: {e}"); return None,None

    sag_sp,  nii_ref = _load(spine_path, 'seg-spine_msk')
    sag_vert, _      = _load(vert_path,  'seg-vert_msk')
    sag_tss,  _      = _load(tss_path,   'TSS sagittal')

    if sag_sp is None:
        logger.error(f"  [{study_id}] Missing seg-spine_msk"); return None
    if sag_vert is None:
        logger.error(f"  [{study_id}] Missing seg-vert_msk");  return None

    vox_mm = voxel_size_mm(nii_ref)
    logger.info(f"  Native voxel: {np.round(vox_mm,3)}  "
                f"shape: {sag_sp.shape}  → resampling to {ISO_MM}mm isotropic")

    # ── RESAMPLE ALL VOLUMES TO ISOTROPIC 1mm³ ────────────────────────────────
    # Every downstream operation (meshing, measurement, TP isolation) works
    # entirely in this space.  vox_mm is no longer needed after this block.
    sp_iso   = resample_label_vol_to_iso(sag_sp.astype(np.int32),   vox_mm)
    vert_iso = resample_label_vol_to_iso(sag_vert.astype(np.int32), vox_mm)
    tss_iso  = (resample_label_vol_to_iso(sag_tss.astype(np.int32), vox_mm)
                if sag_tss is not None else None)
    logger.info(f"  Iso shape: {sp_iso.shape}")

    sp_labels   = set(np.unique(sp_iso).tolist())   - {0}
    vert_labels = set(np.unique(vert_iso).tolist())  - {0}
    tss_labels  = (set(np.unique(tss_iso).tolist()) - {0}
                   if tss_iso is not None else set())
    logger.info(f"  seg-spine iso labels: {sorted(sp_labels)}")
    logger.info(f"  seg-vert  iso labels: {sorted(vert_labels)}")
    if tss_iso is not None:
        logger.info(f"  TSS       iso labels: {sorted(tss_labels)}")

    # ── Origin = centroid of vertebral column in absolute iso mm ──────────────
    col_mask  = vert_iso > 0
    origin_mm = (centroid_mm(col_mask)
                 if col_mask.any()
                 else np.array(sp_iso.shape, float) / 2.0 * ISO_MM)
    logger.info(f"  Origin_mm: {np.round(origin_mm,1)}")

    # ── Sacrum mask ───────────────────────────────────────────────────────────
    if tss_iso is not None and (tss_iso == TSS_SACRUM_LABEL).any():
        sac_iso = (tss_iso == TSS_SACRUM_LABEL)
        logger.info("  Sacrum: TSS label 50")
    elif (sp_iso == SPINEPS_SACRUM).any():
        sac_iso = (sp_iso == SPINEPS_SACRUM)
        logger.warning("  Sacrum: fallback SPINEPS label 26")
    else:
        sac_iso = np.zeros(sp_iso.shape, bool)
        logger.warning("  Sacrum: NOT FOUND")

    # ── Transitional vertebra ─────────────────────────────────────────────────
    tv_label, tv_name = None, 'N/A'
    for cand in LUMBAR_LABELS_ORDERED:
        if cand in vert_labels:
            tv_label = cand
            tv_name  = VERIDAH_NAMES.get(cand, str(cand))
            break
    logger.info(f"  TV: {tv_name}  label={tv_label}")

    # ── TP masks: isolate to TV z-range then keep inferiormost CC ─────────────
    tp_L_full = (sp_iso == TP_LEFT_LABEL)
    tp_R_full = (sp_iso == TP_RIGHT_LABEL)
    logger.info(f"  TP-L full: {tp_L_full.sum()} vox   "
                f"TP-R full: {tp_R_full.sum()} vox")

    if tv_label is not None:
        tv_zr = get_z_range(vert_iso == tv_label)
        if tv_zr is not None:
            z_lo_tv, z_hi_tv = tv_zr
            logger.info(f"  TV z-range (iso mm): [{z_lo_tv*ISO_MM:.0f}, "
                        f"{z_hi_tv*ISO_MM:.0f}]")
            tp_L_iso = isolate_at_z_range(tp_L_full, z_lo_tv, z_hi_tv, margin=20)
            tp_R_iso = isolate_at_z_range(tp_R_full, z_lo_tv, z_hi_tv, margin=20)
            if not tp_L_iso.any():
                logger.warning("  TP-L isolation empty → full volume")
                tp_L_iso = tp_L_full
            if not tp_R_iso.any():
                logger.warning("  TP-R isolation empty → full volume")
                tp_R_iso = tp_R_full
        else:
            tp_L_iso = tp_L_full; tp_R_iso = tp_R_full
    else:
        tp_L_iso = tp_L_full; tp_R_iso = tp_R_full

    tp_L = inferiormost_cc(tp_L_iso, sac_iso if sac_iso.any() else None)
    tp_R = inferiormost_cc(tp_R_iso, sac_iso if sac_iso.any() else None)

    span_L = tp_height_mm(tp_L)
    span_R = tp_height_mm(tp_R)
    dist_L, _, _ = min_dist_mm(tp_L, sac_iso)
    dist_R, _, _ = min_dist_mm(tp_R, sac_iso)
    logger.info(f"  TP-L: {tp_L.sum()} vox  height={span_L:.1f}mm  "
                f"gap={dist_L:.1f}mm")
    logger.info(f"  TP-R: {tp_R.sum()} vox  height={span_R:.1f}mm  "
                f"gap={dist_R:.1f}mm")

    # ── Classification ────────────────────────────────────────────────────────
    castellvi='N/A'; cls_L='N/A'; cls_R='N/A'
    if lstv_result:
        castellvi = lstv_result.get('castellvi_type') or 'None'
        cls_L     = lstv_result.get('left',  {}).get('classification','N/A')
        cls_R     = lstv_result.get('right', {}).get('classification','N/A')
        det_tv    = lstv_result.get('details',{}).get('tv_name')
        if det_tv: tv_name = det_tv

    # ── Build traces ──────────────────────────────────────────────────────────
    traces = []

    # 1. SPINEPS seg-spine_msk
    for lbl,name,col,op,fh in SPINE_MASK_LABELS:
        if lbl not in sp_labels:
            continue
        mask = (tp_L if lbl==TP_LEFT_LABEL
                else tp_R if lbl==TP_RIGHT_LABEL
                else (sp_iso==lbl))
        if not mask.any():
            continue
        t = mask_to_mesh3d(mask, origin_mm, name, col, op,
                           smooth_sigma=smooth, fill_holes=fh)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-spine {lbl:>3}  {name}")
        else:
            logger.warning(f"    ✗ seg-spine {lbl:>3}  {name}")

    # 2. VERIDAH vertebrae
    all_veridah = {**VERIDAH_CERVICAL, **VERIDAH_THORACIC, **VERIDAH_LUMBAR}
    for lbl,(name,col,op) in sorted(all_veridah.items()):
        if lbl not in vert_labels: continue
        t = mask_to_mesh3d(vert_iso==lbl, origin_mm, name, col, op,
                           smooth_sigma=smooth, fill_holes=True)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-vert  {lbl:>3}  {name}")

    # 2b. VERIDAH IVD labels (100+X)
    for base,col in VERIDAH_IVD_COLOURS.items():
        ivd_lbl = VERIDAH_IVD_BASE + base
        if ivd_lbl not in vert_labels: continue
        name = f'IVD below {VERIDAH_NAMES.get(base,str(base))}'
        t = mask_to_mesh3d(vert_iso==ivd_lbl, origin_mm, name, col, 0.55,
                           smooth_sigma=smooth, fill_holes=True)
        if t:
            traces.append(t)
            logger.info(f"    ✓ seg-vert  {ivd_lbl:>3}  {name}")

    # 3. TotalSpineSeg
    if show_tss and tss_iso is not None:
        for lbl,name,col,op in TSS_LABELS:
            if lbl not in tss_labels: continue
            t = mask_to_mesh3d(tss_iso==lbl, origin_mm, f'TSS {name}',
                               col, op*0.50, smooth_sigma=smooth, fill_holes=True)
            if t:
                traces.append(t)
                logger.info(f"    ✓ TSS       {lbl:>3}  {name}")

    if not any(isinstance(tr, go.Mesh3d) for tr in traces):
        logger.error(f"  [{study_id}] Zero meshes — check label maps")
        return None

    # 4-8. Annotations
    if tv_label is not None:
        traces += tv_plane_traces(vert_iso, tv_label, origin_mm, tv_name)
    traces += tp_height_ruler_traces(tp_L, origin_mm,'#ff3333','Left', span_L)
    traces += tp_height_ruler_traces(tp_R, origin_mm,'#00d4ff','Right',span_R)
    traces += gap_ruler_traces(tp_L, sac_iso, origin_mm,'#ff8800','Left', dist_L)
    traces += gap_ruler_traces(tp_R, sac_iso, origin_mm,'#00aaff','Right',dist_R)
    traces += castellvi_contact_traces(tp_L,tp_R,sac_iso,origin_mm,
                                        cls_L,cls_R,dist_L,dist_R)
    traces += ian_pan_bar_traces(uncertainty_row, origin_mm)

    # Annotation box
    def _fmt(v): return f'{v:.1f} mm' if np.isfinite(v) else 'N/A'
    summary = [
        f"TV:          {tv_name}",
        f"TP-L height: {_fmt(span_L)}  {'✗ TypeI' if span_L>=TP_HEIGHT_MM else '✓'}",
        f"TP-R height: {_fmt(span_R)}  {'✗ TypeI' if span_R>=TP_HEIGHT_MM else '✓'}",
        f"Gap L:       {_fmt(dist_L)}  {'←CONTACT' if np.isfinite(dist_L) and dist_L<=CONTACT_DIST_MM else ''}",
        f"Gap R:       {_fmt(dist_R)}  {'←CONTACT' if np.isfinite(dist_R) and dist_R<=CONTACT_DIST_MM else ''}",
        f"Class L:     {cls_L}",
        f"Class R:     {cls_R}",
        f"Castellvi:   {castellvi}",
    ]
    if tp_L.any(): summary.append(f"TP-L: {tp_L.sum()} vox "
                                   f"({tp_L.sum()*ISO_MM**3/1000:.2f}cm³)")
    if tp_R.any(): summary.append(f"TP-R: {tp_R.sum()} vox "
                                   f"({tp_R.sum()*ISO_MM**3/1000:.2f}cm³)")
    if uncertainty_row:
        for lvl,lbl in zip(IAN_PAN_LEVELS,IAN_PAN_LABELS):
            v = uncertainty_row.get(f'{lvl}_confidence',float('nan'))
            if not np.isnan(v): summary.append(f"Ian {lbl}: {v:.3f}")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f"<b>{study_id}</b>  ·  Castellvi: <b>{castellvi}</b>"
                  f"  ·  TV: <b>{tv_name}</b>"
                  f"  ·  L: <b>{cls_L}</b>  ·  R: <b>{cls_R}</b>"),
            font=dict(size=13,color='#e8e8f0'), x=0.01),
        paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a',
        scene=dict(
            bgcolor='#0d0d1a',
            xaxis=dict(title='X (mm)',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            yaxis=dict(title='Y (mm)',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            zaxis=dict(title='Z (mm)',showgrid=True,gridcolor='#1a1a3e',
                       showbackground=True,backgroundcolor='#0d0d1a',
                       tickfont=dict(color='#8888aa'),
                       titlefont=dict(color='#8888aa'),zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6,y=0.0,z=0.3),up=dict(x=0,y=0,z=1))),
        legend=dict(font=dict(color='#e8e8f0',size=10),
                    bgcolor='rgba(13,13,26,0.85)',
                    bordercolor='#2a2a4a',borderwidth=1,
                    x=0.01,y=0.98,itemsizing='constant'),
        margin=dict(l=0,r=0,t=40,b=0),
        annotations=[
            dict(text='drag=rotate · scroll=zoom · legend=toggle · dbl=isolate',
                 xref='paper',yref='paper',x=0.5,y=-0.01,
                 xanchor='center',yanchor='top',showarrow=False,
                 font=dict(size=10,color='#8888aa'),align='center'),
            dict(text='<b>LSTV Measurements</b><br>'+'<br>'.join(summary),
                 xref='paper',yref='paper',x=0.99,y=0.98,
                 xanchor='right',yanchor='top',showarrow=False,
                 font=dict(size=11,color='#e8e8f0',family='monospace'),
                 bgcolor='rgba(13,13,26,0.88)',
                 bordercolor='#2a2a4a',borderwidth=1,align='left'),
        ])
    return fig, castellvi, tv_name, cls_L, cls_R, span_L, span_R, dist_L, dist_R

# ── HTML template ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Spine — {study_id}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0d0d1a;--sf:#13132a;--bd:#2a2a4a;--tx:#e8e8f0;--mu:#6666aa;--bl:#3a86ff}}
html,body{{background:var(--bg);color:var(--tx);font-family:'JetBrains Mono',monospace;
           height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{display:flex;align-items:center;gap:9px;flex-wrap:wrap;padding:6px 12px;
        border-bottom:1px solid var(--bd);background:var(--sf);flex-shrink:0}}
h1{{font-family:'Syne',sans-serif;font-size:.86rem;font-weight:700;white-space:nowrap}}
.b{{display:inline-block;padding:2px 8px;border-radius:20px;
    font-size:.65rem;font-weight:600;letter-spacing:.05em}}
.bs{{background:#2a2a4a;color:var(--mu)}} .bc{{background:#ff8c00;color:#0d0d1a}}
.bt{{background:#1e6fa8;color:#fff}}     .bL{{background:#cc2222;color:#fff}}
.bR{{background:#006688;color:#fff}}     .bi{{background:#1a3a2a;color:#2dc653;border:1px solid #2dc653}}
.tb{{display:flex;gap:5px;align-items:center;margin-left:auto}}
.tb span{{font-size:.59rem;color:var(--mu);text-transform:uppercase;letter-spacing:.08em}}
button{{background:var(--bg);border:1px solid var(--bd);color:var(--tx);
        font-family:inherit;font-size:.65rem;padding:3px 9px;border-radius:4px;cursor:pointer}}
button:hover{{background:var(--bd)}} button.on{{background:var(--bl);border-color:var(--bl);color:#fff}}
.mt{{display:flex;gap:16px;flex-wrap:wrap;align-items:center;padding:4px 12px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.64rem}}
.m{{display:flex;align-items:center;gap:4px;color:var(--mu)}}
.v{{color:var(--tx);font-weight:600}} .ok{{color:#2dc653!important}}
.wn{{color:#ff8800!important}} .cr{{color:#ff3333!important}}
.lg{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:4px 12px;
     border-bottom:1px solid var(--bd);flex-shrink:0;font-size:.62rem}}
.li{{display:flex;align-items:center;gap:3px;color:var(--mu)}}
.sw{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
#pl{{flex:1;min-height:0}} #pl .js-plotly-plot,#pl .plot-container{{height:100%!important}}
</style></head><body>
<header><h1>3D SPINE</h1>
  <span class="b bs">{study_id}</span>
  <span class="b bc">Castellvi: {castellvi}</span>
  <span class="b bt">TV: {tv_name}</span>
  <span class="b bL">L: {cls_L}</span>
  <span class="b bR">R: {cls_R}</span>
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
  <div class="m">L <span class="v">{cls_L}</span></div>
  <div class="m">R <span class="v">{cls_R}</span></div>
  {ian_metrics}
  <div style="margin-left:auto;color:#333355;font-size:.58rem">drag=rotate·scroll=zoom·legend=toggle·dbl=isolate</div>
</div>
<div class="lg">
  <div class="li"><div class="sw" style="background:#ff3333"></div>TP-L(43)</div>
  <div class="li"><div class="sw" style="background:#00d4ff"></div>TP-R(44)</div>
  <div class="li"><div class="sw" style="background:#ff8c00"></div>Sacrum</div>
  <div class="li"><div class="sw" style="background:#8855cc"></div>Arcus</div>
  <div class="li"><div class="sw" style="background:#e8c84a"></div>Spinous</div>
  <div class="li"><div class="sw" style="background:#66ccaa"></div>Articular</div>
  <div class="li"><div class="sw" style="background:#ffcc44"></div>IVD(spine)</div>
  <div class="li"><div class="sw" style="background:#ffe28a"></div>IVD(vert)</div>
  <div class="li"><div class="sw" style="background:#1e6fa8;opacity:.7"></div>L1-L6</div>
  <div class="li"><div class="sw" style="background:#00ffb3;opacity:.5"></div>Canal</div>
  <div class="li"><div class="sw" style="background:#ffe066;opacity:.8"></div>Cord</div>
  <div class="li"><div class="sw" style="background:#00e6b4;opacity:.4"></div>TV plane</div>
</div>
<div id="pl">{plotly_div}</div>
<script>
const V={{
  oblique:  {{eye:{{x:1.6,y:0.8,z:0.4}},up:{{x:0,y:0,z:1}}}},
  lateral:  {{eye:{{x:2.4,y:0.0,z:0.0}},up:{{x:0,y:0,z:1}}}},
  posterior:{{eye:{{x:0.0,y:2.4,z:0.0}},up:{{x:0,y:0,z:1}}}},
  anterior: {{eye:{{x:0.0,y:-2.4,z:0.0}},up:{{x:0,y:0,z:1}}}},
  axial:    {{eye:{{x:0.0,y:0.0,z:3.0}},up:{{x:0,y:1,z:0}}}},
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

# ── Save HTML ─────────────────────────────────────────────────────────────────

def save_html(fig, study_id, output_dir, castellvi, tv_name, cls_L, cls_R,
              span_L, span_R, dist_L, dist_R, uncertainty_row):
    from plotly.io import to_html
    plotly_div = to_html(fig, full_html=False, include_plotlyjs='cdn',
                         config=dict(responsive=True,displayModeBar=True,
                                     modeBarButtonsToRemove=['toImage'],
                                     displaylogo=False))
    def _f(v): return f'{v:.1f} mm' if np.isfinite(v) else 'N/A'
    def _hc(v): return 'wn' if v>=TP_HEIGHT_MM else 'ok'
    def _gc(v): return 'cr' if (np.isfinite(v) and v<=CONTACT_DIST_MM) else 'ok'
    ian_badge = ''; ian_metrics = ''
    if uncertainty_row:
        c = uncertainty_row.get('l5_s1_confidence', float('nan'))
        if not np.isnan(c):
            ian_badge = f'<span class="b bi">Ian L5-S1: {c:.3f}</span>'
        ian_metrics = ''.join(
            f'<div class="m">{lbl} <span class="v">'
            f'{uncertainty_row.get(f"{lvl}_confidence",float("nan")):.2f}'
            f'</span></div>'
            for lvl,lbl in zip(IAN_PAN_LEVELS,IAN_PAN_LABELS)
            if not np.isnan(uncertainty_row.get(f'{lvl}_confidence',float('nan')))
        )
    html = HTML.format(
        study_id=study_id, castellvi=castellvi or 'N/A',
        tv_name=tv_name or 'N/A', cls_L=cls_L or 'N/A', cls_R=cls_R or 'N/A',
        ian_badge=ian_badge, ian_metrics=ian_metrics,
        span_L=_f(span_L), tpl_c=_hc(span_L),
        span_R=_f(span_R), tpr_c=_hc(span_R),
        gap_L=_f(dist_L),  gl_c=_gc(dist_L),
        gap_R=_f(dist_R),  gr_c=_gc(dist_R),
        plotly_div=plotly_div)
    out = output_dir / f"{study_id}_3d_spine.html"
    out.write_text(html, encoding='utf-8')
    logger.info(f"  → {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spineps_dir',    required=True)
    ap.add_argument('--totalspine_dir', required=True)
    ap.add_argument('--output_dir',     required=True)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--study_id', default=None)
    grp.add_argument('--all',      action='store_true')
    ap.add_argument('--uncertainty_csv', default=None)
    ap.add_argument('--valid_ids',       default=None)
    ap.add_argument('--top_n',    type=int,   default=None)
    ap.add_argument('--rank_by',  default='l5_s1_confidence')
    ap.add_argument('--lstv_json', default=None)
    ap.add_argument('--smooth',    type=float, default=1.5)
    ap.add_argument('--no_tss',    action='store_true')
    args = ap.parse_args()

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
        df = pd.read_csv(csv_path)
        df['study_id'] = df['study_id'].astype(str)
        uncertainty_by_id = {r['study_id']: r for r in df.to_dict('records')}
        logger.info(f"Loaded uncertainty for {len(uncertainty_by_id)} studies")

    if args.study_id:
        study_ids = [args.study_id]
    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")
    else:
        if not args.uncertainty_csv or args.top_n is None:
            ap.error("--uncertainty_csv and --top_n required unless --all/--study_id")
        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
        study_ids = select_studies(csv_path, args.top_n, args.rank_by, valid_ids)
        study_ids = [s for s in study_ids if (seg_root/s).is_dir()]
        logger.info(f"Selective mode: {len(study_ids)} studies")

    ok = 0
    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            out = build_3d_figure(
                study_id=sid, spineps_dir=spineps_dir,
                totalspine_dir=totalspine_dir, smooth=args.smooth,
                lstv_result=results_by_id.get(sid),
                uncertainty_row=uncertainty_by_id.get(sid),
                show_tss=not args.no_tss)
            if out is None: continue
            fig,castellvi,tv_name,cls_L,cls_R,span_L,span_R,dist_L,dist_R = out
            save_html(fig,sid,output_dir,castellvi,tv_name,cls_L,cls_R,
                      span_L,span_R,dist_L,dist_R,uncertainty_by_id.get(sid))
            ok += 1
        except Exception as e:
            logger.error(f"  [{sid}] Failed: {e}")
            logger.debug(traceback.format_exc())

    logger.info(f"\nDone. {ok}/{len(study_ids)} HTMLs → {output_dir}")

if __name__ == '__main__':
    main()
