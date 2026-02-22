#!/usr/bin/env python3
"""
05_morphometrics.py  —  Standalone Spine Morphometrics Runner
=============================================================
Computes comprehensive spine morphometrics for the entire dataset
(or a selected subset) WITHOUT any 3D rendering.

Outputs per run:
  results/morphometrics/
  ├── morphometrics_all.json       ← load into 06_visualize_3d_v2.py
  ├── morphometrics_all.csv        ← pandas / R / Excel analysis
  ├── morphometrics_summary.json   ← dataset-level statistics
  └── reports/
      └── {study_id}_report.html   ← per-study clinical report

Usage:
  python3 05_morphometrics.py --spineps_dir results/spineps \\
      --totalspine_dir results/totalspineseg \\
      --output_dir results/morphometrics \\
      [--all | --top_n 5 --uncertainty_csv ... --valid_ids ...] \\
      [--study_id SPECIFIC_ID]
"""

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Local engine — must be in same scripts/ directory
from morphometrics_engine import (
    MorphometricResult, MaskSet, T, LUMBAR_PAIRS, CANAL_SHAPE,
    load_study_masks, run_all_morphometrics,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── Study selection (mirrors 04_detect_lstv.py) ──────────────────────────────

def select_studies(csv_path: Path, top_n: int, rank_by: str,
                   valid_ids) -> list:
    df = pd.read_csv(csv_path)
    df['study_id'] = df['study_id'].astype(str)
    if valid_ids is not None:
        df = df[df['study_id'].isin(valid_ids)]
    df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    top = df.head(top_n)['study_id'].tolist()
    bot = df.tail(top_n)['study_id'].tolist()
    seen, sel = set(), []
    for sid in top+bot:
        if sid not in seen: sel.append(sid); seen.add(sid)
    return sel


# ─── HTML report generation ───────────────────────────────────────────────────

def _fmt(v, unit='mm', digits=1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'N/A'
    return f'{v:.{digits}f} {unit}'.strip()

def _cls(val, green_condition, yellow_condition=None) -> str:
    """Return CSS class string: ok / warn / crit."""
    if val is None: return ''
    if green_condition(val): return 'ok'
    if yellow_condition and yellow_condition(val): return 'warn'
    return 'crit'

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Spine Report — {study_id}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&family=Inter:wght@400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#f5f5f7;--sf:#ffffff;--bd:#d0d0d8;--tx:#1a1a2e;--mu:#6666aa;
       --ok:#1a7a3a;--ok-bg:#e8f5ee;--warn:#b06000;--warn-bg:#fff5e0;
       --crit:#b00020;--crit-bg:#fde8ec}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--tx);
      font-size:13px;line-height:1.5;padding:20px}}
h1{{font-size:1.4rem;font-weight:700;margin-bottom:4px}}
.sub{{color:var(--mu);font-size:.85rem;margin-bottom:20px}}
.section{{background:var(--sf);border:1px solid var(--bd);border-radius:8px;
          margin-bottom:16px;overflow:hidden}}
.sec-hdr{{background:#1a1a2e;color:#e8e8f0;padding:8px 14px;
           font-weight:700;font-size:.85rem;letter-spacing:.06em;
           display:flex;justify-content:space-between;align-items:center}}
.sec-hdr .badge{{background:#3a3a6e;color:#ccc;padding:2px 8px;
                  border-radius:12px;font-size:.7rem;font-weight:400}}
.sec-hdr .badge.crit{{background:#7a0014;color:#ffcccc}}
.sec-hdr .badge.warn{{background:#5a3800;color:#ffe0aa}}
.sec-hdr .badge.ok{{background:#0a3a1a;color:#aaffcc}}
table{{width:100%;border-collapse:collapse}}
th{{background:#f0f0f6;font-weight:600;font-size:.75rem;letter-spacing:.04em;
    text-transform:uppercase;color:var(--mu);padding:6px 12px;text-align:left;
    border-bottom:1px solid var(--bd)}}
td{{padding:6px 12px;border-bottom:1px solid #eee;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#f8f8fc}}
.val{{font-family:'JetBrains Mono',monospace;font-size:.85rem;font-weight:600}}
.ok{{color:var(--ok)}} .warn{{color:var(--warn)}} .crit{{color:var(--crit)}}
.ok-row td{{background:var(--ok-bg)}}
.warn-row td{{background:var(--warn-bg)}}
.crit-row td{{background:var(--crit-bg)}}
.interp{{color:#444;font-size:.8rem;max-width:340px}}
.ref{{color:var(--mu);font-size:.75rem;font-family:'JetBrains Mono',monospace}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}}
.metric-card{{border:1px solid var(--bd);border-radius:6px;padding:10px 14px;
              background:#fafafa}}
.metric-card .label{{font-size:.72rem;color:var(--mu);letter-spacing:.05em;
                      text-transform:uppercase;margin-bottom:3px}}
.metric-card .value{{font-size:1.2rem;font-weight:700;font-family:'JetBrains Mono',monospace}}
.cord-bar{{display:flex;align-items:center;gap:8px;margin:8px 0}}
.cord-bar .bar-track{{flex:1;height:12px;background:#eee;border-radius:6px;overflow:hidden}}
.cord-bar .bar-fill{{height:100%;border-radius:6px;transition:width .3s}}
footer{{text-align:center;color:var(--mu);font-size:.75rem;margin-top:24px}}
</style></head><body>
<h1>🦴 Spine Morphometric Report</h1>
<div class="sub">Study: <strong>{study_id}</strong> &nbsp;|&nbsp;
Generated by lstv-detector 06 &nbsp;|&nbsp; {timestamp}</div>
"""

_HTML_FOOT = "<footer>lstv-detector · morphometrics_engine.py · Anthropic</footer></body></html>"


def _section(title: str, badge_text: str = '', badge_cls: str = '') -> str:
    badge = (f'<span class="badge {badge_cls}">{badge_text}</span>'
             if badge_text else '')
    return f'<div class="section"><div class="sec-hdr">{title} {badge}</div>'

def _table_row(label: str, value: str, ref: str, interp: str,
               row_cls: str = '') -> str:
    return (f'<tr class="{row_cls}"><td>{label}</td>'
            f'<td class="val {row_cls.replace("-row","")}">{value}</td>'
            f'<td class="ref">{ref}</td>'
            f'<td class="interp">{interp}</td></tr>')


def build_html_report(res: MorphometricResult, study_id: str,
                      lstv_result: Optional[dict] = None) -> str:
    import datetime
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    html = _HTML_HEAD.format(study_id=study_id, timestamp=ts)

    # ── LSTV banner (if available) ──────────────────────────────────────────
    if lstv_result:
        cv = lstv_result.get('castellvi_type','N/A')
        tv = lstv_result.get('details',{}).get('tv_name','N/A')
        cl = lstv_result.get('left',{}).get('classification','N/A')
        cr = lstv_result.get('right',{}).get('classification','N/A')
        html += (f'<div class="section"><div class="sec-hdr">LSTV / Castellvi'
                 f'<span class="badge {"crit" if cv not in (None,"None","Normal") else "ok"}">'
                 f'Castellvi: {cv}</span></div>'
                 f'<div style="padding:12px 16px;display:flex;gap:24px">'
                 f'<div class="metric-card"><div class="label">Transitional Vertebra</div>'
                 f'<div class="value">{tv}</div></div>'
                 f'<div class="metric-card"><div class="label">Left Side</div>'
                 f'<div class="value">{cl}</div></div>'
                 f'<div class="metric-card"><div class="label">Right Side</div>'
                 f'<div class="value">{cr}</div></div>'
                 f'</div></div>')

    # ── Global Canal Stenosis ────────────────────────────────────────────────
    ap   = res.canal_ap_mm
    dsca = res.canal_dsca_mm2
    ap_cls_name = res.canal_ap_class or 'N/A'
    canal_badge_cls = 'crit' if 'Absolute' in ap_cls_name else \
                      'warn' if 'Relative' in ap_cls_name else 'ok'

    html += _section('Central Spinal Canal Stenosis',
                     ap_cls_name, canal_badge_cls)
    html += '<table><tr><th>Parameter</th><th>Value</th><th>Reference</th><th>Interpretation</th></tr>'

    def ap_row_cls(v):
        if v is None: return ''
        if v < T.AP_ABSOLUTE_MM: return 'crit-row'
        if v < T.AP_NORMAL_MM:   return 'warn-row'
        return 'ok-row'
    def dsca_row_cls(v):
        if v is None: return ''
        if v < T.DSCA_ABSOLUTE_MM2: return 'crit-row'
        if v < T.DSCA_NORMAL_MM2:   return 'warn-row'
        return 'ok-row'

    html += _table_row('AP Diameter', _fmt(ap),
        f'Normal >12mm | Relative 10–12mm | Absolute <7mm',
        ap_cls_name, ap_row_cls(ap))
    html += _table_row('DSCA (estimated)', _fmt(dsca, 'mm²', 0),
        f'Normal >100mm² | Relative 75–100mm² | Absolute <70mm²',
        res.canal_dsca_class or 'N/A', dsca_row_cls(dsca))
    html += _table_row('Source', res.canal_source or 'N/A', '', '', '')
    html += '</table></div>'

    # ── Per-level canal ──────────────────────────────────────────────────────
    if any(lm.level_ap_mm is not None for lm in res.levels):
        html += _section('Per-Level Canal Diameter (at TSS disc midpoint)')
        html += '<table><tr><th>Level</th><th>AP (mm)</th><th>DSCA (mm²)</th><th>Class</th><th>Canal Shape</th></tr>'
        for lm in res.levels:
            if lm.level_ap_mm is None: continue
            rc = ap_row_cls(lm.level_ap_mm)
            html += (f'<tr class="{rc}"><td>{lm.level_display}</td>'
                     f'<td class="val">{_fmt(lm.level_ap_mm)}</td>'
                     f'<td class="val">{_fmt(lm.level_dsca_mm2,"mm²",0)}</td>'
                     f'<td class="val">{lm.level_ap_class or "N/A"}</td>'
                     f'<td>{lm.canal_shape or "N/A"}</td></tr>')
        html += '</table></div>'

    # ── Cord Compression Profile ─────────────────────────────────────────────
    cp = res.cord_compression_profile
    cord_badge_cls = 'ok'
    cord_badge = 'No compression'
    if cp:
        mc = cp.get('max_mscc', 0)
        cc = cp.get('classification','Normal')
        fc = cp.get('flagged_count', 0)
        cord_badge_cls = 'crit' if cc == 'Severe' else \
                         'warn' if cc in ('Moderate','Mild') else 'ok'
        cord_badge = f'{cc} (max MSCC={mc:.2f}, {fc} flagged slices)'

    html += _section('Spinal Cord Compression (Full Length)', cord_badge, cord_badge_cls)
    html += '<table><tr><th>Parameter</th><th>Value</th><th>Reference</th><th>Interpretation</th></tr>'
    html += _table_row('Cord AP diameter', _fmt(res.cord_ap_mm),
        'Normal reference varies by level', '', '')
    html += _table_row('Cord CSA (estimated)', _fmt(res.cord_csa_mm2, 'mm²', 0),
        '', '', '')
    html += _table_row('MSCC (global proxy)', _fmt(res.mscc_proxy, '', 3),
        f'<0.50 Normal | 0.50–0.67 Mild | 0.67–0.80 Moderate | ≥0.80 Severe',
        '', '')

    if cp:
        html += _table_row('Max MSCC along cord',
            f'{cp.get("max_mscc",0):.3f} at z={cp.get("max_mscc_z_mm",0):.0f}mm',
            '≥0.67 → compression flag', cp.get('classification',''), '')
        html += _table_row('Flagged slices (MSCC≥0.67)',
            str(cp.get('flagged_count',0)), '0 expected', '', '')
        # Inline mini-bar chart of flagged z positions
        if cp.get('flagged_z_mm'):
            zs = ', '.join(f'{z:.0f}' for z in cp['flagged_z_mm'][:20])
            ellipsis = ' …' if len(cp['flagged_z_mm']) > 20 else ''
            html += _table_row('Flagged positions (mm)', f'{zs}{ellipsis}', '', '', '')

        # Full-length compression chart using CSS bars
        slices = cp.get('slices', [])
        if slices:
            html += '</table>'
            html += '<div style="padding:12px 16px">'
            html += '<div style="font-size:.75rem;color:#666;margin-bottom:6px">MSCC per slice (Z = superoinferior)</div>'
            colors = {'Normal':'#2dc653','Mild':'#f0a500','Moderate':'#e07800','Severe':'#e02020'}
            max_z = max(s['z_mm'] for s in slices); min_z = min(s['z_mm'] for s in slices)
            html += '<div style="display:flex;flex-wrap:wrap;gap:2px;align-items:flex-end">'
            for s in slices[::max(1,len(slices)//120)]:   # at most 120 bars
                pct = min(100, s['mscc'] * 100)
                col = colors.get(s['cls'], '#888')
                html += (f'<div title="z={s["z_mm"]:.0f}mm MSCC={s["mscc"]:.2f} {s["cls"]}" '
                         f'style="width:4px;height:{max(2,int(pct*0.5))}px;'
                         f'background:{col};border-radius:1px"></div>')
            html += '</div>'
            html += '<div style="font-size:.68rem;color:#aaa;margin-top:4px">'
            html += f'Sup (z={min_z:.0f}mm) → Inf (z={max_z:.0f}mm) | '
            html += '<span style="color:#2dc653">■</span> Normal '
            html += '<span style="color:#f0a500">■</span> Mild '
            html += '<span style="color:#e07800">■</span> Moderate '
            html += '<span style="color:#e02020">■</span> Severe</div>'
            html += '</div><table>'
        html += '</table>'
    else:
        html += _table_row('Cord compression profile', 'N/A',
            'Requires cord + canal masks', 'Masks not found', '')
        html += '</table>'
    html += '</div>'

    # ── Disc Height Index ────────────────────────────────────────────────────
    def dhi_row_cls(v):
        if v is None: return ''
        if v < T.DHI_SEVERE_PCT:   return 'crit-row'
        if v < T.DHI_MODERATE_PCT: return 'warn-row'
        return 'ok-row'

    html += _section('Disc Height Index (DHI — Farfan Method)')
    html += ('<table><tr><th>Level</th><th>DHI (%)</th><th>Grade</th>'
             '<th>Endplate Dist</th><th>EP Source</th><th>DHI Source</th></tr>')
    for lm in res.levels:
        rc = dhi_row_cls(lm.dhi_pct)
        ep = _fmt(lm.endplate_dist_mm) if lm.endplate_dist_mm is not None else 'N/A'
        ep_cls = 'crit' if (lm.endplate_dist_mm is not None and lm.endplate_dist_mm < 3.0) else ''
        html += (f'<tr class="{rc}"><td>{lm.level_display}</td>'
                 f'<td class="val">{_fmt(lm.dhi_pct,"%",1)}</td>'
                 f'<td>{lm.dhi_grade or "N/A"}</td>'
                 f'<td class="{ep_cls}">{ep}</td>'
                 f'<td>{lm.endplate_source or "N/A"}</td>'
                 f'<td>{lm.disc_source or "N/A"}</td></tr>')
    html += '</table>'
    html += ('<div style="padding:6px 14px;font-size:.75rem;color:var(--mu)">'
             'DHI=(Ha+Hp)/(Ds+Di)×100  |  &lt;50%→Severe  &lt;70%→Moderate  &lt;85%→Mild  '
             '|  ep-ep &lt;3mm flagged</div></div>')

    # ── Vertebral Body Heights ────────────────────────────────────────────────
    html += _section('Vertebral Body Height Ratios (Genant/QM)')
    html += ('<table><tr><th>Vertebra</th><th>Ha (ant)</th><th>Hm (mid)</th>'
             '<th>Hp (post)</th><th>Wedge (Ha/Hp)</th><th>Compression</th>'
             '<th>Genant Grade</th></tr>')
    for lm in res.levels:
        if lm.ha_mm is None: continue
        w = lm.wedge_ha_hp
        rc = ('crit-row' if (w and w < T.HEIGHT_INTERVENTION) else
              'warn-row' if (w and w < T.WEDGE_FRACTURE) else 'ok-row')
        html += (f'<tr class="{rc}"><td>{lm.level_display.split("-")[0]}</td>'
                 f'<td class="val">{_fmt(lm.ha_mm)}</td>'
                 f'<td class="val">{_fmt(lm.hm_mm)}</td>'
                 f'<td class="val">{_fmt(lm.hp_mm)}</td>'
                 f'<td class="val">{_fmt(w,"",2)}</td>'
                 f'<td class="val">{_fmt(lm.compression_hm_ha,"",2)}</td>'
                 f'<td>{lm.genant_label or "N/A"}</td></tr>')
    html += '</table>'
    html += ('<div style="padding:6px 14px;font-size:.75rem;color:var(--mu)">'
             'Wedge &lt;0.80→fracture  &lt;0.75→intervention threshold  '
             '|  Compression Hm/Ha &lt;0.80→biconcave</div></div>')

    # ── Spondylolisthesis ────────────────────────────────────────────────────
    html += _section('Spondylolisthesis (Sagittal Translation)')
    html += '<table><tr><th>Level</th><th>Translation (mm)</th><th>Classification</th><th>Threshold</th></tr>'
    for lm in res.levels:
        if lm.sagittal_translation_mm is None: continue
        rc = ('crit-row' if lm.sagittal_translation_mm >= T.SPONDYLO_MM else 'ok-row')
        html += (f'<tr class="{rc}"><td>{lm.level_display}</td>'
                 f'<td class="val">{_fmt(lm.sagittal_translation_mm)}</td>'
                 f'<td>{lm.spondylolisthesis or "N/A"}</td>'
                 f'<td class="ref">≥{T.SPONDYLO_MM}mm = Positive</td></tr>')
    html += '</table></div>'

    # ── Ligamentum Flavum ────────────────────────────────────────────────────
    lft = res.lft_proxy_mm
    lft_rc = ('crit-row' if (lft and lft > T.LFT_SEVERE_MM) else
              'warn-row' if (lft and lft > T.LFT_HYPERTROPHY_MM) else 'ok-row')
    html += _section('Ligamentum Flavum (Proxy)')
    html += '<table><tr><th>Parameter</th><th>Value</th><th>Reference</th><th>Classification</th></tr>'
    html += _table_row('LFT proxy (arcus→canal)', _fmt(lft),
        f'Normal ≤{T.LFT_NORMAL_MM}mm  Hypertrophy ≥{T.LFT_HYPERTROPHY_MM}mm  Severe >{T.LFT_SEVERE_MM}mm',
        res.lft_class or 'N/A', lft_rc)
    html += _table_row('LFA stenosis cutoff ref', f'{T.LFA_CUTOFF_MM2} mm²',
        'Optimal predictor central canal stenosis (literature)', '', '')
    html += '</table></div>'

    # ── Baastrup ─────────────────────────────────────────────────────────────
    gap = res.min_inter_process_gap_mm
    bstp_badge = 'CONTACT' if res.baastrup_contact else ('Risk' if res.baastrup_risk else 'None')
    bstp_badge_cls = 'crit' if res.baastrup_contact else ('warn' if res.baastrup_risk else 'ok')
    html += _section('Baastrup Disease (Kissing Spine)', bstp_badge, bstp_badge_cls)
    html += '<table><tr><th>Parameter</th><th>Value</th><th>Reference</th><th>Interpretation</th></tr>'
    html += _table_row('Min inter-spinous gap', _fmt(gap),
        f'Contact ≤{T.BAASTRUP_CONTACT_MM}mm  Risk ≤{T.BAASTRUP_RISK_MM}mm',
        bstp_badge, 'crit-row' if res.baastrup_contact else 'warn-row' if res.baastrup_risk else 'ok-row')
    if res.inter_process_gaps_mm:
        gaps_str = '  '.join(f'{g:.1f}' for g in res.inter_process_gaps_mm)
        html += _table_row('All inter-spinous gaps (mm)', gaps_str, '', '', '')
    html += '</table></div>'

    # ── Facet Tropism ────────────────────────────────────────────────────────
    trop = res.facet_tropism_deg
    ft_grade = res.facet_tropism_grade or 'N/A'
    ft_badge_cls = ('crit' if (trop and trop >= T.TROPISM_SEVERE_DEG) else
                    'warn' if (trop and trop >= T.TROPISM_NORMAL_DEG) else 'ok')
    html += _section('Facet Tropism (Ko et al.)', ft_grade.split('(')[0].strip(), ft_badge_cls)
    html += '<table><tr><th>Parameter</th><th>Value</th><th>Reference</th><th>Interpretation</th></tr>'
    ft_rc = ('crit-row' if ft_badge_cls == 'crit' else
             'warn-row' if ft_badge_cls == 'warn' else 'ok-row')
    html += _table_row('Facet tropism angle', _fmt(trop, '°'),
        f'Grade 0 ≤7°  Grade 1 7–10°  Grade 2 ≥10°', ft_grade, ft_rc)
    html += _table_row('Angle Left',  _fmt(res.facet_angle_l_deg, '°'), '', '', '')
    html += _table_row('Angle Right', _fmt(res.facet_angle_r_deg, '°'), '', '', '')
    html += '</table></div>'

    # ── Neural Foraminal Volume ───────────────────────────────────────────────
    html += _section('Neural Foraminal Volume (Lee Grade Equivalent)')
    html += ('<table><tr><th>Level</th><th>Vol L (mm³)</th><th>% Norm L</th>'
             '<th>Class L</th><th>Vol R (mm³)</th><th>% Norm R</th><th>Class R</th></tr>')
    for lm in res.levels:
        if lm.foraminal_vol_L_mm3 is None and lm.foraminal_vol_R_mm3 is None: continue
        def _fcls(c):
            return ('crit-row' if c and 'Grade 3' in c else
                    'warn-row' if c and 'Grade 2' in c else 'ok-row')
        html += (f'<tr><td>{lm.level_display}</td>'
                 f'<td class="val">{_fmt(lm.foraminal_vol_L_mm3,"",0)}</td>'
                 f'<td class="val">{_fmt(lm.foraminal_norm_pct_L,"%",0)}</td>'
                 f'<td>{lm.foraminal_class_L or "N/A"}</td>'
                 f'<td class="val">{_fmt(lm.foraminal_vol_R_mm3,"",0)}</td>'
                 f'<td class="val">{_fmt(lm.foraminal_norm_pct_R,"%",0)}</td>'
                 f'<td>{lm.foraminal_class_R or "N/A"}</td></tr>')
    html += '</table>'
    html += ('<div style="padding:6px 14px;font-size:.75rem;color:var(--mu)">'
             'Lee Grade: 0=Normal &nbsp; 1=Mild &nbsp; 2=Moderate &nbsp; 3=Severe (nerve morphology change)'
             '</div></div>')

    # ── Clinical Thresholds Reference Card ───────────────────────────────────
    html += _section('Quick Reference: Clinical Thresholds')
    html += """<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0;font-size:.75rem">"""
    refs = [
        ('Central Stenosis DSCA', f'Normal &gt;{T.DSCA_NORMAL_MM2:.0f}mm²',
         f'Absolute &lt;{T.DSCA_ABSOLUTE_MM2:.0f}mm²'),
        ('Central Stenosis AP', f'Normal &gt;{T.AP_NORMAL_MM:.0f}mm',
         f'Absolute &lt;{T.AP_ABSOLUTE_MM:.0f}mm'),
        ('MSCC (cord/canal)', f'Mild ≥{T.MSCC_MILD:.2f}',
         f'Severe ≥{T.MSCC_SEVERE:.2f}'),
        ('DHI (Farfan)', f'Mild &lt;{T.DHI_MILD_PCT:.0f}%',
         f'Severe &lt;{T.DHI_SEVERE_PCT:.0f}%'),
        ('Vertebral Wedge', f'Fracture &lt;{T.WEDGE_FRACTURE:.2f}',
         f'Intervention &lt;{T.HEIGHT_INTERVENTION:.2f}'),
        ('Spondylolisthesis', f'Positive ≥{T.SPONDYLO_MM:.0f}mm translation', ''),
        ('LFT Hypertrophy', f'Threshold ≥{T.LFT_HYPERTROPHY_MM:.1f}mm',
         f'Severe &gt;{T.LFT_SEVERE_MM:.0f}mm'),
        ('Baastrup Disease', f'Risk ≤{T.BAASTRUP_RISK_MM:.0f}mm',
         f'Contact ≤{T.BAASTRUP_CONTACT_MM:.0f}mm'),
        ('Facet Tropism Ko', f'Grade 1 &gt;{T.TROPISM_NORMAL_DEG:.0f}°',
         f'Grade 2 ≥{T.TROPISM_SEVERE_DEG:.0f}°'),
    ]
    for name, norm_txt, flag_txt in refs:
        html += (f'<div style="padding:8px 12px;border-right:1px solid #eee;'
                 f'border-bottom:1px solid #eee">'
                 f'<div style="font-weight:600;color:#333;margin-bottom:2px">{name}</div>'
                 f'<div style="color:var(--ok)">{norm_txt}</div>'
                 f'<div style="color:var(--crit)">{flag_txt}</div></div>')
    html += '</div></div>'

    html += _HTML_FOOT
    return html


# ─── Dataset summary statistics ───────────────────────────────────────────────

def compute_summary_stats(results: List[dict]) -> dict:
    """Compute dataset-level summary from list of to_dict() results."""
    numeric_fields = [
        'canal_ap_mm', 'canal_dsca_mm2', 'mscc_proxy',
        'lft_proxy_mm', 'facet_tropism_deg', 'min_inter_process_gap_mm',
        'cord_max_mscc',
        *[f'{lm[2]}_{lm[3]}_dhi_pct'          for lm in LUMBAR_PAIRS],
        *[f'{lm[2]}_{lm[3]}_sagittal_translation_mm' for lm in LUMBAR_PAIRS],
        *[f'{lm[2]}_{lm[3]}_wedge_ha_hp'       for lm in LUMBAR_PAIRS],
    ]
    summary = {'n_studies': len(results)}
    df_flat = pd.DataFrame(results)
    for fld in numeric_fields:
        col_key = fld.replace('-','_')
        if col_key not in df_flat.columns: continue
        col = pd.to_numeric(df_flat[col_key], errors='coerce').dropna()
        if col.empty: continue
        summary[col_key] = {
            'mean': round(float(col.mean()), 3),
            'std':  round(float(col.std()),  3),
            'min':  round(float(col.min()),  3),
            'max':  round(float(col.max()),  3),
            'n':    int(col.count()),
        }
    # Prevalence counts
    summary['baastrup_contact_pct'] = (
        round(df_flat.get('baastrup_contact', pd.Series()).eq(True).mean()*100, 1)
        if 'baastrup_contact' in df_flat else None)
    summary['canal_absolute_stenosis_pct'] = (
        round(df_flat.get('canal_absolute_stenosis', pd.Series()).eq(True).mean()*100, 1)
        if 'canal_absolute_stenosis' in df_flat else None)
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description='Standalone spine morphometrics — no 3D rendering'
    )
    pa.add_argument('--spineps_dir',    required=True)
    pa.add_argument('--totalspine_dir', required=True)
    pa.add_argument('--output_dir',     required=True)
    pa.add_argument('--study_id', default=None,
        help='Single study ID to process')
    pa.add_argument('--all', action='store_true',
        help='Process every study with SPINEPS segmentation')
    pa.add_argument('--uncertainty_csv', default=None)
    pa.add_argument('--valid_ids',       default=None)
    pa.add_argument('--top_n',    type=int, default=None)
    pa.add_argument('--rank_by',  default='l5_s1_confidence')
    pa.add_argument('--lstv_json', default=None,
        help='Optional: load LSTV results from 04_detect_lstv.py for report annotation')
    pa.add_argument('--no_reports', action='store_true',
        help='Skip per-study HTML report generation (faster for large datasets)')
    args = pa.parse_args()

    spineps_dir    = Path(args.spineps_dir)
    totalspine_dir = Path(args.totalspine_dir)
    output_dir     = Path(args.output_dir)
    reports_dir    = output_dir / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_reports: reports_dir.mkdir(exist_ok=True)

    seg_root = spineps_dir / 'segmentations'

    # Load LSTV results if available
    lstv_by_id: Dict[str, dict] = {}
    if args.lstv_json:
        p = Path(args.lstv_json)
        if p.exists():
            with open(p) as f:
                lstv_by_id = {str(r['study_id']): r for r in json.load(f)}
            logger.info(f"Loaded {len(lstv_by_id)} LSTV results from {p}")

    # Study selection
    if args.study_id:
        study_ids = [args.study_id]
    elif args.all:
        study_ids = sorted(d.name for d in seg_root.iterdir() if d.is_dir())
        logger.info(f"ALL mode: {len(study_ids)} studies")
    else:
        if not args.uncertainty_csv or args.top_n is None:
            pa.error("--uncertainty_csv + --top_n required unless --all or --study_id")
        valid_ids = None
        if args.valid_ids:
            valid_ids = set(str(x) for x in np.load(args.valid_ids))
        study_ids = select_studies(Path(args.uncertainty_csv), args.top_n,
                                   args.rank_by, valid_ids)
        study_ids = [s for s in study_ids if (seg_root/s).is_dir()]
        logger.info(f"Selective mode: {len(study_ids)} studies")

    all_results_flat: List[dict] = []
    all_results_full: List[dict] = []
    errors = 0

    for sid in study_ids:
        logger.info(f"\n[{sid}]")
        try:
            masks = load_study_masks(sid, spineps_dir, totalspine_dir)
            res   = run_all_morphometrics(masks)

            flat = res.to_dict()
            flat['study_id'] = sid
            all_results_flat.append(flat)

            # Full serialisable version (includes cord profile slices)
            import dataclasses
            full = dataclasses.asdict(res)
            # The cord compression profile is already a plain dict in the result
            all_results_full.append(full)

            # HTML report
            if not args.no_reports:
                html = build_html_report(res, sid, lstv_by_id.get(sid))
                rp = reports_dir / f"{sid}_report.html"
                rp.write_text(html, encoding='utf-8')
                logger.info(f"  Report → {rp}")

        except Exception as e:
            logger.error(f"  [{sid}] FAILED: {e}")
            logger.debug(traceback.format_exc())
            all_results_flat.append({'study_id': sid, 'error': str(e)})
            all_results_full.append({'study_id': sid, 'error': str(e)})
            errors += 1

    # ── Save outputs ──────────────────────────────────────────────────────────

    # JSON (full — used by 06_visualize_3d_v2.py)
    json_path = output_dir / 'morphometrics_all.json'
    with open(json_path, 'w') as f:
        json.dump(all_results_full, f, indent=2, default=str)
    logger.info(f"\nJSON → {json_path}")

    # CSV (flat — for statistics)
    csv_path = output_dir / 'morphometrics_all.csv'
    df = pd.DataFrame(all_results_flat)
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV  → {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")

    # Summary stats
    valid_flat = [r for r in all_results_flat if 'error' not in r]
    summary = compute_summary_stats(valid_flat)
    summary_path = output_dir / 'morphometrics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary → {summary_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Done: {len(study_ids)-errors}/{len(study_ids)} studies OK | {errors} errors")
    logger.info(f"Outputs: {output_dir}")


if __name__ == '__main__':
    main()
