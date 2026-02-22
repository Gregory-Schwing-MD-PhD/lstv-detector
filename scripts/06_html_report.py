#!/usr/bin/env python3
"""
06_html_report.py — LSTV Pipeline HTML Reports
================================================
Produces two types of self-contained HTML reports:

1. LSTV Classification Report (default):
   From lstv_results.json — distribution, representative cases, full table.

2. Dataset Morphometrics Summary (--morpho_only or --morphometrics_json):
   From morphometrics_all.json — means, std, frequencies, histograms, pathology
   prevalence tables across the entire dataset. No images required.

Both modes can be combined in one call.

Usage
-----
  # LSTV report only:
  python 06_html_report.py \
      --lstv_json   results/lstv_detection/lstv_results.json \
      --image_dir   results/lstv_visualization \
      --output_html results/lstv_report.html

  # Morphometrics summary only:
  python 06_html_report.py \
      --morphometrics_json results/morphometrics/morphometrics_all.json \
      --output_html        results/dataset_morphometrics_report.html \
      --morpho_only

  # Both in one file:
  python 06_html_report.py \
      --lstv_json          results/lstv_detection/lstv_results.json \
      --image_dir          results/lstv_visualization \
      --morphometrics_json results/morphometrics/morphometrics_all.json \
      --output_html        results/full_report.html
"""

import argparse
import base64
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

CASTELLVI_ORDER  = ['Type I', 'Type II', 'Type III', 'Type IV']
CASTELLVI_COLORS = {
    'Type I':   '#3a86ff',
    'Type II':  '#ff9f1c',
    'Type III': '#e63946',
    'Type IV':  '#9d0208',
    'None':     '#444466',
    'N/A':      '#444466',
}
CONFIDENCE_COLORS = {
    'high':     '#2dc653',
    'moderate': '#f4a261',
    'low':      '#e76f51',
}
CONFIDENCE_RANK = {'high': 3, 'moderate': 2, 'low': 1}

# Morphometric thresholds for traffic-light colouring in the summary tables
THRESHOLDS = {
    'canal_ap_mm':              {'crit': (None, 7.0),  'warn': (None, 12.0)},
    'canal_dsca_mm2':           {'crit': (None, 70.0), 'warn': (None, 100.0)},
    'mscc_proxy':               {'crit': (0.80, None),  'warn': (0.67, None)},
    'lft_proxy_mm':             {'crit': (5.0, None),   'warn': (4.0, None)},
    'facet_tropism_deg':        {'crit': (10.0, None),  'warn': (7.0, None)},
    'min_inter_process_gap_mm': {'crit': (None, 0.0),   'warn': (None, 2.0)},
}
# Per-level fields that follow the same threshold rules
LEVEL_NAMES = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
LEVEL_DISPLAY = {'L1_L2':'L1-L2','L2_L3':'L2-L3','L3_L4':'L3-L4',
                 'L4_L5':'L4-L5','L5_S1':'L5-S1'}
DHI_CRIT  = 50.0
DHI_WARN  = 70.0
SPONDY_MM = 3.0


# ============================================================================
# UTILITY
# ============================================================================

def _finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _mean(vals):
    v = [float(x) for x in vals if _finite(x)]
    return sum(v) / len(v) if v else None


def _std(vals):
    v = [float(x) for x in vals if _finite(x)]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def _fmt(v, unit='mm', decimals=1):
    if v is None:
        return 'N/A'
    try:
        s = f'{float(v):.{decimals}f}'
        return f'{s} {unit}'.strip() if unit else s
    except (TypeError, ValueError):
        return 'N/A'


def _pct(count, total):
    return f'{100*count/max(total,1):.1f}%'


def _tl_class(field, value):
    """Return 'crit', 'warn', or 'ok' based on threshold table."""
    if not _finite(value):
        return ''
    t = THRESHOLDS.get(field)
    if not t:
        return ''
    lo, hi = t['crit']
    if (lo is not None and float(value) >= lo) or \
       (hi is not None and float(value) <= hi):
        return 'crit'
    lo, hi = t['warn']
    if (lo is not None and float(value) >= lo) or \
       (hi is not None and float(value) <= hi):
        return 'warn'
    return 'ok'


def _collect(records, key):
    """Collect all finite numeric values for a key across records."""
    vals = []
    for r in records:
        v = r.get(key)
        if _finite(v):
            vals.append(float(v))
    return vals


def _collect_level(records, level, suffix):
    """Collect values for per-level keys, e.g. L1_L2_dhi_pct."""
    return _collect(records, f'{level}_{suffix}')


def _prevalence(records, bool_key):
    """Count of True values for a boolean field."""
    return sum(1 for r in records if r.get(bool_key) is True)


# ============================================================================
# LSTV STATS (unchanged from original)
# ============================================================================

def side_vals(result, key):
    out = []
    for side in ('left', 'right'):
        sd = result.get(side) or {}
        v  = sd.get(key)
        if _finite(v):
            out.append(float(v))
    return out


def p2_side_vals(result, key):
    out = []
    for side in ('left', 'right'):
        p2 = (result.get(side) or {}).get('phase2') or {}
        v  = p2.get(key)
        if _finite(v):
            out.append(float(v))
    return out


def compute_lstv_stats(results):
    total       = len(results)
    lstv_count  = sum(1 for r in results if r.get('lstv_detected'))
    error_count = sum(1 for r in results if r.get('errors'))
    l6_count    = sum(1 for r in results if r.get('details', {}).get('has_l6'))
    p2_avail    = sum(1 for r in results if r.get('details', {}).get('phase2_available'))
    p2_valid_ct = 0
    for r in results:
        for side in ('left', 'right'):
            if (r.get(side) or {}).get('phase2', {}).get('p2_valid'):
                p2_valid_ct += 1
                break

    by_class = defaultdict(lambda: {
        'count': 0, 'heights': [], 'dists': [], 'ax_dists': [], 'tv_names': []
    })
    for r in results:
        ct = r.get('castellvi_type') or 'None'
        by_class[ct]['count'] += 1
        by_class[ct]['heights'].extend(side_vals(r, 'tp_height_mm'))
        by_class[ct]['dists'].extend(side_vals(r, 'dist_mm'))
        by_class[ct]['ax_dists'].extend(p2_side_vals(r, 'axial_dist_mm'))
        tv = r.get('details', {}).get('tv_name', '')
        if tv:
            by_class[ct]['tv_names'].append(tv)

    morpho = {}
    for ct, d in by_class.items():
        morpho[ct] = {
            'mean_h':   _mean(d['heights']),
            'mean_d':   _mean(d['dists']),
            'mean_axd': _mean(d['ax_dists']),
            'min_h':    min(d['heights'])   if d['heights']   else None,
            'max_h':    max(d['heights'])   if d['heights']   else None,
            'min_d':    min(d['dists'])     if d['dists']     else None,
            'max_d':    max(d['dists'])     if d['dists']     else None,
            'l6_frac':  (sum(1 for t in d['tv_names'] if t == 'L6') /
                         len(d['tv_names'])) if d['tv_names'] else None,
        }

    return {
        'overall': {
            'total':          total,
            'lstv_count':     lstv_count,
            'error_count':    error_count,
            'lstv_rate':      100 * lstv_count / max(total, 1),
            'l6_count':       l6_count,
            'p2_avail':       p2_avail,
            'p2_valid_count': p2_valid_ct,
        },
        'by_class': dict(by_class),
        'morpho':   morpho,
    }


def pick_representatives(results, image_dir, n_reps=3):
    by_class = defaultdict(list)
    for r in results:
        ct       = r.get('castellvi_type') or 'None'
        img_path = image_dir / f"{r.get('study_id','')}_lstv_overlay.png"
        if img_path.exists():
            by_class[ct].append(r)

    reps = {}
    for ct, group in by_class.items():
        def _key(r):
            conf = CONFIDENCE_RANK.get(r.get('confidence', 'low'), 0)
            dist = min(
                r.get('left',  {}).get('dist_mm', 999) or 999,
                r.get('right', {}).get('dist_mm', 999) or 999,
            )
            return (-conf, dist)
        group.sort(key=_key)
        reps[ct] = group[:n_reps]
    return reps


# ============================================================================
# MORPHOMETRICS DATASET SUMMARY
# ============================================================================

def flatten_morpho_records(records: list) -> list:
    """
    Flatten nested morphometrics JSON records into per-study flat dicts.
    Handles both the 'flat' format and the nested 'levels' list.
    """
    flat_records = []
    for r in records:
        if r.get('error'):
            continue
        flat = dict(r)  # copy top-level fields
        # Flatten per-level data from levels list
        for lm in r.get('levels', []):
            lvl = lm.get('level', '')  # e.g. "L1_L2"
            for k, v in lm.items():
                if k != 'level' and k != 'level_display':
                    flat[f'{lvl}_{k}'] = v
        # Flatten cord_compression_profile summary fields
        cp = r.get('cord_compression_profile')
        if cp:
            flat['cord_max_mscc']           = cp.get('max_mscc')
            flat['cord_classification']     = cp.get('classification')
            flat['cord_flagged_count']      = cp.get('flagged_count', 0)
        flat_records.append(flat)
    return flat_records


def compute_morpho_summary(records: list) -> dict:
    """Compute dataset-level descriptive statistics from flat morpho records."""
    n = len(records)
    if n == 0:
        return {'n': 0}

    def stats(vals):
        if not vals:
            return {'n': 0, 'mean': None, 'std': None, 'min': None,
                    'median': None, 'max': None, 'missing_pct': 100.0}
        arr = sorted(vals)
        m   = _mean(arr)
        s   = _std(arr)
        med = arr[len(arr) // 2] if arr else None
        missing = round(100 * (n - len(arr)) / n, 1)
        return {
            'n':           len(arr),
            'mean':        round(m, 3) if m is not None else None,
            'std':         round(s, 3) if s is not None else None,
            'min':         round(min(arr), 3),
            'median':      round(med, 3),
            'max':         round(max(arr), 3),
            'missing_pct': missing,
        }

    result = {'n': n}

    # ── Global canal / cord ────────────────────────────────────────────────
    result['canal_ap_mm']   = stats(_collect(records, 'canal_ap_mm'))
    result['canal_dsca_mm2']= stats(_collect(records, 'canal_dsca_mm2'))
    result['mscc_proxy']    = stats(_collect(records, 'mscc_proxy'))
    result['cord_max_mscc'] = stats(_collect(records, 'cord_max_mscc'))
    result['cord_ap_mm']    = stats(_collect(records, 'cord_ap_mm'))
    result['cord_csa_mm2']  = stats(_collect(records, 'cord_csa_mm2'))

    # ── Ligamentum flavum / Baastrup / facet ───────────────────────────────
    result['lft_proxy_mm']             = stats(_collect(records, 'lft_proxy_mm'))
    result['min_inter_process_gap_mm'] = stats(_collect(records, 'min_inter_process_gap_mm'))
    result['facet_tropism_deg']        = stats(_collect(records, 'facet_tropism_deg'))
    result['facet_angle_l_deg']        = stats(_collect(records, 'facet_angle_l_deg'))
    result['facet_angle_r_deg']        = stats(_collect(records, 'facet_angle_r_deg'))

    # ── Per-level disc / vertebral / spondylolisthesis ─────────────────────
    result['levels'] = {}
    for lvl in LEVEL_NAMES:
        result['levels'][lvl] = {
            'dhi_pct':                 stats(_collect_level(records, lvl, 'dhi_pct')),
            'sagittal_translation_mm': stats(_collect_level(records, lvl, 'sagittal_translation_mm')),
            'endplate_dist_mm':        stats(_collect_level(records, lvl, 'endplate_dist_mm')),
            'level_ap_mm':             stats(_collect_level(records, lvl, 'level_ap_mm')),
            'wedge_ha_hp':             stats(_collect_level(records, lvl, 'wedge_ha_hp')),
            'compression_hm_ha':       stats(_collect_level(records, lvl, 'compression_hm_ha')),
        }

    # ── Pathology prevalences ──────────────────────────────────────────────
    prev = {}

    # Canal stenosis
    prev['canal_absolute_stenosis']   = _prevalence(records, 'canal_absolute_stenosis')
    prev['canal_absolute_stenosis_pct'] = round(
        100 * prev['canal_absolute_stenosis'] / n, 1)

    # Baastrup
    prev['baastrup_contact']     = _prevalence(records, 'baastrup_contact')
    prev['baastrup_risk']        = _prevalence(records, 'baastrup_risk')
    prev['baastrup_contact_pct'] = round(100 * prev['baastrup_contact'] / n, 1)
    prev['baastrup_risk_pct']    = round(100 * prev['baastrup_risk']    / n, 1)

    # Spondylolisthesis (any level ≥ 3 mm)
    spondy_any = 0
    for r in records:
        for lvl in LEVEL_NAMES:
            t = r.get(f'{lvl}_sagittal_translation_mm')
            if _finite(t) and float(t) >= SPONDY_MM:
                spondy_any += 1
                break
    prev['spondylolisthesis_any']     = spondy_any
    prev['spondylolisthesis_any_pct'] = round(100 * spondy_any / n, 1)

    # Per-level spondy
    for lvl in LEVEL_NAMES:
        ct = sum(1 for r in records
                 if _finite(r.get(f'{lvl}_sagittal_translation_mm'))
                 and float(r[f'{lvl}_sagittal_translation_mm']) >= SPONDY_MM)
        prev[f'spondy_{lvl}']     = ct
        prev[f'spondy_{lvl}_pct'] = round(100 * ct / n, 1)

    # Severe DHI (any level < 50 %)
    severe_dhi_any = 0
    for r in records:
        for lvl in LEVEL_NAMES:
            d = r.get(f'{lvl}_dhi_pct')
            if _finite(d) and float(d) < DHI_CRIT:
                severe_dhi_any += 1
                break
    prev['severe_dhi_any']     = severe_dhi_any
    prev['severe_dhi_any_pct'] = round(100 * severe_dhi_any / n, 1)

    # Per-level severe DHI
    for lvl in LEVEL_NAMES:
        ct = sum(1 for r in records
                 if _finite(r.get(f'{lvl}_dhi_pct'))
                 and float(r[f'{lvl}_dhi_pct']) < DHI_CRIT)
        prev[f'severe_dhi_{lvl}']     = ct
        prev[f'severe_dhi_{lvl}_pct'] = round(100 * ct / n, 1)

    # Cord compression
    cord_mild_plus = sum(
        1 for r in records
        if r.get('cord_classification') in ('Mild', 'Moderate', 'Severe'))
    cord_mod_plus  = sum(
        1 for r in records
        if r.get('cord_classification') in ('Moderate', 'Severe'))
    prev['cord_compression_any']     = cord_mild_plus
    prev['cord_compression_any_pct'] = round(100 * cord_mild_plus / n, 1)
    prev['cord_moderate_plus']       = cord_mod_plus
    prev['cord_moderate_plus_pct']   = round(100 * cord_mod_plus / n, 1)

    # LFT hypertrophy (> 4 mm proxy)
    lft_hyp = sum(1 for r in records
                  if _finite(r.get('lft_proxy_mm')) and float(r['lft_proxy_mm']) > 4.0)
    prev['lft_hypertrophy']     = lft_hyp
    prev['lft_hypertrophy_pct'] = round(100 * lft_hyp / n, 1)

    # Facet tropism grade ≥ 10° (Grade 2)
    ft_severe = sum(1 for r in records
                    if _finite(r.get('facet_tropism_deg'))
                    and float(r['facet_tropism_deg']) >= 10.0)
    prev['facet_tropism_severe']     = ft_severe
    prev['facet_tropism_severe_pct'] = round(100 * ft_severe / n, 1)

    # Canal shape distribution
    shape_counts = defaultdict(int)
    for r in records:
        for lvl in LEVEL_NAMES:
            sh = r.get(f'{lvl}_canal_shape')
            if sh:
                shape_counts[sh.split('(')[0].strip()] += 1
    prev['canal_shapes'] = dict(shape_counts)

    result['prevalence'] = prev
    return result


# ============================================================================
# CSS (shared)
# ============================================================================

CSS = """
:root {
  --bg:      #0d0d1a;
  --surface: #161628;
  --border:  #2a2a4a;
  --text:    #e8e8f0;
  --muted:   #8888aa;
  --accent:  #3a86ff;
  --ok:      #2dc653;
  --warn:    #f0a500;
  --crit:    #e63946;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px; padding: 28px;
  max-width: 1800px; margin: 0 auto;
}
h1 { font-size: 1.9rem; color: var(--accent); margin-bottom: 4px; }
h2 { font-size: 1.25rem; color: var(--accent); margin: 32px 0 12px;
     border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { font-size: 1.0rem; color: var(--muted); margin: 18px 0 6px; font-weight: 600; }
.subtitle { color: var(--muted); margin-bottom: 20px; }

/* nav tabs */
.tab-bar { display: flex; gap: 4px; margin-bottom: 28px; flex-wrap: wrap; }
.tab-btn {
  background: var(--surface); border: 1px solid var(--border);
  color: var(--muted); padding: 7px 18px; border-radius: 6px;
  cursor: pointer; font-size: .85rem; font-family: inherit;
}
.tab-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* summary stat cards */
.stats-bar {
  display: flex; gap: 14px; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; margin-bottom: 24px;
}
.stat { display: flex; flex-direction: column; min-width: 85px; }
.stat .val { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.stat .lbl { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing:.04em; }

/* pathology prevalence cards */
.prev-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  gap: 12px; margin-bottom: 24px;
}
.prev-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px; position: relative; overflow: hidden;
}
.prev-card .pc-label { font-size: .72rem; color: var(--muted);
  text-transform: uppercase; letter-spacing:.05em; margin-bottom: 4px; }
.prev-card .pc-count { font-size: 1.8rem; font-weight: 700; line-height: 1; }
.prev-card .pc-pct   { font-size: .85rem; color: var(--muted); margin-top: 2px; }
.prev-card .pc-bar   { height: 4px; border-radius: 2px; margin-top: 10px; }
.prev-card.ok  .pc-count { color: var(--ok); }
.prev-card.warn .pc-count { color: var(--warn); }
.prev-card.crit .pc-count { color: var(--crit); }
.prev-card.ok  .pc-bar { background: var(--ok); }
.prev-card.warn .pc-bar { background: var(--warn); }
.prev-card.crit .pc-bar { background: var(--crit); }

/* stats tables */
table.st {
  width: 100%; border-collapse: collapse; font-size: .85rem; margin-bottom: 24px;
}
table.st th {
  text-align: left; color: var(--muted); padding: 6px 10px;
  border-bottom: 2px solid var(--border); font-weight: 600; white-space: nowrap;
  background: var(--surface);
}
table.st td { padding: 7px 10px; border-bottom: 1px solid #1e1e36; }
table.st tr:hover td { background: #1a1a2e; }
table.st td.ok   { color: var(--ok);   font-weight: 600; }
table.st td.warn { color: var(--warn); font-weight: 600; }
table.st td.crit { color: var(--crit); font-weight: 600; }
table.st td.num  { font-family: 'Courier New', monospace; }

/* per-level breakdown table */
.level-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 16px; margin-bottom: 24px;
}
.level-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden;
}
.level-card .lc-hdr {
  background: #1a1a3a; padding: 7px 12px;
  font-weight: 700; font-size: .9rem; color: #c8c8f0;
}
.level-card table { width: 100%; border-collapse: collapse; font-size: .82rem; }
.level-card table td { padding: 5px 10px; border-bottom: 1px solid #1e1e36; }
.level-card table td:first-child { color: var(--muted); }
.level-card table td:last-child  { text-align: right;
  font-family: 'Courier New', monospace; font-weight: 600; }
.level-card table tr:last-child td { border-bottom: none; }

/* inline bar chart for distributions */
.dist-bar-wrap { margin-bottom: 24px; }
.dist-bar-row  { display: flex; align-items: center; gap: 8px; margin-bottom: 5px;
                 font-size: .8rem; }
.dist-bar-label { width: 120px; flex-shrink: 0; text-align: right; color: var(--muted); }
.dist-bar-track { flex: 1; height: 16px; background: #1a1a2e; border-radius: 3px;
                  overflow: hidden; }
.dist-bar-fill  { height: 100%; border-radius: 3px; transition: width .4s; }
.dist-bar-val   { width: 60px; flex-shrink: 0; color: var(--text); }

/* badge */
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 0.75rem; font-weight: 600; color: #fff;
}

/* distribution cards (LSTV) */
.dist-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 24px; }
.dist-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px; min-width: 150px; flex: 1;
}
.dist-card .ct-name  { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
.dist-card .ct-count { font-size: 2rem; font-weight: 700; }
.dist-card .ct-pct   { color: var(--muted); font-size: 0.85rem; }
.dist-card .ct-bar   { height: 5px; border-radius: 3px; margin-top: 10px; }

table.morpho {
  width: 100%; border-collapse: collapse; margin-bottom: 28px; font-size: 0.87rem;
}
table.morpho th {
  text-align: left; color: var(--muted); padding: 6px 10px;
  border-bottom: 2px solid var(--border); font-weight: 600; white-space: nowrap;
}
table.morpho td { padding: 8px 10px; border-bottom: 1px solid #1e1e36; }
table.morpho tr:last-child td { border-bottom: none; }
.ct-badge {
  display: inline-block; padding: 2px 10px; border-radius: 10px;
  font-size: 0.78rem; font-weight: 600; color: #fff;
}

.rep-section { margin-bottom: 36px; }
.rep-class-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.rep-grid { display: flex; gap: 14px; flex-wrap: wrap; }
.rep-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden; flex: 1; min-width: 320px; max-width: 580px;
}
.rep-card img  { width: 100%; height: auto; display: block; }
.rep-card .rep-meta { padding: 10px 12px; font-size: 0.82rem; color: var(--muted); }
.rep-card .rep-meta strong { color: var(--text); }

table.results {
  width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-bottom: 32px;
}
table.results th {
  text-align: left; color: var(--muted); padding: 6px 8px;
  border-bottom: 2px solid var(--border); font-weight: 600;
  position: sticky; top: 0; background: var(--bg); z-index: 1;
}
table.results td { padding: 6px 8px; border-bottom: 1px solid #1e1e36; }
table.results tr:hover td { background: #1a1a2e; }
.p2-cell { font-size: 0.78rem; color: var(--muted); }

footer {
  text-align: center; color: var(--muted); font-size: 0.8rem;
  margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
}
"""


def _badge(text, color):
    return f'<span class="badge" style="background:{color}">{text}</span>'


def _stat_row(label, s, field='', unit='', note=''):
    """HTML row for a descriptive statistics dict."""
    if not s or s.get('n', 0) == 0:
        return f'<tr><td>{label}</td><td class="num" colspan="6">N/A</td><td>{note}</td></tr>'
    mean_val = s.get('mean')
    std_val  = s.get('std')
    tl = _tl_class(field, mean_val) if field else ''
    mean_s = _fmt(mean_val, unit) if mean_val is not None else 'N/A'
    std_s  = f'±{_fmt(std_val,"",2)}' if std_val is not None else '—'
    return (f'<tr><td>{label}</td>'
            f'<td class="num {tl}">{mean_s}</td>'
            f'<td class="num">{std_s}</td>'
            f'<td class="num">{_fmt(s.get("min"),unit)}</td>'
            f'<td class="num">{_fmt(s.get("median"),unit)}</td>'
            f'<td class="num">{_fmt(s.get("max"),unit)}</td>'
            f'<td class="num">{s["n"]} ({100-s.get("missing_pct",0):.0f}% valid)</td>'
            f'<td style="font-size:.78rem;color:var(--muted)">{note}</td></tr>')


def _prev_card(label, count, pct, color_class, max_pct=100):
    bar_w = min(100, float(pct.rstrip('%')))
    return (f'<div class="prev-card {color_class}">'
            f'<div class="pc-label">{label}</div>'
            f'<div class="pc-count">{count}</div>'
            f'<div class="pc-pct">{pct} of dataset</div>'
            f'<div class="pc-bar" style="width:{bar_w:.0f}%"></div>'
            f'</div>')


def _prev_color(pct_str):
    try:
        p = float(pct_str.rstrip('%'))
        if p >= 30: return 'crit'
        if p >= 10: return 'warn'
        return 'ok'
    except Exception:
        return 'ok'


# ============================================================================
# MORPHOMETRICS SUMMARY HTML SECTIONS
# ============================================================================

def section_morpho_overview(summary):
    n = summary['n']
    html = '<div class="stats-bar">'
    # Errors
    items = [
        (n,                   'Studies analysed'),
        (summary['prevalence']['canal_absolute_stenosis'],
         'Canal abs. stenosis'),
        (summary['prevalence']['baastrup_contact'],   'Baastrup contact'),
        (summary['prevalence']['spondylolisthesis_any'], 'Spondylolisthesis (any level)'),
        (summary['prevalence']['severe_dhi_any'],     'Severe DHI (any level)'),
        (summary['prevalence']['cord_compression_any'], 'Cord compression (any)'),
        (summary['prevalence']['lft_hypertrophy'],    'LFT hypertrophy >4mm'),
        (summary['prevalence']['facet_tropism_severe'], 'Facet tropism ≥10°'),
    ]
    for val, lbl in items:
        html += (f'<div class="stat"><span class="val">{val}</span>'
                 f'<span class="lbl">{lbl}</span></div>')
    html += '</div>'
    return html


def section_pathology_prevalence(summary):
    p = summary['prevalence']
    n = summary['n']
    html = '<h2>Pathology Prevalence</h2>'

    # Global pathologies
    html += '<div class="prev-grid">'
    cards = [
        ('Canal Absolute Stenosis', p['canal_absolute_stenosis'],
         f'{p["canal_absolute_stenosis_pct"]}%'),
        ('Baastrup Contact',        p['baastrup_contact'],
         f'{p["baastrup_contact_pct"]}%'),
        ('Baastrup Risk Zone',      p['baastrup_risk'],
         f'{p["baastrup_risk_pct"]}%'),
        ('Spondylolisthesis (any)', p['spondylolisthesis_any'],
         f'{p["spondylolisthesis_any_pct"]}%'),
        ('Severe DHI Any Level',    p['severe_dhi_any'],
         f'{p["severe_dhi_any_pct"]}%'),
        ('Cord Compression Any',    p['cord_compression_any'],
         f'{p["cord_compression_any_pct"]}%'),
        ('Cord Moderate/Severe',    p['cord_moderate_plus'],
         f'{p["cord_moderate_plus_pct"]}%'),
        ('LFT Hypertrophy >4mm',    p['lft_hypertrophy'],
         f'{p["lft_hypertrophy_pct"]}%'),
        ('Facet Tropism ≥10° (G2)', p['facet_tropism_severe'],
         f'{p["facet_tropism_severe_pct"]}%'),
    ]
    for label, count, pct in cards:
        html += _prev_card(label, count, pct, _prev_color(pct))
    html += '</div>'

    # Per-level spondylolisthesis bar chart
    html += '<h3>Spondylolisthesis Rate by Level (≥3 mm translation)</h3>'
    html += '<div class="dist-bar-wrap">'
    max_ct = max((p.get(f'spondy_{l}', 0) for l in LEVEL_NAMES), default=1)
    for lvl in LEVEL_NAMES:
        ct  = p.get(f'spondy_{lvl}', 0)
        pct = p.get(f'spondy_{lvl}_pct', 0)
        w   = int(100 * ct / max(max_ct, 1))
        html += (f'<div class="dist-bar-row">'
                 f'<div class="dist-bar-label">{LEVEL_DISPLAY[lvl]}</div>'
                 f'<div class="dist-bar-track"><div class="dist-bar-fill" '
                 f'style="width:{w}%;background:var(--crit)"></div></div>'
                 f'<div class="dist-bar-val">{ct} ({pct}%)</div>'
                 f'</div>')
    html += '</div>'

    # Per-level severe DHI bar chart
    html += '<h3>Severe Disc Degeneration Rate by Level (DHI &lt;50%)</h3>'
    html += '<div class="dist-bar-wrap">'
    max_ct = max((p.get(f'severe_dhi_{l}', 0) for l in LEVEL_NAMES), default=1)
    for lvl in LEVEL_NAMES:
        ct  = p.get(f'severe_dhi_{lvl}', 0)
        pct = p.get(f'severe_dhi_{lvl}_pct', 0)
        w   = int(100 * ct / max(max_ct, 1))
        html += (f'<div class="dist-bar-row">'
                 f'<div class="dist-bar-label">{LEVEL_DISPLAY[lvl]}</div>'
                 f'<div class="dist-bar-track"><div class="dist-bar-fill" '
                 f'style="width:{w}%;background:var(--warn)"></div></div>'
                 f'<div class="dist-bar-val">{ct} ({pct}%)</div>'
                 f'</div>')
    html += '</div>'

    # Canal shape distribution
    shapes = p.get('canal_shapes', {})
    if shapes:
        html += '<h3>Canal Shape Distribution (all levels combined)</h3>'
        html += '<div class="dist-bar-wrap">'
        total_shapes = sum(shapes.values())
        for sh, ct in sorted(shapes.items(), key=lambda x: -x[1]):
            pct_f = 100 * ct / max(total_shapes, 1)
            html += (f'<div class="dist-bar-row">'
                     f'<div class="dist-bar-label">{sh}</div>'
                     f'<div class="dist-bar-track"><div class="dist-bar-fill" '
                     f'style="width:{pct_f:.0f}%;background:var(--accent)"></div></div>'
                     f'<div class="dist-bar-val">{ct} ({pct_f:.1f}%)</div>'
                     f'</div>')
        html += '</div>'

    return html


def _st_table_header():
    return ('<table class="st"><thead><tr>'
            '<th>Measurement</th><th>Mean</th><th>Std Dev</th>'
            '<th>Min</th><th>Median</th><th>Max</th>'
            '<th>Valid N</th><th>Reference / Note</th>'
            '</tr></thead><tbody>')


def section_global_measurements(summary):
    html = '<h2>Global Measurements — Descriptive Statistics</h2>'
    html += _st_table_header()
    html += _stat_row('Canal AP diameter', summary['canal_ap_mm'],
        'canal_ap_mm', 'mm', 'Normal &gt;12mm | Absolute &lt;7mm')
    html += _stat_row('Canal DSCA (estimated)', summary['canal_dsca_mm2'],
        'canal_dsca_mm2', 'mm²', 'Normal &gt;100mm² | Absolute &lt;70mm²')
    html += _stat_row('MSCC proxy (global)', summary['mscc_proxy'],
        'mscc_proxy', '', '≥0.67 → compression | ≥0.80 → severe')
    html += _stat_row('Cord max MSCC', summary['cord_max_mscc'],
        'mscc_proxy', '', 'Per-study maximum along full cord length')
    html += _stat_row('Cord AP diameter', summary['cord_ap_mm'], '', 'mm', '')
    html += _stat_row('Cord CSA', summary['cord_csa_mm2'], '', 'mm²', '')
    html += _stat_row('LFT proxy', summary['lft_proxy_mm'],
        'lft_proxy_mm', 'mm', 'Hypertrophy ≥4mm | Severe &gt;5mm')
    html += _stat_row('Min inter-spinous gap', summary['min_inter_process_gap_mm'],
        'min_inter_process_gap_mm', 'mm', 'Baastrup contact ≤0mm | Risk ≤2mm')
    html += _stat_row('Facet tropism angle', summary['facet_tropism_deg'],
        'facet_tropism_deg', '°', 'Grade 2 ≥10° (spondylolisthesis risk)')
    html += _stat_row('Facet angle Left',  summary['facet_angle_l_deg'],  '', '°', '')
    html += _stat_row('Facet angle Right', summary['facet_angle_r_deg'], '', '°', '')
    html += '</tbody></table>'
    return html


def section_level_measurements(summary):
    html = '<h2>Per-Level Measurements — Descriptive Statistics</h2>'

    # DHI table
    html += '<h3>Disc Height Index (DHI %) — Farfan Method</h3>'
    html += _st_table_header()
    for lvl in LEVEL_NAMES:
        s = summary['levels'][lvl].get('dhi_pct', {})
        note = 'Severe &lt;50% | Moderate &lt;70% | Mild &lt;85%'
        m = s.get('mean')
        tl = 'crit' if (m and m < DHI_CRIT) else 'warn' if (m and m < DHI_WARN) else 'ok'
        html += (f'<tr><td>{LEVEL_DISPLAY[lvl]}</td>'
                 f'<td class="num {tl}">{_fmt(m, "%")}</td>'
                 f'<td class="num">±{_fmt(s.get("std"),"",1)}</td>'
                 f'<td class="num">{_fmt(s.get("min"),"%")}</td>'
                 f'<td class="num">{_fmt(s.get("median"),"%")}</td>'
                 f'<td class="num">{_fmt(s.get("max"),"%")}</td>'
                 f'<td class="num">{s.get("n",0)}</td>'
                 f'<td style="font-size:.78rem;color:var(--muted)">{note}</td></tr>')
    html += '</tbody></table>'

    # Sagittal translation table
    html += '<h3>Sagittal Translation / Spondylolisthesis (mm)</h3>'
    html += _st_table_header()
    for lvl in LEVEL_NAMES:
        s = summary['levels'][lvl].get('sagittal_translation_mm', {})
        m = s.get('mean')
        tl = 'crit' if (m and m >= SPONDY_MM) else 'ok'
        html += (f'<tr><td>{LEVEL_DISPLAY[lvl]}</td>'
                 f'<td class="num {tl}">{_fmt(m,"mm")}</td>'
                 f'<td class="num">±{_fmt(s.get("std"),"",2)}</td>'
                 f'<td class="num">{_fmt(s.get("min"),"mm")}</td>'
                 f'<td class="num">{_fmt(s.get("median"),"mm")}</td>'
                 f'<td class="num">{_fmt(s.get("max"),"mm")}</td>'
                 f'<td class="num">{s.get("n",0)}</td>'
                 f'<td style="font-size:.78rem;color:var(--muted)">Positive ≥3mm</td></tr>')
    html += '</tbody></table>'

    # Level canal AP table
    html += '<h3>Canal AP at Level (mm)</h3>'
    html += _st_table_header()
    for lvl in LEVEL_NAMES:
        s = summary['levels'][lvl].get('level_ap_mm', {})
        m = s.get('mean')
        tl = _tl_class('canal_ap_mm', m) if m else ''
        html += (f'<tr><td>{LEVEL_DISPLAY[lvl]}</td>'
                 f'<td class="num {tl}">{_fmt(m,"mm")}</td>'
                 f'<td class="num">±{_fmt(s.get("std"),"",2)}</td>'
                 f'<td class="num">{_fmt(s.get("min"),"mm")}</td>'
                 f'<td class="num">{_fmt(s.get("median"),"mm")}</td>'
                 f'<td class="num">{_fmt(s.get("max"),"mm")}</td>'
                 f'<td class="num">{s.get("n",0)}</td>'
                 f'<td style="font-size:.78rem;color:var(--muted)">Normal &gt;12mm</td></tr>')
    html += '</tbody></table>'

    # Wedge ratio table
    html += '<h3>Vertebral Wedge Ratio (Ha/Hp) — Genant</h3>'
    html += _st_table_header()
    for lvl in LEVEL_NAMES:
        s = summary['levels'][lvl].get('wedge_ha_hp', {})
        m = s.get('mean')
        tl = 'crit' if (m and m < 0.75) else 'warn' if (m and m < 0.80) else 'ok'
        html += (f'<tr><td>{LEVEL_DISPLAY[lvl]}</td>'
                 f'<td class="num {tl}">{_fmt(m,"",3)}</td>'
                 f'<td class="num">±{_fmt(s.get("std"),"",3)}</td>'
                 f'<td class="num">{_fmt(s.get("min"),"",3)}</td>'
                 f'<td class="num">{_fmt(s.get("median"),"",3)}</td>'
                 f'<td class="num">{_fmt(s.get("max"),"",3)}</td>'
                 f'<td class="num">{s.get("n",0)}</td>'
                 f'<td style="font-size:.78rem;color:var(--muted)">&lt;0.80 fracture | &lt;0.75 intervention</td></tr>')
    html += '</tbody></table>'

    return html


# ============================================================================
# LSTV REPORT SECTIONS (unchanged from original)
# ============================================================================

def section_summary(stats):
    o    = stats['overall']
    html = '<div class="stats-bar">'
    items = [
        (o['total'],             'Total Studies'),
        (o['lstv_count'],        'LSTV Detected'),
        (f"{o['lstv_rate']:.1f}%", 'LSTV Rate'),
        (o['error_count'],       'Errors/Incomplete'),
        (o['l6_count'],          'L6 Present'),
        (o['p2_avail'],          'Phase 2 Available'),
        (o['p2_valid_count'],    'Phase 2 Valid Sides'),
    ]
    for val, lbl in items:
        html += (f'<div class="stat"><span class="val">{val}</span>'
                 f'<span class="lbl">{lbl}</span></div>')
    html += '</div>'
    return html


def section_distribution(stats):
    by_class = stats['by_class']
    total    = stats['overall']['total']
    html     = '<h2>Castellvi Class Distribution</h2><div class="dist-grid">'
    for ct in CASTELLVI_ORDER + ['None']:
        d = by_class.get(ct)
        if not d:
            continue
        count = d['count']
        pct   = 100 * count / max(total, 1)
        color = CASTELLVI_COLORS.get(ct, '#444466')
        html += f"""<div class="dist-card">
  <div class="ct-name" style="color:{color}">{ct}</div>
  <div class="ct-count" style="color:{color}">{count}</div>
  <div class="ct-pct">{pct:.1f}% of all studies</div>
  <div class="ct-bar" style="background:{color};width:{min(pct,100):.0f}%"></div>
</div>"""
    html += '</div>'
    return html


def section_morphometrics_lstv(stats):
    morpho   = stats['morpho']
    by_class = stats['by_class']
    html     = '<h2>Average Morphometrics by Castellvi Class</h2>'
    html += '''<table class="morpho"><thead><tr>
  <th>Class</th><th>N</th>
  <th>TP height mean (range) mm</th>
  <th>TP–Sacrum dist mean (range) mm</th>
  <th>P2 Axial dist mean mm</th>
  <th>L6 rate</th>
</tr></thead><tbody>'''
    for ct in CASTELLVI_ORDER + ['None']:
        m = morpho.get(ct)
        d = by_class.get(ct)
        if not m or not d:
            continue
        color = CASTELLVI_COLORS.get(ct, '#444466')
        def _rng(mn, mx):
            return f'{mn:.1f}–{mx:.1f}' if mn is not None else 'N/A'
        mean_h  = f'{m["mean_h"]:.1f} ({_rng(m["min_h"],m["max_h"])})' if m['mean_h'] else 'N/A'
        mean_d  = f'{m["mean_d"]:.1f} ({_rng(m["min_d"],m["max_d"])})' if m['mean_d'] else 'N/A'
        mean_ax = f'{m["mean_axd"]:.2f}' if m['mean_axd'] else 'N/A'
        l6_frac = f'{100*m["l6_frac"]:.0f}%' if m['l6_frac'] is not None else 'N/A'
        html += f'''<tr>
  <td><span class="ct-badge" style="background:{color}">{ct}</span></td>
  <td>{d["count"]}</td><td>{mean_h}</td><td>{mean_d}</td>
  <td>{mean_ax}</td><td>{l6_frac}</td>
</tr>'''
    html += '</tbody></table>'
    return html


def section_representatives(reps, image_dir):
    html = '<h2>Representative Cases by Castellvi Class</h2>'
    for ct in CASTELLVI_ORDER + ['None']:
        cases = reps.get(ct)
        if not cases:
            continue
        color = CASTELLVI_COLORS.get(ct, '#444466')
        html += f'<div class="rep-section"><div class="rep-class-header">{_badge(ct, color)}<h3>{ct}</h3></div><div class="rep-grid">'
        for r in cases:
            sid    = r.get('study_id', '?')
            conf   = r.get('confidence', 'N/A')
            tv     = r.get('details', {}).get('tv_name', '?')
            dist_L = r.get('left',  {}).get('dist_mm')
            dist_R = r.get('right', {}).get('dist_mm')
            h_L    = r.get('left',  {}).get('tp_height_mm')
            h_R    = r.get('right', {}).get('tp_height_mm')
            p2l    = (r.get('left',  {}).get('phase2') or {})
            p2r    = (r.get('right', {}).get('phase2') or {})
            p2v_L  = '✓' if p2l.get('p2_valid') else '—'
            p2v_R  = '✓' if p2r.get('p2_valid') else '—'
            conf_c = CONFIDENCE_COLORS.get(conf, '#888')
            img_path = image_dir / f"{sid}_lstv_overlay.png"
            b64  = base64.b64encode(open(img_path, 'rb').read()).decode()
            html += (f'<div class="rep-card">'
                     f'<img src="data:image/png;base64,{b64}" loading="lazy">'
                     f'<div class="rep-meta"><strong>{sid}</strong> &nbsp;'
                     f'{_badge(ct, color)} {_badge(conf, conf_c)}'
                     f'&nbsp; TV: {tv}<br>'
                     f'Ht L/R: {_fmt(h_L)}/{_fmt(h_R)} &nbsp;|&nbsp;'
                     f'P1-dist L/R: {_fmt(dist_L)}/{_fmt(dist_R)}<br>'
                     f'P2 valid L/R: {p2v_L}/{p2v_R}</div></div>')
        html += '</div></div>'
    return html


def section_full_table(results):
    lstv = [r for r in results if r.get('lstv_detected')]
    lstv.sort(key=lambda r: (
        -{'Type IV':4,'Type III':3,'Type II':2,'Type I':1}.get(r.get('castellvi_type') or '', 0),
        -CONFIDENCE_RANK.get(r.get('confidence','low'), 0),
    ))
    if not lstv:
        return '<h2>LSTV Cases</h2><p style="color:var(--muted)">No LSTV detected.</p>'

    html = f'<h2>All LSTV Cases ({len(lstv)})</h2>'
    html += '''<table class="results"><thead><tr>
  <th>#</th><th>Study</th><th>Castellvi</th><th>TV</th>
  <th>L ht mm</th><th>R ht mm</th><th>L P1-dist</th><th>R P1-dist</th>
  <th>L class</th><th>R class</th><th>P2-L</th><th>P2-R</th><th>Errors</th>
</tr></thead><tbody>'''

    for i, r in enumerate(lstv, 1):
        sid  = r.get('study_id', '?')
        ct   = r.get('castellvi_type') or 'N/A'
        tv   = r.get('details', {}).get('tv_name', '?')
        ld   = r.get('left',  {}) or {}
        rd   = r.get('right', {}) or {}
        p2l  = ld.get('phase2') or {}
        p2r  = rd.get('phase2') or {}
        errs = '; '.join(r.get('errors', []))
        ct_c = CASTELLVI_COLORS.get(ct, '#444')

        def _p2_cell(p2):
            if not p2.get('phase2_attempted'):
                return '<span class="p2-cell">—</span>'
            valid   = p2.get('p2_valid', False)
            cls     = p2.get('classification', '?')
            ax_dist = p2.get('axial_dist_mm')
            color   = '#2dc653' if valid else '#888888'
            dist_s  = f' ({ax_dist:.2f}mm)' if _finite(ax_dist) else ''
            return (f'<span class="p2-cell" style="color:{color}">'
                    f'{"✓" if valid else "✗"} {cls}{dist_s}</span>')

        html += (f'<tr><td>{i}</td><td><strong>{sid}</strong></td>'
                 f'<td>{_badge(ct, ct_c)}</td><td>{tv}</td>'
                 f'<td>{_fmt(ld.get("tp_height_mm"), "")}</td>'
                 f'<td>{_fmt(rd.get("tp_height_mm"), "")}</td>'
                 f'<td>{_fmt(ld.get("dist_mm"), "")}</td>'
                 f'<td>{_fmt(rd.get("dist_mm"), "")}</td>'
                 f'<td>{ld.get("classification","—")}</td>'
                 f'<td>{rd.get("classification","—")}</td>'
                 f'<td>{_p2_cell(p2l)}</td><td>{_p2_cell(p2r)}</td>'
                 f'<td style="color:#ff8080;font-size:0.78em">{errs}</td></tr>')
    html += '</tbody></table>'
    return html


# ============================================================================
# BUILD
# ============================================================================

def build_report(
    results_path:    Optional[Path],
    morpho_path:     Optional[Path],
    image_dir:       Optional[Path],
    output_html:     Path,
    n_reps:          int  = 3,
    morpho_only:     bool = False,
):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    tabs = []   # list of (tab_id, tab_label, content_html)

    # ── Morphometrics summary tab ─────────────────────────────────────────────
    if morpho_path and morpho_path.exists():
        logger.info(f"Loading morphometrics from {morpho_path}")
        with open(morpho_path) as f:
            morpho_data = json.load(f)
        flat_records = flatten_morpho_records(morpho_data)
        n_ok  = len(flat_records)
        n_err = len(morpho_data) - n_ok
        logger.info(f"  {n_ok} valid records, {n_err} errors/skipped")
        summary = compute_morpho_summary(flat_records)

        content = ''
        content += section_morpho_overview(summary)
        content += section_pathology_prevalence(summary)
        content += section_global_measurements(summary)
        content += section_level_measurements(summary)
        content += (f'<p style="font-size:.75rem;color:var(--muted);margin-top:8px">'
                    f'{n_err} studies skipped due to errors/missing masks.</p>')
        tabs.append(('morpho', f'Dataset Summary (n={n_ok})', content))

    # ── LSTV tabs ─────────────────────────────────────────────────────────────
    if not morpho_only and results_path and results_path.exists():
        logger.info(f"Loading LSTV results from {results_path}")
        with open(results_path) as f:
            all_results = json.load(f)
        logger.info(f"  {len(all_results)} LSTV results loaded")

        stats = compute_lstv_stats(all_results)
        reps  = pick_representatives(all_results, image_dir or Path('.'), n_reps)

        overview_content = (section_summary(stats)
                            + section_distribution(stats)
                            + section_morphometrics_lstv(stats))
        tabs.append(('lstv_overview', 'LSTV Overview', overview_content))

        if reps:
            tabs.append(('lstv_reps', 'Representative Cases',
                         section_representatives(reps, image_dir or Path('.'))))

        tabs.append(('lstv_table', 'All LSTV Cases',
                     section_full_table(all_results)))

    if not tabs:
        logger.error("No data found — check --lstv_json / --morphometrics_json paths")
        return

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    tab_bar = '<div class="tab-bar">'
    panels  = ''
    for i, (tid, tlabel, tcontent) in enumerate(tabs):
        active = 'active' if i == 0 else ''
        tab_bar += (f'<button class="tab-btn {active}" '
                    f'onclick="showTab(\'{tid}\')" id="tbtn-{tid}">'
                    f'{tlabel}</button>')
        panels += (f'<div class="tab-panel {active}" id="tab-{tid}">'
                   f'{tcontent}</div>')
    tab_bar += '</div>'

    tab_js = """
<script>
function showTab(id) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  document.getElementById('tbtn-' + id).classList.add('active');
}
</script>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LSTV Pipeline Report — {ts}</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>LSTV Pipeline Report</h1>
  <p class="subtitle">
    Hybrid Two-Phase Pipeline &nbsp;|&nbsp; {ts}
  </p>
  {tab_bar}
  {panels}
  <footer>LSTV Detection Pipeline — {ts}</footer>
  {tab_js}
</body>
</html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding='utf-8')
    size_mb = output_html.stat().st_size / 1e6
    logger.info(f"Report written: {output_html}  ({size_mb:.2f} MB)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LSTV Pipeline HTML Report — LSTV classification + morphometrics summary'
    )
    parser.add_argument('--lstv_json',          default=None,
        help='results/lstv_detection/lstv_results.json')
    parser.add_argument('--image_dir',          default=None,
        help='Directory containing *_lstv_overlay.png files')
    parser.add_argument('--morphometrics_json', default=None,
        help='results/morphometrics/morphometrics_all.json')
    parser.add_argument('--output_html',        required=True,
        help='Path for output HTML file')
    parser.add_argument('--n_reps',             type=int, default=3,
        help='Representative cases per Castellvi class (default: 3)')
    parser.add_argument('--morpho_only',        action='store_true',
        help='Only generate morphometrics summary (skip LSTV report)')
    args = parser.parse_args()

    if not args.lstv_json and not args.morphometrics_json:
        parser.error("At least one of --lstv_json or --morphometrics_json is required")

    build_report(
        results_path = Path(args.lstv_json)          if args.lstv_json          else None,
        morpho_path  = Path(args.morphometrics_json) if args.morphometrics_json else None,
        image_dir    = Path(args.image_dir)          if args.image_dir          else None,
        output_html  = Path(args.output_html),
        n_reps       = args.n_reps,
        morpho_only  = args.morpho_only,
    )


if __name__ == '__main__':
    main()
