"""
pathology_score.py  —  Morphometric Pathology Burden Scorer
============================================================
Computes a weighted pathology burden score from a morphometrics_all.json
record (output of 05_morphometrics.py).

Score is additive across independent pathology domains so that the
highest-scoring studies cover a range of pathology types rather than
being dominated by a single extreme metric.

Designed for study selection in 06_visualize_3d.py: the highest-scoring
studies are "most pathologic", the lowest are "most normal".

Domain weights
--------------
  Canal stenosis    absolute            +3
                    relative            +1
  Cord compression  severe MSCC         +4
                    moderate            +3
                    mild                +1
  DHI (per level)   severe (<50%)       +2 each
                    moderate (<70%)     +1 each
  Spondylolisthesis (per level, ≥3mm)   +2 each
  Vertebral fracture wedge <0.80        +2 each
                    intervention <0.75  +1 (additional)
  LSTV / Castellvi  type IIIa/b, IVa/b  +2
                    type Ia/b, IIa/b    +1
  Baastrup          contact             +2
                    risk zone           +1
  Facet tropism     grade 2 (≥10°)      +2
                    grade 1 (7–10°)     +1
  LFT hypertrophy   severe (>5mm)       +2
                    hypertrophy (>4mm)  +1

Maximum possible score: ~50 (theoretical). Practically expect 0–20.
"""

from __future__ import annotations

from typing import Optional


# ── Thresholds (duplicated from morphometrics_engine.T for standalone use) ────
_DHI_SEVERE   = 50.0
_DHI_MODERATE = 70.0
_SPONDYLO_MM  = 3.0
_WEDGE_FRAC   = 0.80
_WEDGE_INTER  = 0.75
_MSCC_MILD    = 0.50
_MSCC_MOD     = 0.67
_MSCC_SEV     = 0.80
_AP_NORM      = 12.0
_AP_ABS       = 7.0
_LFT_HYPER    = 4.0
_LFT_SEV      = 5.0
_TROP_WARN    = 7.0
_TROP_SEV     = 10.0
_BSTP_RISK    = 2.0
_BSTP_CONTACT = 0.0

_LUMBAR_PAIRS = [
    (20, 21, 'L1', 'L2'),
    (21, 22, 'L2', 'L3'),
    (22, 23, 'L3', 'L4'),
    (23, 24, 'L4', 'L5'),
    (24, 25, 'L5', 'S1'),  # also covers L5-L6 if present
]


def compute_pathology_score(m: dict) -> float:
    """
    Compute a scalar pathology burden score from one morphometrics JSON record.

    Parameters
    ----------
    m : dict
        One entry from morphometrics_all.json (output of 05_morphometrics.py).
        May be a flat dict (CSV-style) or the nested dataclass-dict format.

    Returns
    -------
    float  —  pathology burden score (higher = more pathologic)
    """
    if m.get('error'):
        return -1.0  # failed studies sort to the bottom

    score = 0.0

    # ── Canal stenosis ────────────────────────────────────────────────────────
    ap_cls = (m.get('canal_ap_class') or '').lower()
    if 'absolute' in ap_cls:
        score += 3
    elif 'relative' in ap_cls:
        score += 1

    # Also check per-level AP — pick the worst level
    for _, _, up, lo in _LUMBAR_PAIRS:
        lk = f'{up}_{lo}_level_ap_class'
        cls = (m.get(lk) or '').lower()
        if 'absolute' in cls:
            score += 2
            break
        elif 'relative' in cls:
            score += 1
            break

    # ── Cord compression ──────────────────────────────────────────────────────
    # Nested format (from dataclasses.asdict)
    cp = m.get('cord_compression_profile') or {}
    if isinstance(cp, dict):
        max_mscc = cp.get('max_mscc') or m.get('cord_max_mscc')
        cord_cls = (cp.get('classification') or '').lower()
    else:
        max_mscc = m.get('cord_max_mscc') or m.get('mscc_proxy')
        cord_cls = ''

    if max_mscc is not None:
        try:
            v = float(max_mscc)
            if v >= _MSCC_SEV:
                score += 4
            elif v >= _MSCC_MOD:
                score += 3
            elif v >= _MSCC_MILD:
                score += 1
        except (TypeError, ValueError):
            pass

    if 'severe' in cord_cls:
        score += 1   # bonus on top of MSCC points
    elif 'moderate' in cord_cls and max_mscc is not None and float(max_mscc or 0) < _MSCC_MOD:
        score += 1

    # ── DHI per level ─────────────────────────────────────────────────────────
    # Try both flat and nested (levels list) formats
    levels = m.get('levels') or []  # nested list of LevelMorphometrics dicts
    if levels:
        for lm in levels:
            if not isinstance(lm, dict): continue
            dhi = lm.get('dhi_pct')
            if dhi is None: continue
            try:
                v = float(dhi)
                if v < _DHI_SEVERE:
                    score += 2
                elif v < _DHI_MODERATE:
                    score += 1
            except (TypeError, ValueError):
                pass
    else:
        # flat CSV-style keys
        for _, _, up, lo in _LUMBAR_PAIRS:
            dhi = m.get(f'{up}_{lo}_dhi_pct')
            if dhi is None: continue
            try:
                v = float(dhi)
                if v < _DHI_SEVERE:
                    score += 2
                elif v < _DHI_MODERATE:
                    score += 1
            except (TypeError, ValueError):
                pass

    # ── Spondylolisthesis per level ───────────────────────────────────────────
    if levels:
        for lm in levels:
            if not isinstance(lm, dict): continue
            t = lm.get('sagittal_translation_mm')
            if t is None: continue
            try:
                if float(t) >= _SPONDYLO_MM:
                    score += 2
            except (TypeError, ValueError):
                pass
    else:
        for _, _, up, lo in _LUMBAR_PAIRS:
            t = m.get(f'{up}_{lo}_sagittal_translation_mm')
            if t is None: continue
            try:
                if float(t) >= _SPONDYLO_MM:
                    score += 2
            except (TypeError, ValueError):
                pass

    # ── Vertebral fracture (wedge ratio) ──────────────────────────────────────
    if levels:
        for lm in levels:
            if not isinstance(lm, dict): continue
            w = lm.get('wedge_ha_hp')
            if w is None: continue
            try:
                v = float(w)
                if v < _WEDGE_INTER:
                    score += 3   # intervention threshold — extra severe
                elif v < _WEDGE_FRAC:
                    score += 2
            except (TypeError, ValueError):
                pass
    else:
        for _, _, up, lo in _LUMBAR_PAIRS:
            w = m.get(f'{up}_{lo}_wedge_ha_hp')
            if w is None: continue
            try:
                v = float(w)
                if v < _WEDGE_INTER:
                    score += 3
                elif v < _WEDGE_FRAC:
                    score += 2
            except (TypeError, ValueError):
                pass

    # ── Baastrup ──────────────────────────────────────────────────────────────
    if m.get('baastrup_contact') is True or str(m.get('baastrup_contact','')).lower() == 'true':
        score += 2
    elif m.get('baastrup_risk') is True or str(m.get('baastrup_risk','')).lower() == 'true':
        score += 1

    # ── Facet tropism ─────────────────────────────────────────────────────────
    trop = m.get('facet_tropism_deg')
    if trop is not None:
        try:
            v = float(trop)
            if v >= _TROP_SEV:
                score += 2
            elif v >= _TROP_WARN:
                score += 1
        except (TypeError, ValueError):
            pass

    # ── Ligamentum flavum ─────────────────────────────────────────────────────
    lft = m.get('lft_proxy_mm')
    if lft is not None:
        try:
            v = float(lft)
            if v > _LFT_SEV:
                score += 2
            elif v > _LFT_HYPER:
                score += 1
        except (TypeError, ValueError):
            pass

    # ── LSTV / Castellvi (from LSTV JSON merged into morphometrics or directly) ─
    # morpho record won't normally have this unless merged upstream;
    # handled at call site in select_studies_by_morpho when lstv_by_id is available
    castellvi = str(m.get('castellvi_type') or '').upper()
    if any(t in castellvi for t in ('IIIA','IIIB','IVA','IVB','III','IV')):
        score += 2
    elif any(t in castellvi for t in ('IIA','IIB','IA','IB','II',)):
        score += 1

    return score


def select_studies_by_morpho(
    morpho_all: list[dict],
    n_pathologic: int,
    n_normal: int = 1,
    lstv_by_id: Optional[dict] = None,
) -> tuple[list[str], list[str]]:
    """
    Score all studies and return two lists:
      pathologic_ids  — top n_pathologic by score (descending)
      normal_ids      — bottom n_normal by score (ascending, score >= 0)

    Parameters
    ----------
    morpho_all      : list of morphometric dicts from morphometrics_all.json
    n_pathologic    : number of most-pathologic studies to return
    n_normal        : number of most-normal studies to return
    lstv_by_id      : optional dict of {study_id: lstv_result} to enrich scores

    Returns
    -------
    (pathologic_ids, normal_ids)  — may overlap if dataset is very small
    """
    scored = []
    for m in morpho_all:
        sid = str(m.get('study_id', ''))
        if not sid:
            continue
        # Optionally merge Castellvi type into the record for scoring
        if lstv_by_id and sid in lstv_by_id:
            m = {**m, 'castellvi_type': lstv_by_id[sid].get('castellvi_type')}
        s = compute_pathology_score(m)
        scored.append((sid, s))

    # Sort descending for pathologic
    scored_desc = sorted(scored, key=lambda x: x[1], reverse=True)
    # Sort ascending for normal (exclude error records with score == -1)
    scored_asc  = sorted(
        [(sid, s) for sid, s in scored if s >= 0],
        key=lambda x: x[1]
    )

    pathologic_ids = [sid for sid, _ in scored_desc[:n_pathologic]]
    normal_ids     = [sid for sid, _ in scored_asc[:n_normal]]

    return pathologic_ids, normal_ids, {sid: s for sid, s in scored}
