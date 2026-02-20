#!/usr/bin/env python3
"""
06_html_report.py — Top-30 LSTV Cases HTML Report
===================================================
Reads lstv_results.json and the overlay PNG directory, ranks studies by
Castellvi severity and confidence, and produces a self-contained HTML
report with embedded images showing the top 30 LSTV cases.

Usage
-----
  python 06_html_report.py \
      --lstv_json   results/lstv_detection/lstv_results.json \
      --image_dir   results/lstv_visualization \
      --output_html results/lstv_report.html \
      [--top_n 30]
"""

import argparse
import base64
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# RANKING
# ============================================================================

CASTELLVI_RANK = {'Type IV': 4, 'Type III': 3, 'Type II': 2, 'Type I': 1}
CONFIDENCE_RANK = {'high': 3, 'moderate': 2, 'low': 1}


def sort_key(result: dict):
    ct   = result.get('castellvi_type') or ''
    conf = result.get('confidence') or 'low'
    dist = min(
        result.get('left',  {}).get('dist_mm', float('inf')),
        result.get('right', {}).get('dist_mm', float('inf')),
    )
    dist_score = -dist if not (dist == float('inf')) else -999
    return (
        CASTELLVI_RANK.get(ct, 0),
        CONFIDENCE_RANK.get(conf, 0),
        dist_score,
    )


# ============================================================================
# IMAGE EMBEDDING
# ============================================================================

def img_to_b64(path: Path) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ============================================================================
# HTML GENERATION
# ============================================================================

CASTELLVI_BADGE_COLORS = {
    'Type I':   '#3a86ff',
    'Type II':  '#ff9f1c',
    'Type III': '#e63946',
    'Type IV':  '#9d0208',
}
CONFIDENCE_BADGE_COLORS = {
    'high':     '#2dc653',
    'moderate': '#f4a261',
    'low':      '#e76f51',
}


def _d(v, unit='mm'):
    if v is None or (isinstance(v, float) and v == float('inf')):
        return 'N/A'
    return f'{v:.1f} {unit}'


def render_card(rank: int, result: dict, img_b64: str) -> str:
    sid  = result.get('study_id', '?')
    ct   = result.get('castellvi_type') or 'N/A'
    conf = result.get('confidence') or 'N/A'
    detected = result.get('lstv_detected', False)

    ct_color   = CASTELLVI_BADGE_COLORS.get(ct, '#6c757d')
    conf_color = CONFIDENCE_BADGE_COLORS.get(conf, '#6c757d')

    details = result.get('details', {})
    tv_name = details.get('tv_name', 'L5')
    has_l6  = details.get('has_l6', False)

    def side_row(side_name, sd):
        if not sd:
            return ''
        cls   = sd.get('classification', 'N/A')
        h     = _d(sd.get('tp_height_mm', 0))
        d     = _d(sd.get('dist_mm', float('inf')))
        p3    = sd.get('p_type_iii')
        p3_str = f'{p3:.2f}' if p3 is not None else '—'
        note  = sd.get('note', '')
        return f"""
        <tr>
          <td>{side_name}</td>
          <td><b>{cls}</b></td>
          <td>{h}</td>
          <td>{d}</td>
          <td>{p3_str}</td>
          <td style="color:#aaa;font-size:0.8em">{note}</td>
        </tr>"""

    left_row  = side_row('Left',  result.get('left',  {}))
    right_row = side_row('Right', result.get('right', {}))

    errors = result.get('errors', [])
    err_html = ''
    if errors:
        err_html = '<div class="errors">⚠ ' + '<br>'.join(errors) + '</div>'

    l6_badge = '<span class="badge" style="background:#9d0208">L6</span> ' if has_l6 else ''

    return f"""
  <div class="card">
    <div class="card-header">
      <span class="rank">#{rank}</span>
      <span class="study-id">{sid}</span>
      {l6_badge}
      <span class="badge" style="background:{ct_color}">{ct}</span>
      <span class="badge" style="background:{conf_color}">{conf} confidence</span>
      <span class="tv-label">TV: {tv_name}</span>
    </div>

    <div class="card-body">
      <div class="image-wrap">
        <img src="data:image/png;base64,{img_b64}" alt="LSTV overlay {sid}" loading="lazy">
      </div>

      <div class="metrics">
        <table>
          <thead>
            <tr>
              <th>Side</th><th>Class</th><th>Height</th>
              <th>TP–Sacrum</th><th>P(Type III)</th><th>Note</th>
            </tr>
          </thead>
          <tbody>
            {left_row}
            {right_row}
          </tbody>
        </table>
        {err_html}
      </div>
    </div>
  </div>"""


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LSTV Detection Report — Top {top_n} Cases</title>
  <style>
    :root {{
      --bg:       #0d0d1a;
      --surface:  #161628;
      --border:   #2a2a4a;
      --text:     #e8e8f0;
      --muted:    #8888aa;
      --accent:   #3a86ff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', system-ui, sans-serif;
      font-size: 14px;
      padding: 24px;
    }}
    h1 {{ font-size: 1.8rem; color: var(--accent); margin-bottom: 4px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 8px; }}
    .stats-bar {{
      display: flex; gap: 24px; flex-wrap: wrap;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px; margin-bottom: 28px;
    }}
    .stat {{ display: flex; flex-direction: column; }}
    .stat .val {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
    .stat .lbl {{ font-size: 0.75rem; color: var(--muted); }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 32px;
      overflow: hidden;
    }}
    .card-header {{
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      padding: 12px 16px;
      background: #1a1a30;
      border-bottom: 1px solid var(--border);
    }}
    .rank {{ font-size: 1.1rem; font-weight: 700; color: var(--muted); min-width: 36px; }}
    .study-id {{ font-weight: 700; font-size: 1rem; }}
    .badge {{
      display: inline-block; padding: 3px 10px; border-radius: 12px;
      font-size: 0.78rem; font-weight: 600; color: #fff;
    }}
    .tv-label {{ margin-left: auto; color: var(--muted); font-size: 0.85rem; }}
    .card-body {{ display: flex; flex-direction: column; }}
    .image-wrap {{ width: 100%; overflow: hidden; background: #000; }}
    .image-wrap img {{
      width: 100%; height: auto; display: block;
      max-height: 700px; object-fit: contain;
    }}
    .metrics {{ padding: 14px 16px; }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 0.88rem;
      margin-bottom: 8px;
    }}
    th {{ text-align: left; color: var(--muted); padding: 4px 8px;
          border-bottom: 1px solid var(--border); font-weight: 600; }}
    td {{ padding: 5px 8px; border-bottom: 1px solid #1e1e36; }}
    tr:last-child td {{ border-bottom: none; }}
    .errors {{
      background: #2d1010; border: 1px solid #5a1010;
      border-radius: 4px; padding: 8px; color: #ff8080;
      font-size: 0.82rem; margin-top: 6px;
    }}
    footer {{
      text-align: center; color: var(--muted); font-size: 0.8rem;
      margin-top: 40px; padding-top: 16px;
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>
"""

HTML_FOOT = """
  <footer>
    LSTV Detection Pipeline &nbsp;|&nbsp; SPINEPS + VERIDAH &nbsp;|&nbsp;
    Generated {timestamp}
  </footer>
</body>
</html>"""


def build_report(results_path: Path,
                 image_dir: Path,
                 output_html: Path,
                 top_n: int = 30):

    with open(results_path) as f:
        all_results = json.load(f)

    lstv_results = [r for r in all_results if r.get('lstv_detected')]
    lstv_results.sort(key=sort_key, reverse=True)
    top_cases    = lstv_results[:top_n]

    total        = len(all_results)
    lstv_count   = len(lstv_results)
    castellvi_counts = {}
    for r in lstv_results:
        ct = r.get('castellvi_type') or 'Unknown'
        castellvi_counts[ct] = castellvi_counts.get(ct, 0) + 1

    logger.info(f"Total studies: {total}  LSTV: {lstv_count}  "
                f"Rendering top {len(top_cases)}")

    stats_html = f"""
  <div class="stats-bar">
    <div class="stat"><span class="val">{total}</span><span class="lbl">Total Studies</span></div>
    <div class="stat"><span class="val">{lstv_count}</span><span class="lbl">LSTV Detected</span></div>
    <div class="stat"><span class="val">{100*lstv_count/max(total,1):.1f}%</span><span class="lbl">LSTV Rate</span></div>
    {''.join(f'<div class="stat"><span class="val">{n}</span><span class="lbl">{ct}</span></div>'
             for ct, n in sorted(castellvi_counts.items()))}
  </div>"""

    cards_html = []
    missing    = 0
    for rank, result in enumerate(top_cases, 1):
        sid      = result.get('study_id', '')
        img_path = image_dir / f"{sid}_lstv_overlay.png"
        if not img_path.exists():
            logger.warning(f"  Image not found: {img_path.name}")
            missing += 1
            continue
        try:
            img_b64 = img_to_b64(img_path)
            cards_html.append(render_card(rank, result, img_b64))
            logger.info(f"  [{rank}/{len(top_cases)}] {sid} embedded")
        except Exception as e:
            logger.error(f"  [{sid}] Image embed failed: {e}")
            missing += 1

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = (
        HTML_HEAD.format(top_n=top_n)
        + f'\n  <h1>LSTV Detection Report — Top {len(cards_html)} Cases</h1>\n'
        + f'  <p class="subtitle">SPINEPS + VERIDAH pipeline &nbsp;|&nbsp; '
          f'Castellvi classification &nbsp;|&nbsp; {timestamp}</p>\n'
        + stats_html
        + '\n'.join(cards_html)
        + HTML_FOOT.format(timestamp=timestamp)
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding='utf-8')
    size_mb = output_html.stat().st_size / 1e6
    logger.info(f"Report written: {output_html}  ({size_mb:.1f} MB)")
    if missing:
        logger.warning(f"  {missing} images were missing and skipped")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='LSTV HTML Report Generator')
    parser.add_argument('--lstv_json',   required=True)
    parser.add_argument('--image_dir',   required=True)
    parser.add_argument('--output_html', required=True)
    parser.add_argument('--top_n',       type=int, default=30)
    args = parser.parse_args()

    build_report(
        results_path = Path(args.lstv_json),
        image_dir    = Path(args.image_dir),
        output_html  = Path(args.output_html),
        top_n        = args.top_n,
    )


if __name__ == '__main__':
    main()
