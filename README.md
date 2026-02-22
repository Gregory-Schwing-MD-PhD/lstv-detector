# LSTV Detection + Spine Morphometrics Pipeline

Automated detection and Castellvi classification of lumbosacral transitional vertebrae (LSTV),
combined with a comprehensive standalone spine morphometrics engine for dataset-level analysis.

---

## Architecture Overview

```
Input: DICOM studies
│
├─ Step 01: DICOM → NIfTI conversion
│
├─ Step 02b: SPINEPS  (selective — sagittal T2)
│   └─ seg-spine_msk (TPs, arcus, spinous, canal, cord, sacrum…)
│   └─ seg-vert_msk  (VERIDAH instance labels: L1-L6, IVDs, endplates)
│
├─ Step 03b: TotalSpineSeg  (selective — sagittal + axial T2)
│   └─ sagittal_labeled.nii.gz  (cord, canal, vertebrae, discs, sacrum)
│   └─ axial_labeled.nii.gz
│
├─ Step 03c: Registration  (SPINEPS → axial T2w space)
│
├─ Step 04: LSTV Detection  (Hybrid Two-Phase Castellvi Classifier)
│   └─ results/lstv_detection/lstv_results.json
│
├─ Step 05: Morphometrics  ← NEW: standalone, no 3D rendering
│   └─ results/morphometrics/morphometrics_all.json   ← feeds step 06
│   └─ results/morphometrics/morphometrics_all.csv    ← statistics
│   └─ results/morphometrics/morphometrics_summary.json
│   └─ results/morphometrics/reports/{study_id}_report.html
│
└─ Step 06: 3D Visualisation  (reads JSON from step 05, no recalculation)
    └─ results/lstv_3d/{study_id}_3d_spine.html
```

### Key architectural principle
**Steps 05 and 06 are fully decoupled.**
Run 05 once on the whole dataset (compute-heavy, CPU-only, parallelisable).
Run 06 on whichever studies you want to inspect visually (fast, reads JSON).

---

## File Structure

```
lstv-detector/
├── scripts/
│   ├── morphometrics_engine.py   ← NEW: all morphometric calculations (importable)
│   ├── 01_dicom_to_nifti.py
│   ├── 02b_spineps_selective.py
│   ├── 03b_totalspineseg_selective.py
│   ├── 03c_register.py
│   ├── 04_detect_lstv.py         ← LSTV / Castellvi classifier
│   ├── 05_morphometrics.py       ← NEW: standalone batch morphometrics
│   └── 06_visualize_3d_v2.py    ← NEW: 3D viewer reading pre-computed JSON
│
├── slurm_scripts/
│   ├── 01_dicom_to_nifti.sh
│   ├── 02b_spineps_selective.sh
│   ├── 03b_totalspineseg_selective.sh
│   ├── 03c_register.sh
│   ├── 04_lstv_detection.sh
│   ├── 05_morphometrics.sh       ← NEW
│   └── 06_visualize_3d.sh        ← now wraps 06_visualize_3d_v2.py
│
└── results/
    ├── spineps/segmentations/
    ├── totalspineseg/
    ├── registered/
    ├── lstv_detection/
    ├── morphometrics/             ← NEW
    │   ├── morphometrics_all.json
    │   ├── morphometrics_all.csv
    │   ├── morphometrics_summary.json
    │   └── reports/
    └── lstv_3d/
```

---

## Quick Start

### Standard pipeline (sequential)

```bash
# Convert DICOM
sbatch slurm_scripts/01_dicom_to_nifti.sh

# Segment (can run in parallel)
sbatch slurm_scripts/02b_spineps_selective.sh
sbatch slurm_scripts/03b_totalspineseg_selective.sh

# Register SPINEPS → axial space
sbatch slurm_scripts/03c_register.sh

# Detect LSTV
sbatch slurm_scripts/04_lstv_detection.sh

# Compute all morphometrics (whole dataset, no 3D)
sbatch slurm_scripts/05_morphometrics.sh

# Visualise selected studies in 3D (reads pre-computed JSON)
sbatch slurm_scripts/06_visualize_3d.sh
```

### Dependency chain (fully automated)

```bash
J1=$(sbatch --parsable slurm_scripts/01_dicom_to_nifti.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm_scripts/02b_spineps_selective.sh)
J3=$(sbatch --parsable --dependency=afterok:$J1 slurm_scripts/03b_totalspineseg_selective.sh)
J4=$(sbatch --parsable --dependency=afterok:$J2:$J3 slurm_scripts/03c_register.sh)
J5=$(sbatch --parsable --dependency=afterok:$J4 slurm_scripts/04_lstv_detection.sh)
J6=$(sbatch --parsable --dependency=afterok:$J5 slurm_scripts/05_morphometrics.sh)
sbatch --dependency=afterok:$J6 slurm_scripts/06_visualize_3d.sh
```

---

## Morphometrics Engine (`morphometrics_engine.py`)

All measurements live in this single importable module. No rendering dependencies.

### Public API

```python
from morphometrics_engine import (
    load_study_masks, run_all_morphometrics,
    classify_stenosis, cord_compression_profile, T
)

masks  = load_study_masks("1234567", spineps_dir, totalspine_dir)
result = run_all_morphometrics(masks)

# Flat dict for CSV / pandas
row = result.to_dict()

# Access a specific field
print(result.canal_ap_mm)
print(result.cord_compression_profile)   # full per-slice profile

# Standalone canal classification
ap_class, dsca_class = classify_stenosis(ap_mm=11.5, dsca_mm2=95.0)
```

### Morphometrics computed

| Category | Parameters |
|---|---|
| **Central canal** | Global AP, DSCA, stenosis class (Normal/Relative/Absolute/Critical) |
| **Per-level canal** | AP + DSCA at each disc midpoint (L1-L2 → L5-S1), canal shape ref |
| **Cord compression** | **Full-length per-slice MSCC profile** — cord AP / canal AP at every Z slice, flagged zones, worst-case classification |
| **Disc (DHI)** | Farfan DHI, Method 2, endplate-to-endplate distance, source tracing |
| **Vertebral body** | Ha/Hm/Hp heights, Wedge/Compression/Crush ratios, Genant grade |
| **Spondylolisthesis** | Sagittal anterior translation per level (mm), ≥3mm flagged |
| **Ligamentum flavum** | LFT proxy (arcus→canal), classification |
| **Baastrup** | All inter-spinous gaps, contact/risk flags |
| **Facet tropism** | L/R articular process PCA orientation, Ko grade (0/1/2) |
| **Foraminal volume** | Elliptical cylinder proxy per level L/R, Lee grade equivalent |

### Cord compression profile (new)

The engine now profiles cord compression **along the full superior-inferior extent** of the cord
rather than at a single mid-point:

```python
profile = result.cord_compression_profile
# {
#   'max_mscc': 0.71,          # worst ratio (cord_AP / canal_AP)
#   'max_mscc_z_mm': 145.0,    # Z position of worst compression
#   'classification': 'Moderate',
#   'flagged_z_mm': [143.0, 145.0, 148.0],   # slices with MSCC ≥ 0.67
#   'flagged_count': 3,
#   'slices': [{'z_mm':..., 'cord_ap':..., 'canal_ap':..., 'mscc':..., 'cls':...}, ...]
# }
```

MSCC thresholds (cord AP / canal AP):
- **< 0.50** → Normal
- **0.50–0.67** → Mild
- **0.67–0.80** → Moderate (flagged)
- **≥ 0.80** → Severe (flagged)

Flagged slices are rendered as coloured spheres on the cord mesh in the 3D viewer.

---

## Label Reference

### SPINEPS `seg-spine_msk` (subregion / semantic)
| Label | Structure | Use in pipeline |
|---|---|---|
| 26 | Sacrum | Phase 1 sacrum fallback |
| 41 | Arcus Vertebrae | LF proxy, facet context |
| 42 | Spinous Process | Baastrup detection |
| **43** | **Costal Process Left** ← TP source | Phase 1 & 2 LSTV |
| **44** | **Costal Process Right** ← TP source | Phase 1 & 2 LSTV |
| 45/46 | Superior Articular L/R | Facet tropism |
| 47/48 | Inferior Articular L/R | Foraminal volume |
| 49 | Corpus Border | Precise vertebral height Ha/Hm/Hp |
| 60 | Spinal Cord | MSCC proxy, cord CSA |
| 61 | Spinal Canal | Canal AP/DSCA |
| 62 | Endplate (all merged) | DHI fallback |
| 100 | IVD (all merged) | DHI fallback |

### SPINEPS `seg-vert_msk` (VERIDAH instance)
| Label | Structure |
|---|---|
| 20–25 | L1–L6 vertebrae |
| 26 | Sacrum |
| 100+X | IVD below vertebra X (e.g. 124 = IVD below L5=24) |
| 200+X | Endplate of vertebra X (preferred DHI source) |

### TotalSpineSeg `step2_output` (sagittal)
| Labels | Structure |
|---|---|
| 1 | Spinal cord (preferred cord source) |
| 2 | Spinal canal (preferred canal source) |
| 11–17 | C1–C7 |
| 21–32 | T1–T12 |
| 41–45 | L1–L5 vertebrae (**≠ SPINEPS 41–44**) |
| **50** | **Sacrum** (preferred sacrum source) |
| 91–100 | Discs T12-L1 through L5-S1 |

> ⚠ **TSS labels 43/44 = L3/L4 vertebral bodies, NOT transverse processes.**
> TP source is always `seg-spine_msk` labels 43/44 (costal processes).

---

## HTML Clinical Reports (`05_morphometrics.py`)

Per-study reports include:
- **LSTV / Castellvi** banner (if `--lstv_json` provided)
- **Central canal stenosis** table with colour-coded rows and reference ranges
- **Per-level canal AP** profile
- **Full-length cord compression** chart (mini bar-chart visualisation per Z-slice)
- **Disc Height Index** per level with source tracing
- **Vertebral body height ratios** with Genant grading
- **Spondylolisthesis** per level
- **Ligamentum flavum** proxy
- **Baastrup disease** gaps
- **Facet tropism** Ko grading
- **Neural foraminal volume** Lee grade equivalents
- **Reference card** — all clinical thresholds in one table

Abnormal values are rendered in red; borderline values in amber.

---

## 3D Visualiser v2 (`06_visualize_3d_v2.py`)

Changes from v1:
- **Reads morphometrics JSON** — no recalculation, much faster per study
- **Sidebar annotation column** — all floating text is offset to the right of the volume
  (X + 55 mm). Spondylolisthesis, facet tropism, stenosis, and DHI annotations no longer
  overlap the mesh.
- **Cord compression spheres** — coloured markers (amber/orange/red) at each flagged
  Z-slice directly on the cord surface.
- **Sidebar camera preset** — new "Sidebar" view button angles the camera to show the
  annotation column alongside the 3D mesh.
- Falls back to inline computation if no JSON is available.

---

## Clinical Thresholds Reference

All thresholds live in `morphometrics_engine.T` for a single source of truth.

| Parameter | Normal | Flag |
|---|---|---|
| Canal AP | > 12 mm | < 7 mm (Absolute) |
| Canal DSCA | > 100 mm² | < 70 mm² (Absolute) |
| MSCC (cord/canal) | < 0.50 | ≥ 0.67 (Moderate) |
| DHI (Farfan) | > 85% | < 50% (Severe) |
| Vertebral Wedge ratio | ≥ 0.80 | < 0.75 (intervention) |
| Spondylolisthesis | < 3 mm | ≥ 3 mm |
| LFT | ≤ 3.5 mm | > 5 mm (severe) |
| Baastrup gap | > 2 mm | ≤ 0 mm (contact) |
| Facet tropism | ≤ 7° | ≥ 10° (Grade 2) |
| TP height | < 19 mm | ≥ 19 mm → Type I LSTV |

---

## Troubleshooting

### Morphometrics all N/A for a study
- Confirm `seg-spine_msk.nii.gz` and `seg-vert_msk.nii.gz` are present in
  `results/spineps/segmentations/{study_id}/`
- Check that SPINEPS labels are in the expected range (`sp_labels` logged at INFO)

### Cord profile absent
- Requires both cord (TSS label 1 or SPINEPS 60) and canal (TSS label 2 or SPINEPS 61)
- Check TSS sagittal output: `results/totalspineseg/{study_id}/sagittal/`

### 3D viewer very slow
- Run `05_morphometrics.py` first and pass `--morphometrics_json` to `06_visualize_3d_v2.py`
- Set `--smooth 1.0` for faster (less smooth) meshing

### DHI all from SPINEPS-merged (not VERIDAH)
- VERIDAH IVD labels are `100+X` (e.g. 124 for L5). Check `vert_labels` in logs.
- If absent, TSS disc labels (92–100) are used as secondary source.

---

## Citation

If you use this pipeline, please cite:
- **SPINEPS**: Märk et al. (2024)
- **TotalSpineSeg**: Molinier et al. (2024)
- **Castellvi classification**: Castellvi et al. (1984) Spine 9(5):493-495
- **DHI (Farfan method)**: Farfan et al. (1972)
- **Ko facet tropism**: Ko et al. (1994)
- **Lee foraminal grading**: Lee et al. (1988)

---

## Contact

For pipeline issues: go2432@wayne.edu  
Wayne State University School of Medicine — Spine Imaging & AI Lab
