#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=lstv_3d
#SBATCH -o logs/lstv_3d_%j.out
#SBATCH -e logs/lstv_3d_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration — edit these to change behaviour ────────────────────────────
STUDY_ID=""          # single study ID — leave empty to use ALL or RANK_BY mode
RANK_BY=morpho       # "morpho"          → rank by pathology burden score from JSON
                     # <csv_column>      → legacy: rank by uncertainty CSV column
TOP_N=5              # most-pathologic studies to render (morpho mode)
                     # OR top/bottom N by csv column (legacy mode)
TOP_NORMAL=1         # most-normal studies to render (morpho mode only)
ALL=false            # set to true to render every study with SPINEPS segmentation
SMOOTH=3             # Gaussian pre-smoothing sigma for marching cubes surfaces
NO_TSS=false         # set to true to skip TotalSpineSeg label rendering

# ── Pathology burden scoring (RANK_BY=morpho) ─────────────────────────────────
#
# Weighted score computed per study from morphometrics_all.json:
#   Canal absolute stenosis  +3  |  relative  +1
#   Cord severe MSCC         +4  |  moderate  +3  |  mild  +1
#   DHI severe (<50%)        +2/level  |  moderate (<70%)  +1/level
#   Spondylolisthesis        +2/level (≥3mm translation)
#   Vertebral wedge fracture +2/level (Ha/Hp <0.80), +3 if <0.75
#   Baastrup contact         +2  |  risk zone  +1
#   Facet tropism grade 2    +2  |  grade 1    +1
#   LFT hypertrophy severe   +2  |  hypertrophy  +1
#   Castellvi III/IV         +2  |  I/II         +1
#
# Studies are sorted descending → top TOP_N = most pathologic
# Studies are sorted ascending  → top TOP_NORMAL = most normal (score ≥ 0)
# See scripts/pathology_score.py for full implementation.
#
# ── Mask sources used in morphometrics ───────────────────────────────────────
#
# SPINEPS seg-spine_msk (ALL 14 labels rendered + used in morphometrics):
#   26  Sacrum             41  Arcus_Vertebrae       42  Spinosus_Process
#   43  Costal_Process_L   44  Costal_Process_R      45  Superior_Articular_L
#   46  Superior_Articular_R  47  Inferior_Articular_L  48  Inferior_Articular_R
#   49  Vertebra_Corpus_border  ← used for accurate Ha/Hm/Hp vertebral heights
#   60  Spinal_Cord        61  Spinal_Canal
#   62  Endplate (all merged) ← rendered + used as ep-ep DHI fallback (coral #ff6b6b)
#   100 Vertebra_Disc (all merged)
#
# VERIDAH seg-vert_msk (fully used):
#   1-25  per-vertebra labels (C1-L6)
#   100+X IVD below vertebra X      ← primary disc source for per-level DHI
#   200+X Endplate of vertebra X    ← primary source for ep-to-ep distance (pink #ff8888)
#   If 200+X absent, falls back to SPINEPS label 62 sliced to level Z-range
#
# TotalSpineSeg step2_output (ALL 50 labels rendered):
#   1=cord  2=canal  (preferred over SPINEPS for canal AP/DSCA — full spine coverage)
#   11-17=C1-C7    21-32=T1-T12    41-45=L1-L5    50=sacrum
#   63-100=all discs (used for per-level canal AP sampling at disc midpoints)
#
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "LSTV 3D VISUALIZER + COMPREHENSIVE MORPHOMETRICS"
echo "STUDY_ID=${STUDY_ID:-<batch>}  RANK_BY=$RANK_BY  ALL=$ALL"
echo "TOP_N=$TOP_N  TOP_NORMAL=$TOP_NORMAL  SMOOTH=$SMOOTH  NO_TSS=$NO_TSS"
echo ""
echo "Morphometrics: Vertebral QM · DHI · DSCA · Canal shape · LF ·"
echo "               Spondylolisthesis · Baastrup · Facet tropism ·"
echo "               Foraminal volume · Cord CSA/MSCC · Lee grade"
echo "Job: $SLURM_JOB_ID  |  Start: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || { echo "ERROR: singularity not found"; exit 1; }
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
UNCERTAINTY_CSV="${PROJECT_DIR}/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
MODELS_DIR="${PROJECT_DIR}/models"
LSTV_JSON="${PROJECT_DIR}/results/lstv_detection/lstv_results.json"
MORPHO_JSON="${PROJECT_DIR}/results/morphometrics/morphometrics_all.json"

mkdir -p logs results/lstv_3d

# --- Preflight ---
if [[ ! -d "${PROJECT_DIR}/results/spineps/segmentations" ]]; then
    echo "ERROR: SPINEPS segmentations not found. Run 02b_spineps_selective.sh first"
    exit 1
fi

if [[ ! -d "${PROJECT_DIR}/results/totalspineseg" ]]; then
    echo "ERROR: TotalSpineSeg results not found. Run 03b_totalspineseg_selective.sh first"
    exit 1
fi

if [[ "$RANK_BY" == "morpho" && ! -f "$MORPHO_JSON" ]]; then
    echo "ERROR: RANK_BY=morpho requires morphometrics_all.json at:"
    echo "  $MORPHO_JSON"
    echo "Run 05_morphometrics.sh first"
    exit 1
fi

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container image..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Build selection args ---
SELECTION_ARGS=()

if [[ -n "$STUDY_ID" ]]; then
    SELECTION_ARGS+=( "--study_id" "$STUDY_ID" )
    echo "Single-study mode: $STUDY_ID"

elif [[ "$ALL" == "true" ]]; then
    SELECTION_ARGS+=( "--all" )
    echo "ALL mode: rendering every study with SPINEPS segmentations"

elif [[ "$RANK_BY" == "morpho" ]]; then
    SELECTION_ARGS+=(
        "--rank_by"   "morpho"
        "--top_n"     "$TOP_N"
        "--top_normal" "$TOP_NORMAL"
    )
    echo "Morpho mode: top $TOP_N pathologic + $TOP_NORMAL normal (by pathology burden score)"
    echo "Pathology score source: $MORPHO_JSON"

else
    # Legacy uncertainty-CSV mode
    if [[ ! -f "$UNCERTAINTY_CSV" ]]; then
        echo "ERROR: Uncertainty CSV not found: $UNCERTAINTY_CSV"
        echo "Run 00_ian_pan_inference.sh first, or set RANK_BY=morpho"
        exit 1
    fi
    if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
        echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
        exit 1
    fi
    SELECTION_ARGS+=(
        "--rank_by"          "$RANK_BY"
        "--top_n"            "$TOP_N"
        "--uncertainty_csv"  "/work/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
        "--valid_ids"        "/app/models/valid_id.npy"
    )
    echo "Legacy mode: top/bottom $TOP_N by $RANK_BY"
fi

# --- Optional rendering flags ---
if [[ "$NO_TSS" == "true" ]]; then
    SELECTION_ARGS+=( "--no_tss" )
    echo "TotalSpineSeg label rendering disabled"
fi

# --- Morphometrics JSON ---
if [[ -f "$MORPHO_JSON" ]]; then
    SELECTION_ARGS+=( "--morphometrics_json" "/work/results/morphometrics/morphometrics_all.json" )
    echo "Pre-computed morphometrics found — loading from JSON"
else
    echo "WARNING: morphometrics_all.json not found — computations will be done inline"
fi

# --- Annotation from detection results ---
if [[ -f "$LSTV_JSON" ]]; then
    SELECTION_ARGS+=( "--lstv_json" "/work/results/lstv_detection/lstv_results.json" )
    echo "Detection results found — Castellvi annotations + burden scoring enabled"
else
    echo "WARNING: lstv_results.json not found — run 04_lstv_detection.sh first"
    echo "         3D measurement rulers will still be drawn from seg-spine_msk"
fi

# --- Bind models dir only when valid_ids is needed ---
BIND_ARGS="--bind ${PROJECT_DIR}:/work"
if [[ -d "$MODELS_DIR" ]]; then
    BIND_ARGS="${BIND_ARGS} --bind ${MODELS_DIR}:/app/models"
fi

# --- Run ---
singularity exec \
    $BIND_ARGS \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/06_visualize_3d.py \
        --spineps_dir    /work/results/spineps \
        --totalspine_dir /work/results/totalspineseg \
        --output_dir     /work/results/lstv_3d \
        --smooth         "$SMOOTH" \
        "${SELECTION_ARGS[@]}"

echo "================================================================"
echo "3D visualization complete"
echo "HTMLs → results/lstv_3d/"
echo ""
echo "Each HTML includes a pathology burden score badge and right-side"
echo "metrics panel with all flagged thresholds colour-coded."
echo "End: $(date)"
echo "================================================================"
