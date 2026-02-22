#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=spine_morpho
#SBATCH -o logs/spine_morpho_%j.out
#SBATCH -e logs/spine_morpho_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
TOP_N=1                    # studies from each end  (ignored if ALL=true)
RANK_BY=l5_s1_confidence   # uncertainty column to rank by
ALL=false                  # true → run entire dataset
STUDY_ID=""                # single study override (leave empty for batch)
NO_REPORTS=false           # true → skip per-study HTML reports (faster)
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "SPINE MORPHOMETRICS (standalone — no 3D rendering)"
echo "TOP_N=$TOP_N  RANK_BY=$RANK_BY  ALL=$ALL  STUDY_ID=${STUDY_ID:-<batch>}"
echo "Job: $SLURM_JOB_ID  |  Start: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
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

mkdir -p logs results/morphometrics results/morphometrics/reports

# --- Preflight ---
if [[ ! -d "${PROJECT_DIR}/results/spineps/segmentations" ]]; then
    echo "ERROR: SPINEPS segmentations not found. Run 02b_spineps_selective.sh first"
    exit 1
fi

# --- Container ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Build args ---
ARGS=()

if [[ -n "$STUDY_ID" ]]; then
    ARGS+=("--study_id" "$STUDY_ID")
    echo "Single-study mode: $STUDY_ID"

elif [[ "$ALL" == "true" ]]; then
    ARGS+=("--all")
    echo "ALL mode: processing every study with SPINEPS segmentations"

else
    if [[ ! -f "$UNCERTAINTY_CSV" ]]; then
        echo "ERROR: Uncertainty CSV not found: $UNCERTAINTY_CSV"
        echo "Run 00_ian_pan_inference.sh first, or set ALL=true"
        exit 1
    fi
    if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
        echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
        exit 1
    fi
    ARGS+=(
        "--uncertainty_csv" "/work/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
        "--valid_ids"       "/app/models/valid_id.npy"
        "--top_n"           "$TOP_N"
        "--rank_by"         "$RANK_BY"
    )
    echo "Selective mode: top/bottom $TOP_N by $RANK_BY"
fi

# Annotate reports with LSTV results if available
if [[ -f "$LSTV_JSON" ]]; then
    ARGS+=("--lstv_json" "/work/results/lstv_detection/lstv_results.json")
    echo "LSTV results found — reports will include Castellvi annotations"
fi

if [[ "$NO_REPORTS" == "true" ]]; then
    ARGS+=("--no_reports")
    echo "HTML report generation disabled"
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
    python3 -u /work/scripts/05_morphometrics.py \
        --spineps_dir    /work/results/spineps \
        --totalspine_dir /work/results/totalspineseg \
        --output_dir     /work/results/morphometrics \
        "${ARGS[@]}"

echo "================================================================"
echo "Morphometrics complete | End: $(date)"
echo ""
echo "Outputs:"
echo "  results/morphometrics/morphometrics_all.json   ← feed into 06_visualize_3d_v2.sh"
echo "  results/morphometrics/morphometrics_all.csv    ← statistics / R / Excel"
echo "  results/morphometrics/morphometrics_summary.json"
echo "  results/morphometrics/reports/*.html           ← per-study clinical reports"
echo "================================================================"
