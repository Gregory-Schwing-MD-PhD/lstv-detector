#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=spineps_selective
#SBATCH -o logs/spineps_selective_%j.out
#SBATCH -e logs/spineps_selective_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# Set MODE to "all" to process every valid study, or "selective" for top/bottom N
MODE=all                   # "all" or "selective"
TOP_N=30                   # used only when MODE=selective
RANK_BY=l5_s1_confidence   # used only when MODE=selective
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "SPINEPS SEGMENTATION"
echo "Mode:         $MODE"
[[ "$MODE" == "selective" ]] && echo "Top/Bottom N: $TOP_N  |  Rank by: $RANK_BY"
echo "Job ID:       $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start:        $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
SPINEPS_PKG_MODELS="${PWD}/models/spineps_pkg_models"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR" \
         "$SPINEPS_PKG_MODELS"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
UNCERTAINTY_CSV="${PROJECT_DIR}/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv"
NIFTI_DIR="${PROJECT_DIR}/results/nifti"
SPINEPS_DIR="${PROJECT_DIR}/results/spineps"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
MODELS_DIR="${PROJECT_DIR}/models"

# Create ALL bind-mount targets before singularity exec (critical!)
mkdir -p logs \
         "$SPINEPS_DIR" \
         "$SPINEPS_DIR/segmentations" \
         "$SPINEPS_DIR/metadata"

# --- Preflight ---
if [[ "$MODE" == "selective" && ! -f "$UNCERTAINTY_CSV" ]]; then
    echo "ERROR: Uncertainty CSV not found: $UNCERTAINTY_CSV"
    echo "Run sbatch slurm_scripts/00_ian_pan_inference.sh first"
    exit 1
fi

if [[ ! -f "${MODELS_DIR}/valid_id.npy" ]]; then
    echo "ERROR: valid_id.npy not found at ${MODELS_DIR}/valid_id.npy"
    exit 1
fi

# --- Container ---
CONTAINER="docker://go2432/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Build python args ---
if [[ "$MODE" == "all" ]]; then
    PYTHON_ARGS=(
        --nifti_dir   /work/results/nifti
        --spineps_dir /work/results/spineps
        --series_csv  /work/data/raw/train_series_descriptions.csv
        --valid_ids   /app/models/valid_id.npy
        --all
    )
else
    PYTHON_ARGS=(
        --uncertainty_csv /work/results/epistemic_uncertainty/lstv_uncertainty_metrics.csv
        --nifti_dir       /work/results/nifti
        --spineps_dir     /work/results/spineps
        --series_csv      /work/data/raw/train_series_descriptions.csv
        --valid_ids       /app/models/valid_id.npy
        --top_n           "$TOP_N"
        --rank_by         "$RANK_BY"
    )
fi

# --- Run ---
singularity exec --nv \
    --bind "${PROJECT_DIR}":/work \
    --bind "${NIFTI_DIR}":/work/results/nifti \
    --bind "${SPINEPS_DIR}":/work/results/spineps \
    --bind "${MODELS_DIR}":/app/models \
    --bind "${SPINEPS_PKG_MODELS}":/opt/conda/lib/python3.10/site-packages/spineps/models \
    --env SPINEPS_SEGMENTOR_MODELS=/app/models \
    --env SPINEPS_ENVIRONMENT_DIR=/app/models \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/02b_spineps_selective.py "${PYTHON_ARGS[@]}"

echo "================================================================"
echo "SPINEPS complete | End: $(date)"
echo "Already-done studies are skipped automatically."
echo "================================================================"
