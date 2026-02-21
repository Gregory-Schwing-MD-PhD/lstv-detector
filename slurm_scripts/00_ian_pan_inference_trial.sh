#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=12:00:00
#SBATCH --job-name=ian_pan_inference
#SBATCH -o logs/ian_pan_%j.out
#SBATCH -e logs/ian_pan_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

MODE=${MODE:-prod}
RETRY_FAILED=${RETRY_FAILED:-false}

echo "================================================================"
echo "IAN PAN EPISTEMIC UNCERTAINTY INFERENCE"
echo "Mode: $MODE"
echo "Job ID: $SLURM_JOB_ID | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
INPUT_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/epistemic_uncertainty"
MODEL_PATH="${PROJECT_DIR}/models/ian_pan_lstv.pth"

mkdir -p logs "$OUTPUT_DIR"

# --- Preflight ---
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Ian Pan model not found at $MODEL_PATH"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: DICOM input dir not found: $INPUT_DIR"
    exit 1
fi

N_STUDIES=$(ls -d "${INPUT_DIR}"/*/ 2>/dev/null | wc -l)
echo "Studies found in input dir: $N_STUDIES"

# --- Container ---
CONTAINER="docker://go2432/spineps-segmentation:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Args ---
RETRY_ARG=""
if [[ "$RETRY_FAILED" == "true" ]]; then
    RETRY_ARG="--retry_failed"
fi

# --- Run ---
singularity exec --nv \
    --bind "${PROJECT_DIR}":/work \
    --bind "${INPUT_DIR}":/data/input \
    --bind "${OUTPUT_DIR}":/data/output \
    --bind "$(dirname $SERIES_CSV)":/data/raw \
    --bind "${PROJECT_DIR}/models":/app/models \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python /work/scripts/inference_dicom.py \
        --input_dir   /data/input \
        --series_csv  /data/raw/train_series_descriptions.csv \
        --output_dir  /data/output \
        --model_path  /app/models/ian_pan_lstv.pth \
        --mode        "$MODE" \
        --valid_ids   /app/models/valid_id.npy \
        --n_mc_passes 20 \
        $RETRY_ARG

echo "================================================================"
echo "Inference complete | End: $(date)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next: set TOP_N and run sbatch slurm_scripts/02b_spineps_selective.sh"
echo "================================================================"
