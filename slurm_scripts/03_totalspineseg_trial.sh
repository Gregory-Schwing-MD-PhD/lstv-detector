#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=36:00:00
#SBATCH --job-name=totalspineseg
#SBATCH -o logs/totalspineseg_%j.out
#SBATCH -e logs/totalspineseg_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

MODE=${MODE:-trial}
RETRY_FAILED=${RETRY_FAILED:-false}

echo "================================================================"
echo "TOTALSPINESEG SEGMENTATION (Full Pipeline)"
echo "Mode: $MODE"
if [[ "$RETRY_FAILED" == "true" ]]; then
    echo "Retry Failed: YES"
fi
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

SCRATCH_DIR="/wsu/tmp/${USER}/totalspineseg_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
export SINGULARITY_TMPDIR="$SCRATCH_DIR"

mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

echo "Scratch dir: $SCRATCH_DIR"
df -h "$SCRATCH_DIR"

export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/nifti"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/totalspineseg"
MODELS_DIR="${PROJECT_DIR}/models/totalspineseg_models"

# Writable host copy of the nnUNetTrainer directory.
# auglab.add_trainer() copies nnUNetTrainerDAExt.py into the nnunetv2 trainer
# directory at runtime. Singularity containers are read-only so this fails.
# Same pattern as spineps binding its models dir: extract the full directory
# from the container to a writable host path once, then bind it back in.
NNUNET_TRAINER_DIR="${PROJECT_DIR}/models/nnunetv2_trainer"

mkdir -p logs "$OUTPUT_DIR" "$MODELS_DIR" "$NNUNET_TRAINER_DIR"

# --- Container ---
CONTAINER="docker://go2432/totalspineseg:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/totalspineseg.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling TotalSpineSeg container (first time only)..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# Extract nnUNetTrainer directory from container if not already done.
# Only runs once — subsequent jobs reuse the extracted directory.
if [[ -z "$(ls -A $NNUNET_TRAINER_DIR 2>/dev/null)" ]]; then
    echo "Extracting nnUNetTrainer directory from container (one-time setup)..."
    singularity exec "$IMG_PATH" \
        cp -r /opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/. \
        "$NNUNET_TRAINER_DIR/"
    echo "Extracted $(ls $NNUNET_TRAINER_DIR | wc -l) files"
fi

# --- Run ---
RETRY_ARG=""
if [[ "$RETRY_FAILED" == "true" ]]; then
    RETRY_ARG="--retry-failed"
fi

singularity exec --nv \
    --bind "${PROJECT_DIR}":/work \
    --bind "${NIFTI_DIR}":/work/results/nifti \
    --bind "${OUTPUT_DIR}":/work/results/totalspineseg \
    --bind "${MODELS_DIR}":/app/models \
    --bind "${NNUNET_TRAINER_DIR}":/opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer \
    --env TOTALSPINESEG_DATA=/app/models \
    --env PYTHONUNBUFFERED=1 \
    --pwd /work \
    "$IMG_PATH" \
    python3 -u /work/scripts/03_run_totalspineseg.py \
        --nifti_dir  /work/results/nifti \
        --series_csv /work/data/raw/train_series_descriptions.csv \
        --output_dir /work/results/totalspineseg \
        --mode       "$MODE" \
        $RETRY_ARG

echo "================================================================"
echo "TotalSpineSeg complete | End: $(date)"
echo "================================================================"

# Cleanup scratch space
rm -rf "$SCRATCH_DIR"
echo "Scratch cleanup complete."
