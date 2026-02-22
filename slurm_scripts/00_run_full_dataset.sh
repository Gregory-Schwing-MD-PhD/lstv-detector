#!/bin/bash
#SBATCH -q primary 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=168:00:00
#SBATCH --job-name=lstv_full_dataset
#SBATCH -o logs/lstv_full_dataset_%j.out
#SBATCH -e logs/lstv_full_dataset_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# LSTV FULL-DATASET PIPELINE — processes every study in valid_id.npy
#
# Step 0: Ian-Pan uncertainty inference   (GPU)
# Step 1: SPINEPS segmentation — ALL      (GPU, long)
# Step 2: TotalSpineSeg — ALL             (GPU, long)
# Step 3: Morphometrics — ALL             (CPU, parallel)
# Step 4: Dataset summary HTML report     (CPU, fast)
#
# Skips studies that are already done at each stage (safe to re-run).
# ══════════════════════════════════════════════════════════════════════════════

echo "================================================================"
echo "LSTV FULL-DATASET PIPELINE"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
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

PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

# ─── Preflight checks ─────────────────────────────────────────────────────────
if [[ ! -d "data/raw/train_images" ]] || \
   [[ $(ls -1 data/raw/train_images 2>/dev/null | wc -l) -eq 0 ]]; then
    echo "ERROR: data/raw/train_images is empty or missing."
    echo "  Download data first with: sbatch slurm_scripts/00_download_data.sh"
    exit 1
fi

if [[ ! -f "models/valid_id.npy" ]]; then
    echo "ERROR: models/valid_id.npy not found."
    exit 1
fi

if [[ ! -f "models/point_net_checkpoint.pth" ]]; then
    echo "ERROR: models/point_net_checkpoint.pth not found."
    exit 1
fi

N_VALID=$(python3 -c "import numpy as np; a=np.load('models/valid_id.npy'); print(len(a))")
echo "Valid study IDs: $N_VALID"
echo ""
echo "Pipeline will submit:"
echo "  Step 0: Ian-Pan inference (GPU, ~1-2h)"
echo "  Step 1: SPINEPS — ALL $N_VALID studies (GPU, long)"
echo "  Step 2: TotalSpineSeg — ALL $N_VALID studies (GPU, very long)"
echo "  Step 3: Morphometrics — ALL $N_VALID studies (CPU)"
echo "  Step 4: Dataset summary HTML report (CPU, fast)"
echo ""

# ─── Step 0: Ian-Pan inference ────────────────────────────────────────────────
echo "Submitting: Step 0 — Ian-Pan inference"
JOB0=$(sbatch --parsable slurm_scripts/00_ian_pan_inference.sh)
echo "  Job ID: $JOB0"

# ─── Step 1: SPINEPS — ALL ────────────────────────────────────────────────────
# We submit a modified call that passes --all to 02b_spineps_selective.py.
# This job does NOT depend on Ian-Pan (inference not needed for segmentation).
echo ""
echo "Submitting: Step 1 — SPINEPS (ALL valid studies)"
JOB1=$(sbatch --parsable \
    --job-name=spineps_all \
    --time=96:00:00 \
    --dependency=afterok:$JOB0 \
    --wrap="
set -euo pipefail
export CONDA_PREFIX=\"\${HOME}/mambaforge/envs/nextflow\"
export PATH=\"\${CONDA_PREFIX}/bin:\$PATH\"
export XDG_RUNTIME_DIR=\"\${HOME}/xdr\"
export NXF_SINGULARITY_CACHEDIR=\"\${HOME}/singularity_cache\"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

PROJECT_DIR=\"$(pwd)\"
SPINEPS_PKG_MODELS=\"\${PROJECT_DIR}/models/spineps_pkg_models\"
mkdir -p \"\$SPINEPS_PKG_MODELS\" logs

IMG_PATH=\"\${NXF_SINGULARITY_CACHEDIR}/spineps-segmentation.sif\"
if [[ ! -f \"\$IMG_PATH\" ]]; then
    singularity pull \"\$IMG_PATH\" docker://go2432/spineps-segmentation:latest
fi

singularity exec --nv \\
    --bind \"\${PROJECT_DIR}\":/work \\
    --bind \"\${PROJECT_DIR}/results/nifti\":/work/results/nifti \\
    --bind \"\${PROJECT_DIR}/results/spineps\":/work/results/spineps \\
    --bind \"\${PROJECT_DIR}/models\":/app/models \\
    --bind \"\${SPINEPS_PKG_MODELS}\":/opt/conda/lib/python3.10/site-packages/spineps/models \\
    --env SPINEPS_SEGMENTOR_MODELS=/app/models \\
    --env SPINEPS_ENVIRONMENT_DIR=/app/models \\
    --env PYTHONUNBUFFERED=1 \\
    --pwd /work \\
    \"\$IMG_PATH\" \\
    python /work/scripts/02b_spineps_selective.py \\
        --nifti_dir    /work/results/nifti \\
        --spineps_dir  /work/results/spineps \\
        --series_csv   /work/data/raw/train_series_descriptions.csv \\
        --valid_ids    /app/models/valid_id.npy \\
        --all
" \
    --output="logs/spineps_all_%j.out" \
    --error="logs/spineps_all_%j.err" \
    -q gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G)
echo "  Job ID: $JOB1 (GPU, after Ian-Pan)"

# ─── Step 2: TotalSpineSeg — ALL ──────────────────────────────────────────────
echo ""
echo "Submitting: Step 2 — TotalSpineSeg (ALL valid studies)"
JOB2=$(sbatch --parsable \
    --job-name=tss_all \
    --time=96:00:00 \
    --dependency=afterok:$JOB0 \
    --wrap="
set -euo pipefail
export CONDA_PREFIX=\"\${HOME}/mambaforge/envs/nextflow\"
export PATH=\"\${CONDA_PREFIX}/bin:\$PATH\"
export XDG_RUNTIME_DIR=\"\${HOME}/xdr\"
export NXF_SINGULARITY_CACHEDIR=\"\${HOME}/singularity_cache\"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

PROJECT_DIR=\"$(pwd)\"
SCRATCH_DIR=\"/wsu/tmp/\${USER}/tss_all_\${SLURM_JOB_ID}\"
mkdir -p \"\$SCRATCH_DIR\" logs
export SINGULARITY_TMPDIR=\"\$SCRATCH_DIR\"
trap 'rm -rf \"\$SCRATCH_DIR\"' EXIT

TOTALSPINESEG_MODELS=\"\${PROJECT_DIR}/models/totalspineseg_models\"
NNUNET_TRAINER_DIR=\"\${PROJECT_DIR}/models/nnunetv2_trainer\"
mkdir -p \"\$TOTALSPINESEG_MODELS\" \"\$NNUNET_TRAINER_DIR\"

IMG_PATH=\"\${NXF_SINGULARITY_CACHEDIR}/totalspineseg.sif\"
if [[ ! -f \"\$IMG_PATH\" ]]; then
    singularity pull \"\$IMG_PATH\" docker://go2432/totalspineseg:latest
fi

if [[ -z \"\$(ls -A \$NNUNET_TRAINER_DIR 2>/dev/null)\" ]]; then
    singularity exec \"\$IMG_PATH\" \\
        cp -r /opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/. \\
        \"\$NNUNET_TRAINER_DIR/\"
fi

singularity exec --nv \\
    --bind \"\${PROJECT_DIR}\":/work \\
    --bind \"\${PROJECT_DIR}/results/nifti\":/work/results/nifti \\
    --bind \"\${PROJECT_DIR}/results/totalspineseg\":/work/results/totalspineseg \\
    --bind \"\${PROJECT_DIR}/models\":/app/models \\
    --bind \"\${TOTALSPINESEG_MODELS}\":/app/totalspineseg_models \\
    --bind \"\${NNUNET_TRAINER_DIR}\":/opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer \\
    --env TOTALSPINESEG_DATA=/app/totalspineseg_models \\
    --env PYTHONUNBUFFERED=1 \\
    --pwd /work \\
    \"\$IMG_PATH\" \\
    python3 -u /work/scripts/03b_totalspineseg_selective.py \\
        --nifti_dir   /work/results/nifti \\
        --output_dir  /work/results/totalspineseg \\
        --series_csv  /work/data/raw/train_series_descriptions.csv \\
        --valid_ids   /app/models/valid_id.npy \\
        --all
" \
    --output="logs/tss_all_%j.out" \
    --error="logs/tss_all_%j.err" \
    -q gpu --gres=gpu:1 --constraint=v100 --cpus-per-task=4 --mem=64G)
echo "  Job ID: $JOB2 (GPU, parallel with SPINEPS)"

# ─── Step 3: Morphometrics — ALL ──────────────────────────────────────────────
echo ""
echo "Submitting: Step 3 — Morphometrics (ALL)"
JOB3=$(sbatch --parsable \
    --job-name=morpho_all \
    --time=24:00:00 \
    --dependency=afterok:${JOB1}:${JOB2} \
    --wrap="
set -euo pipefail
export CONDA_PREFIX=\"\${HOME}/mambaforge/envs/nextflow\"
export PATH=\"\${CONDA_PREFIX}/bin:\$PATH\"
export XDG_RUNTIME_DIR=\"\${HOME}/xdr\"
export NXF_SINGULARITY_CACHEDIR=\"\${HOME}/singularity_cache\"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

PROJECT_DIR=\"$(pwd)\"
mkdir -p logs results/morphometrics results/morphometrics/reports

IMG_PATH=\"\${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif\"
if [[ ! -f \"\$IMG_PATH\" ]]; then
    singularity pull \"\$IMG_PATH\" docker://go2432/spineps-preprocessing:latest
fi

EXTRA_ARGS=\"\"
if [[ -f \"\${PROJECT_DIR}/results/lstv_detection/lstv_results.json\" ]]; then
    EXTRA_ARGS=\"--lstv_json /work/results/lstv_detection/lstv_results.json\"
fi

singularity exec \\
    --bind \"\${PROJECT_DIR}\":/work \\
    --bind \"\${PROJECT_DIR}/models\":/app/models \\
    --env PYTHONUNBUFFERED=1 \\
    --pwd /work \\
    \"\$IMG_PATH\" \\
    python3 -u /work/scripts/05_morphometrics.py \\
        --spineps_dir    /work/results/spineps \\
        --totalspine_dir /work/results/totalspineseg \\
        --output_dir     /work/results/morphometrics \\
        --all \\
        \$EXTRA_ARGS
" \
    --output="logs/morpho_all_%j.out" \
    --error="logs/morpho_all_%j.err" \
    -q primary --cpus-per-task=8 --mem=64G)
echo "  Job ID: $JOB3 (CPU, after SPINEPS + TotalSpineSeg)"

# ─── Step 4: Dataset summary report ──────────────────────────────────────────
echo ""
echo "Submitting: Step 4 — Dataset summary HTML report"
JOB4=$(sbatch --parsable \
    --job-name=dataset_report \
    --time=1:00:00 \
    --dependency=afterok:${JOB3} \
    --wrap="
set -euo pipefail
export CONDA_PREFIX=\"\${HOME}/mambaforge/envs/nextflow\"
export PATH=\"\${CONDA_PREFIX}/bin:\$PATH\"
export XDG_RUNTIME_DIR=\"\${HOME}/xdr\"
export NXF_SINGULARITY_CACHEDIR=\"\${HOME}/singularity_cache\"
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

PROJECT_DIR=\"$(pwd)\"
mkdir -p logs results

IMG_PATH=\"\${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif\"
if [[ ! -f \"\$IMG_PATH\" ]]; then
    singularity pull \"\$IMG_PATH\" docker://go2432/spineps-preprocessing:latest
fi

# LSTV summary report (if detection results exist)
if [[ -f \"\${PROJECT_DIR}/results/lstv_detection/lstv_results.json\" ]]; then
    echo 'Generating LSTV classification report...'
    singularity exec \\
        --bind \"\${PROJECT_DIR}\":/work \\
        --env PYTHONUNBUFFERED=1 \\
        --pwd /work \\
        \"\$IMG_PATH\" \\
        python3 -u /work/scripts/06_html_report.py \\
            --lstv_json      /work/results/lstv_detection/lstv_results.json \\
            --image_dir      /work/results/lstv_viz \\
            --output_html    /work/results/lstv_report.html \\
            --n_reps         3
    echo 'LSTV report → results/lstv_report.html'
fi

# Dataset morphometrics summary report (always)
singularity exec \\
    --bind \"\${PROJECT_DIR}\":/work \\
    --env PYTHONUNBUFFERED=1 \\
    --pwd /work \\
    \"\$IMG_PATH\" \\
    python3 -u /work/scripts/06_html_report.py \\
        --morphometrics_json /work/results/morphometrics/morphometrics_all.json \\
        --output_html        /work/results/dataset_morphometrics_report.html \\
        --morpho_only

echo 'Morphometrics report → results/dataset_morphometrics_report.html'
" \
    --output="logs/dataset_report_%j.out" \
    --error="logs/dataset_report_%j.err" \
    -q primary --cpus-per-task=2 --mem=16G)
echo "  Job ID: $JOB4 (CPU, after morphometrics)"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "ALL JOBS SUBMITTED"
echo "================================================================"
echo ""
echo "Dependency chain:"
echo "  $JOB0  Ian-Pan inference      (GPU)"
echo "    ├─→  $JOB1  SPINEPS ALL     (GPU, parallel)"
echo "    └─→  $JOB2  TotalSpineSeg ALL (GPU, parallel)"
echo "              ↓ (both done)"
echo "         $JOB3  Morphometrics ALL  (CPU)"
echo "              ↓"
echo "         $JOB4  Dataset report     (CPU)"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  watch -n 60 squeue -u $USER"
echo ""
echo "Expected runtime: 48-96h (depending on GPU queue)"
echo ""
echo "Final outputs:"
echo "  results/morphometrics/morphometrics_all.json"
echo "  results/morphometrics/morphometrics_all.csv"
echo "  results/morphometrics/morphometrics_summary.json"
echo "  results/dataset_morphometrics_report.html  ← dataset summary"
echo "  results/lstv_report.html                   ← LSTV classification (if available)"
echo "================================================================"

# ─── Wait and monitor ─────────────────────────────────────────────────────────
echo ""
echo "Master job now monitoring progress..."
echo ""

ALL_JOBS=("$JOB0" "$JOB1" "$JOB2" "$JOB3" "$JOB4")
JOB_NAMES=("Ian-Pan" "SPINEPS-ALL" "TotalSpineSeg-ALL" "Morphometrics-ALL" "Dataset-Report")

for i in "${!ALL_JOBS[@]}"; do
    job_id="${ALL_JOBS[$i]}"
    name="${JOB_NAMES[$i]}"
    echo "Waiting for job $job_id ($name)..."

    while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
        sleep 120
    done

    STATE=$(sacct -j "$job_id" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$STATE" == "COMPLETED" ]]; then
        echo "  ✓ $job_id ($name) — COMPLETED"
    else
        echo "  ✗ $job_id ($name) — $STATE"
        echo ""
        echo "Check logs:"
        echo "  ls -ltr logs/ | grep $job_id"
        echo "  cat logs/*${job_id}*.err"
        exit 1
    fi
done

echo ""
echo "================================================================"
echo "FULL DATASET PIPELINE COMPLETE!"
echo "$(date)"
echo "================================================================"
echo ""
echo "Key outputs:"
echo "  results/morphometrics/morphometrics_all.csv"
echo "  results/dataset_morphometrics_report.html"
echo "  results/lstv_report.html  (if LSTV detection was run)"
