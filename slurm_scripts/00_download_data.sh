#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=dl_rsna
#SBATCH -o logs/dl_rsna_%j.out
#SBATCH -e logs/dl_rsna_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "================================================================"
echo "DOWNLOAD RSNA DATA + IAN PAN MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================"

# --- Environment ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME
which singularity || echo "WARNING: singularity not found"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- Paths ---
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw"
MODELS_DIR="${PROJECT_DIR}/models"
mkdir -p "$DATA_DIR" "$MODELS_DIR" logs "${PROJECT_DIR}/.tmp_dl"

# --- Kaggle check ---
KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: Kaggle credentials not found at $KAGGLE_JSON"
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to: https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New Token' under API section"
    echo "  3. Save kaggle.json to ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi
echo "✓ Kaggle credentials found"

# --- Container (KEEP THE ORIGINAL) ---
CONTAINER="docker://go2432/spineps-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-preprocessing.sif"
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# --- Prepare Kaggle mount ---
mkdir -p ${PROJECT_DIR}/.kaggle_tmp
cp ${HOME}/.kaggle/kaggle.json ${PROJECT_DIR}/.kaggle_tmp/
chmod 600 ${PROJECT_DIR}/.kaggle_tmp/kaggle.json

echo ""
echo "================================================================"
echo "STEP 1: Download Model Checkpoint + Validation IDs"
echo "================================================================"

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind ${PROJECT_DIR}/.kaggle_tmp:/root/.kaggle \
    --pwd /work \
    "$IMG_PATH" \
    bash -c "
        set -e
        cd .tmp_dl
        
        echo 'Downloading rsna2024-demo-workflow (Ian Pan checkpoint)...'
        if [ ! -f rsna2024-demo-workflow.zip ]; then
            kaggle datasets download -d hengck23/rsna2024-demo-workflow
        else
            echo '  Already downloaded, skipping.'
        fi
        
        echo 'Extracting checkpoint and validation IDs...'
        unzip -j -o rsna2024-demo-workflow.zip 00002484.pth valid_id.npy
        
        mv 00002484.pth /work/models/point_net_checkpoint.pth
        mv valid_id.npy /work/models/valid_id.npy
        
        echo '✓ Model checkpoint: /work/models/point_net_checkpoint.pth'
        echo '✓ Validation IDs: /work/models/valid_id.npy'
    "

if [[ ! -f "${MODELS_DIR}/point_net_checkpoint.pth" ]]; then
    echo "ERROR: Model checkpoint not extracted"
    exit 1
fi

echo ""
echo "================================================================"
echo "STEP 2: Download RSNA Competition Data"
echo "================================================================"

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind ${PROJECT_DIR}/.kaggle_tmp:/root/.kaggle \
    --pwd /work \
    "$IMG_PATH" \
    bash -c "
        set -e
        cd .tmp_dl
        
        echo 'Downloading train_series_descriptions.csv...'
        kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification -f train_series_descriptions.csv
        
        if [ -f 'train_series_descriptions.csv.zip' ]; then
            echo 'Unzipping CSV...'
            unzip -o train_series_descriptions.csv.zip
        elif [ -f 'train_series_descriptions.csv' ]; then
            echo 'CSV downloaded uncompressed (skipping unzip).'
        else
            echo 'ERROR: CSV file not found after download'
            ls -lh
            exit 1
        fi
        
        mv train_series_descriptions.csv /work/data/raw/
        echo '✓ Series CSV extracted'
        
        echo ''
        echo 'Downloading full competition zip (DICOM images)...'
        ZIP_FILE='rsna-2024-lumbar-spine-degenerative-classification.zip'
        
        if [ ! -f \"\$ZIP_FILE\" ]; then
            kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
        else
            echo '  Zip already downloaded, skipping.'
        fi
        
        echo ''
        echo 'Extracting validation set studies only...'
        python3 -c \"
import zipfile, numpy as np, os
from pathlib import Path

valid_ids = set(str(x) for x in np.load('/work/models/valid_id.npy'))
print(f'Target: {len(valid_ids)} validation studies')

# Check which studies already exist
output_dir = Path('/work/data/raw/train_images')
existing_studies = set(d.name for d in output_dir.iterdir() if d.is_dir()) if output_dir.exists() else set()
studies_to_extract = valid_ids - existing_studies

if not studies_to_extract:
    print('✓ All studies already extracted, skipping.')
else:
    print(f'Already extracted: {len(existing_studies)}')
    print(f'Extracting: {len(studies_to_extract)} new studies')
    
    with zipfile.ZipFile('\$ZIP_FILE', 'r') as z:
        to_extract = [
            f for f in z.namelist()
            if f.startswith('train_images/') 
            and len(f.split('/')) > 1 
            and f.split('/')[1] in studies_to_extract
        ]
        print(f'Total files to extract: {len(to_extract)}')
        z.extractall('/work/data/raw', members=to_extract)
    
    print('✓ Extraction complete')
\"
    "

exit_code=$?

# Cleanup
rm -rf ${PROJECT_DIR}/.kaggle_tmp

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Download failed"
    exit $exit_code
fi

# Verify everything
echo ""
echo "================================================================"
echo "VERIFICATION"
echo "================================================================"

echo ""
echo "Model files:"
ls -lh ${MODELS_DIR}/point_net_checkpoint.pth
ls -lh ${MODELS_DIR}/valid_id.npy

echo ""
echo "Data files:"
ls -lh ${DATA_DIR}/train_series_descriptions.csv
N_STUDIES=$(ls ${DATA_DIR}/train_images/ | wc -l)
echo "DICOM studies extracted: $N_STUDIES"

echo ""
echo "================================================================"
echo "DOWNLOAD COMPLETE"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Run SPINEPS:    sbatch slurm_scripts/02_spineps_segmentation.sh"
echo "  2. Run inference:  sbatch slurm_scripts/03_centroid_inference.sh"
echo "  3. View report:    sbatch slurm_scripts/04_generate_report.sh"
echo "================================================================"
