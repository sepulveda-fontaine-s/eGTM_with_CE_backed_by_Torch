#!/bin/bash
# Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
# SPDX-License-Identifier: MIT

#SBATCH --job-name=torch_yamls_gpu
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --output=/path_to_project/eGTM/logs/gpu-%A_%a.out
#SBATCH --error=/path_to_project/eGTM/logs/gpu-%A_%a.err
#SBATCH --mail-user=emaila@gserver.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT="/path_to_project/eGTM"
CODE_DIR="${PROJECT_ROOT}/code"
YAML_DIR="${CODE_DIR}/yamls_GPU"
RUNNER="${CODE_DIR}/run_yaml_slurm.py"
MAIN_FILE="${CODE_DIR}/main_torch.py"
OUTPUT_DIR="${PROJECT_ROOT}/results"
PROCESSED_DIR="${YAML_DIR}/Processed_GPU"
LOG_DIR="${PROJECT_ROOT}/logs/yaml_runs_gpu"
FOLDER_LIST="${PROJECT_ROOT}/yaml_folders_gpu.txt"

module load Python3/3.10.5
source "${PROJECT_ROOT}/python-env/bin/activate"

PYTHON_BIN="$(command -v python)"
"${PYTHON_BIN}" --version



export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${OUTPUT_DIR}" "${PROCESSED_DIR}" "${LOG_DIR}"

cd "${CODE_DIR}"

if [ ! -f "${FOLDER_LIST}" ]; then
    echo "ERROR: folder list not found: ${FOLDER_LIST}"
    echo "Create it before submitting the array."
    exit 1
fi

FOLDER="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${FOLDER_LIST}")"

if [ -z "${FOLDER}" ]; then
    echo "No folder for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 0
fi

if [ ! -d "${FOLDER}" ]; then
    echo "Folder does not exist, probably already processed:"
    echo "${FOLDER}"
    exit 0
fi

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "YAML_DIR: ${YAML_DIR}"
echo "SELECTED FOLDER: ${FOLDER}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "PROCESSED_DIR: ${PROCESSED_DIR}"
echo "LOG_DIR: ${LOG_DIR}"

echo "NVIDIA-SMI:"
nvidia-smi || true

echo "TORCH CUDA CHECK:"
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY

"${PYTHON_BIN}" "${RUNNER}" \
    --folder "${FOLDER}" \
    --main-file "${MAIN_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --processed-dir "${PROCESSED_DIR}" \
    --log-dir "${LOG_DIR}" \
    --workers 1
