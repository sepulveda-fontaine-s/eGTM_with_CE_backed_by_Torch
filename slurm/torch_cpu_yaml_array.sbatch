#!/bin/bash
# Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
# SPDX-License-Identifier: MIT

#SBATCH --job-name=torch_yamls_cpu
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=150:00:00
#SBATCH --output=/path_to_project/eGTM/logs/cpu-%A_%a.out
#SBATCH --error=/path_to_project/eGTM/logs/cpu-%A_%a.err
#SBATCH --mail-user=email@server.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT="/path_to_project/eGTM"
CODE_DIR="${PROJECT_ROOT}/code"
YAML_DIR="${CODE_DIR}/yamls_CPU"
RUNNER="${CODE_DIR}/run_yaml_slurm.py"
MAIN_FILE="${CODE_DIR}/main_torch.py"
OUTPUT_DIR="${PROJECT_ROOT}/results"
PROCESSED_DIR="${YAML_DIR}/Processed_CPU"
LOG_DIR="${PROJECT_ROOT}/logs/yaml_runs_cpu_torch"
FOLDER_LIST="${PROJECT_ROOT}/yaml_folders_cpu_torch.txt"

# Use Python 3.10.5 env on Dantzig.
module load Python3/3.10.5
source "${PROJECT_ROOT}/python-env/bin/activate"

PYTHON_BIN="$(command -v python)"
"${PYTHON_BIN}" --version

# One Python process per YAML. Avoid hidden CPU over-subscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Force CPU execution. Remove this only if you intentionally want CUDA/GPU use.
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${OUTPUT_DIR}" "${PROCESSED_DIR}" "${LOG_DIR}"

cd "${CODE_DIR}"

if [ ! -f "${FOLDER_LIST}" ]; then
    echo "ERROR: folder list not found: ${FOLDER_LIST}"
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
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "PROCESSED_DIR: ${PROCESSED_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo

"${PYTHON_BIN}" "${RUNNER}" \
    --folder "${FOLDER}" \
    --main-file "${MAIN_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --processed-dir "${PROCESSED_DIR}" \
    --log-dir "${LOG_DIR}" \
    --workers 4
