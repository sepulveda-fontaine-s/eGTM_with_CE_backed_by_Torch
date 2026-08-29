# Code for “eGTM: data dimensionality reduction with Cross Entropy as a data-adaptive regularization factor, backed by Torch”

This repository corresponds to the paper **“eGTM: dimensionality reduction with Cross Entropy (CE) as a data-adaptive regularization factor, backed by Torch”** and contains the research code developed for that study.

Specifically, it provides a PyTorch-hybrid-backed implementation of **extended Generative Topographic Mapping (eGTM)** with **Cross Entropy (CE)** as a data-adaptive regularization signal. The method extends classical GTM by rescaling the baseline smoothness coefficient into a run-specific, data-dependent effective coefficient used throughout EM optimization:

```text
alpha_eff = alpha_0 * (1 + eta * CE)
```

where `alpha_0` is the baseline GTM regularization parameter, `eta` controls the sensitivity of the adaptive term, and `CE` is the cross-entropy signal computed from the data representation used during optimization.

The goal of the project is to evaluate whether CE-driven adaptive regularization can improve GTM model selection and latent-space organization while preserving the probabilistic structure of the original GTM formulation.

## Project context

Generative Topographic Mapping (GTM) is a probabilistic nonlinear dimensionality-reduction model in which a regular latent grid is mapped into the data space through radial basis functions. Classical GTM controls manifold smoothness through a fixed regularization coefficient. In high-dimensional settings, a fixed coefficient may be too rigid for some datasets and too flexible for others.

This project introduces a CE-based data-adaptive regularization mechanism. For each training run, the data matrix actually used for optimization is summarized through a Gaussian approximation and compared with an isotropic Gaussian reference distribution. The resulting CE value is used to modulate the regularization strength. Because the data representation and the reference distribution remain fixed within a run, both CE and alpha_eff remain constant across EM iterations. Larger CE values induce stronger smoothing, while smaller CE values keep the effective regularization closer to the classical GTM setting.

The implementation also adapts the eGTM execution path to PyTorch, enabling device-aware execution on CPU or GPU while preserving the original GTM workflow and evaluation logic.

## Main features

- Cross-Entropy-driven data-adaptive regularization for eGTM.
- PyTorch-backed execution path with explicit `device` and `dtype` handling.
- CPU and GPU execution support for the Torch implementation.
- YAML-driven experiment configuration.
- Grid search over GTM hyperparameters.
- Evaluation through normalized negative log-likelihood, normalized reconstruction error, and Silhouette score.
- Local/Colab runner for simple reproducible execution.
- Optional Slurm runner for folder-based execution on HPC clusters.

## Repository layout

The current public repository is organized as follows:

```text
.
├── code/
│   ├── egtm/
│   │   ├── __init__.py
│   │   ├── ugtm_classes.py
│   │   ├── ugtm_core.py
│   │   ├── ugtm_gtm.py
│   │   └── ugtm_sklearn.py
│   ├── yamls_CPU/
│   │   ├── 2_gen_by_MAD_1230x5000/
│   │   │   ├── TF_1_gen_proc_sorted5000.yaml
│   │   │   └── TT_1_gen_proc_sorted5000.yaml.done
│   │   └── Processed_CPU/
│   │       └── .gitkeep
│   ├── yamls_GPU/
│   │   ├── 2_gen_by_MAD_1230x5000/
│   │   │   ├── TF_1_gen_proc_sorted5000.yaml
│   │   │   └── TT_1_gen_proc_sorted5000.yaml.done
│   │   └── Processed_GPU/
│   │       └── .gitkeep
│   ├── main_torch.py
│   ├── TOP3_torch.py
│   ├── run_colab_or_local.py
│   └── run_yaml_slurm.py
├── licenses/
│   └── ugtm-MIT-LICENSE.txt
├── logs/
│   ├── .gitkeep
│   └── yaml_runs_cpu_torch/
│       └── example_runner_log.log
├── results/
│   ├── native_results/
│   ├── torch_results_CPU/
│   └── torch_results_GPU/
├── slurm/
│   ├── torch_cpu_yaml_array.sbatch
│   └── torch_gpu_yaml_array.sbatch
├── LICENSE
├── README.md
├── THIRD_PARTY_LICENSES.md
├── requirements.txt
├── yaml_folders_cpu_torch.txt
└── yaml_folders_gpu.txt
```

The `code/egtm/` directory contains the adapted eGTM implementation derived in part from the original `ugtm` package. The original file names are preserved where appropriate, but the public package namespace used by this repository is `egtm`.

The main Torch execution path is defined by:

```text
code/main_torch.py
code/TOP3_torch.py
```

The repository also includes two execution helpers:

```text
code/run_colab_or_local.py
code/run_yaml_slurm.py
```

The `yamls_CPU/` and `yamls_GPU/` folders contain example YAML batches. Files ending in `.yaml` are pending experiments. Files ending in `.yaml.done` illustrate YAML files that have already completed successfully during a previous run.

The `Processed_CPU/` and `Processed_GPU/` folders are destination folders used by the Slurm runner. A YAML batch folder is moved there only after all pending `*.yaml` files inside that batch have completed.

The `logs/` folder contains a sanitized example of a per-YAML runner log. Runtime Slurm output, error files, and large execution logs are not part of the repository by default.

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the repository dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The provided Slurm templates assume that the virtual environment is located at:

```text
/path_to_project/eGTM/python-env/
```

If you use a different environment name, such as `.venv`, edit the activation line in the sbatch files accordingly.

For GPU execution, install a PyTorch build compatible with your CUDA version. In Google Colab, selecting a GPU runtime is usually sufficient.

## Data format

Input data must be provided as a preprocessed CSV file containing a numeric matrix. Rows correspond to observations and columns correspond to model variables after loading.

The current loader reads CSV files with the first column as the row index:

```python
pd.read_csv(cfg["csv_path"], index_col=0)
```

Therefore, the first column of the CSV is treated as an identifier column and is not used as a model variable. This first column may contain dates, sample identifiers, gene identifiers, row labels, or a placeholder column, depending on the dataset.

All remaining columns must be numeric, because they are converted to a `float64` matrix before fitting the model.

After loading, the code selects the first `num_rows` rows and the first `num_columns` model columns:

```python
df = df.iloc[:cfg["num_rows"], :]
df = df.iloc[:, :cfg["num_columns"]]
```

The matrix passed to eGTM therefore has shape:

```text
(num_rows, num_columns)
```

Large datasets are not included in this repository. Public YAML examples use portable placeholders such as:

```text
/path_to_project/eGTM/datasets/dataset.csv
```

Users should replace these placeholders with the path to their own preprocessed CSV file.

## YAML configuration

Each experiment is defined by one YAML file. The repository includes example YAML batches under:

```text
code/yamls_CPU/
code/yamls_GPU/
```

The YAML files included in these folders are placeholders that illustrate the expected organization of batch folders. They are not intended to be executed without first replacing their contents with a complete YAML configuration following the structure below.

Each batch folder may contain several YAML files. The Slurm runner processes all pending `*.yaml` files inside a selected batch folder.

The repository uses the field name `standarization` because this is the spelling used internally by the current codebase.

Example YAML structure:

```yaml
ce_regularization: True   # True or False
standarization: True      # True or False
acronym: "TT"             # TT, TF, FT, or FF

# FILE TO FEED THE MODEL
num_rows: 1230            # integer
num_columns: 5000         # integer
csv_path: "/path_to_project/eGTM/datasets/dataset.csv"

# Grid
param_grid: { "k": [8, 12, 16, 20], "m": [3, 5, 7, 9], "s": [0.75, 1, 1.25, 1.5], "alpha_0": [0.1, 0.15, 0.2, 0.25],
  "k_sil": [2, 3, 4, 5, 6], "eta_nonstd": [1e-14, 1e-13, 1e-12, 1e-11, 1e-10], "eta_std": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] }
```

The acronym summarizes two binary settings:

```text
TT = CE enabled, standardization enabled
TF = CE enabled, standardization disabled
FT = CE disabled, standardization enabled
FF = CE disabled, standardization disabled
```

During execution, completed YAML files are renamed with the `.done` suffix. For example:

```text
TF_1_gen_proc_sorted5000.yaml       # pending
TT_1_gen_proc_sorted5000.yaml.done  # completed
```

A batch folder is moved to `Processed_CPU/` or `Processed_GPU/` only when there are no pending `*.yaml` files left inside it.

## Running one experiment locally or in Colab

Run a single YAML file using `main_torch.py`:

```bash
python code/main_torch.py \
  --config code/yamls_CPU/2_gen_by_MAD_1230x5000/TF_1_gen_proc_sorted5000.yaml \
  --output-dir results/torch_results_CPU
```

Alternatively, use the local/Colab runner:

```bash
python code/run_colab_or_local.py \
  --config code/yamls_CPU/2_gen_by_MAD_1230x5000/TF_1_gen_proc_sorted5000.yaml \
  --main-file code/main_torch.py \
  --output-dir results/torch_results_CPU
```

Both commands execute one YAML experiment. The local/Colab runner is mainly useful when you want to reuse the same interface for either a single YAML file or a folder of YAML files.

## Running a folder of YAML files locally or in Colab

A YAML batch folder can be executed with the local/Colab runner. The runner selects pending files ending in `.yaml` inside the folder.

Before running the example commands below, replace the placeholder YAML files with complete experiment YAML files. Empty placeholder `.yaml` files are intentionally included only to document the batch-folder workflow.

For example, to execute a CPU YAML batch sequentially:

```bash
python code/run_colab_or_local.py \
  --config-dir code/yamls_CPU/2_gen_by_MAD_1230x5000 \
  --main-file code/main_torch.py \
  --output-dir results/torch_results_CPU \
  --workers 1 \
  --mark-done
```

To execute several YAML files from the same folder in parallel, increase `--workers`:

```bash
python code/run_colab_or_local.py \
  --config-dir code/yamls_CPU/2_gen_by_MAD_1230x5000 \
  --main-file code/main_torch.py \
  --output-dir results/torch_results_CPU \
  --workers 4 \
  --mark-done
```

In both examples, `code/yamls_CPU/2_gen_by_MAD_1230x5000` is the folder containing the YAML files to execute, and `results/torch_results_CPU` is created automatically if it does not already exist.

The `--mark-done` option renames successfully completed YAML files with the `.done` suffix. This prevents them from being selected again in later folder runs.


## Running on a Slurm cluster

The Slurm runner is optional. It is intended for HPC executions where experiments are organized into YAML batch folders.

The repository includes two Slurm templates:

```text
slurm/torch_cpu_yaml_array.sbatch
slurm/torch_gpu_yaml_array.sbatch
```

The CPU template is used for PyTorch CPU runs. The GPU template is used for PyTorch GPU runs. The `cpu_torch` suffix is used to distinguish PyTorch CPU runs from the original/native CPU execution path.

Before submitting a Slurm job, edit the path and cluster-specific fields at the top of each sbatch file, especially:

```bash
PROJECT_ROOT="/path_to_project/eGTM"
CODE_DIR="${PROJECT_ROOT}/code"
RUNNER="${CODE_DIR}/run_yaml_slurm.py"
MAIN_FILE="${CODE_DIR}/main_torch.py"
OUTPUT_DIR="${PROJECT_ROOT}/results"
```

The CPU template uses:

```bash
YAML_DIR="${CODE_DIR}/yamls_CPU"
PROCESSED_DIR="${YAML_DIR}/Processed_CPU"
LOG_DIR="${PROJECT_ROOT}/logs/yaml_runs_cpu_torch"
FOLDER_LIST="${PROJECT_ROOT}/yaml_folders_cpu_torch.txt"
```

The GPU template uses:

```bash
YAML_DIR="${CODE_DIR}/yamls_GPU"
PROCESSED_DIR="${YAML_DIR}/Processed_GPU"
LOG_DIR="${PROJECT_ROOT}/logs/yaml_runs_gpu"
FOLDER_LIST="${PROJECT_ROOT}/yaml_folders_gpu.txt"
```

The folder-list files are stored at the repository root:

```text
yaml_folders_cpu_torch.txt
yaml_folders_gpu.txt
```

Each line in these files must point to one YAML batch folder, not to an individual YAML file. For example:

```text
/path_to_project/eGTM/code/yamls_CPU/4_gen_by_MAD_1230x15000
/path_to_project/eGTM/code/yamls_CPU/5_gen_by_corr_1230x30000
```

and for GPU:

```text
/path_to_project/eGTM/code/yamls_GPU/4_gen_by_MAD_1230x15000
/path_to_project/eGTM/code/yamls_GPU/5_gen_by_corr_1230x30000
```

The example paths are methodological placeholders. Replace them with the actual YAML batch folders available in your execution environment.

Submit the CPU array job with:

```bash
CPU_N=$(wc -l < yaml_folders_cpu_torch.txt)
sbatch --array=1-${CPU_N}%4 slurm/torch_cpu_yaml_array.sbatch
```

Submit the GPU array job with:

```bash
GPU_N=$(wc -l < yaml_folders_gpu.txt)
sbatch --array=1-${GPU_N}%1 slurm/torch_gpu_yaml_array.sbatch
```

The `%4` and `%1` limits control how many array tasks are allowed to run at the same time. Adjust these values according to the available CPUs, GPUs, memory, and cluster policy.

During execution, the Slurm script selects one folder from the corresponding folder-list file using the current `SLURM_ARRAY_TASK_ID`. The runner then executes all pending `*.yaml` files inside that folder.

Completed YAML files are renamed with the `.done` suffix. The full YAML batch folder is moved to `Processed_CPU/` or `Processed_GPU/` only when no pending `*.yaml` files remain inside it.


### Slurm and runner logs

Slurm generates standard output and error files for each array task. In the provided templates, these files are directed to the `logs/` folder through the `#SBATCH --output` and `#SBATCH --error` directives:

```text
logs/cpu-<job_id>_<array_id>.out
logs/cpu-<job_id>_<array_id>.err
logs/gpu-<job_id>_<array_id>.out
logs/gpu-<job_id>_<array_id>.err
```

The parent `logs/` directory should exist before submitting the Slurm job, because Slurm resolves the stdout/stderr paths before the script body is executed.

Older or cluster-specific runs may instead produce files in the submission directory, for example:

```text
slurm-8960_1.out
slurm-8960_1.err
```

These files are runtime artifacts and are not included in the repository.

The Python runner also creates one log file per YAML execution under:

```text
logs/yaml_runs_cpu_torch/
logs/yaml_runs_gpu/
```

A sanitized example runner log is included only to document the expected execution trace.

## CPU and GPU execution

The Torch implementation resolves the execution device automatically:

```text
CUDA available     -> GPU execution
CUDA not available -> CPU execution
```

For CPU-only Torch runs, the CPU Slurm template explicitly disables CUDA:

```bash
export CUDA_VISIBLE_DEVICES=""
```

This prevents the PyTorch execution path from selecting a GPU even if CUDA is available on the system.

For GPU Torch runs, use the GPU Slurm template:

```text
slurm/torch_gpu_yaml_array.sbatch
```

The GPU template requests one GPU resource and performs a CUDA diagnostic before running the selected YAML folder:

```bash
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
```

It also prints:

```text
nvidia-smi
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
```

Native uGTM execution is CPU-only. GPU execution applies only to the Torch-adapted implementation.

## Output files

Each completed YAML experiment writes a CSV file with the selected best configurations. The filename follows the pattern:

```text
METRICS_torch_<device>_<dataset_name>_<num_rows>_<num_columns>_<acronym>.csv
```

where `<device>` indicates whether the Torch execution used CPU or GPU.

Typical reported columns include:

```text
Dataset
Data Size
Standardization
Best k
Best m
Best s
Best alpha_0
Best eta
Best k_sil
CE value
Normalized NLL
Normalized MSE
Silhouette Score
Execution Time
Grid Search Time
GTM Time
```

The repository keeps output folders separated by execution path:

```text
results/native_results/
results/torch_results_CPU/
results/torch_results_GPU/
```

The Slurm templates use:

```bash
OUTPUT_DIR="${PROJECT_ROOT}/results"
```

This can be changed to a more specific folder, such as:

```bash
OUTPUT_DIR="${PROJECT_ROOT}/results/torch_results_CPU"
OUTPUT_DIR="${PROJECT_ROOT}/results/torch_results_GPU"
```

depending on how the user wants to organize runtime outputs.

## Evaluation metrics

The experimental pipeline reports three complementary criteria:

1. **Normalized negative log-likelihood**: measures probabilistic fit under the GTM density model. Lower values are better.
2. **Normalized MSE**: measures reconstruction error normalized by the variance of the input data. Lower values are better.
3. **Silhouette score**: evaluates geometric separation in the latent representation using K-Means over candidate `k_sil` values. Higher values are better.

These metrics are reported jointly because they capture different aspects of GTM behavior and may select different hyperparameter configurations.

## Path policy for public repositories

Replace local or cluster-specific paths such as:

```text
/home/username/project/...
/content/drive/MyDrive/...
```

with portable placeholders:

```text
/path_to_project/eGTM
/path_to_project/eGTM/datasets/dataset.csv
/path_to_project/eGTM/code/yamls_CPU/example_batch
/path_to_project/eGTM/code/yamls_GPU/example_batch
results/torch_results_CPU
results/torch_results_GPU
```

The example YAML files and folder-list files included in this repository use placeholder paths. Users should replace them with paths that match their own local, Colab, or cluster environment.

## Reproducibility notes

- Experiments are controlled through YAML files.
- YAML batch folders can be executed locally, in Colab, or through Slurm array jobs.
- Completed YAML files are renamed with the `.done` suffix to avoid re-running them.
- Batch folders are moved to `Processed_CPU/` or `Processed_GPU/` only after all pending `*.yaml` files inside the folder have completed.
- The current implementation uses `torch.float64` for numerical stability.
- The same YAML structure can be used for Torch CPU and Torch GPU runs.
- Native uGTM execution is CPU-only; GPU execution applies to the Torch-adapted implementation.
- Random behavior is controlled through the `RANDOM_STATE` value set in the code.

## Citation

If you use this repository, please cite the associated manuscript once available:

```bibtex
@article{sepulveda_fontaine_egtm_ce,
  title = {eGTM: data dimensionality reduction with Cross Entropy as a data-adaptive regularization factor, backed by Torch},
  author  = {Sepúlveda-Fontaine, Salomé A. and Gómez Alcobendas, David and Rodríguez-Sala, Jesús J. and Amigó, José M.},
  year    = {2026},
  }
```

## Third-party code attribution

Parts of this repository are derived from the `ugtm` Python package by
Héléna A. Gaspar, originally distributed under the MIT License. The full
original `ugtm` license text is preserved in
`licenses/ugtm-MIT-LICENSE.txt`, with additional attribution provided in
`THIRD_PARTY_LICENSES.md`.

## License

Original source code written for this repository is released under the MIT
License.

Files derived from the original `ugtm` package remain subject to the original
`ugtm` MIT License notice, preserved in `licenses/ugtm-MIT-LICENSE.txt`.

Documentation, figures, tables, and accompanying written materials are released
under the Creative Commons Attribution 4.0 International License (CC BY 4.0),
unless otherwise stated.
