'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


# Make imports work when running locally, in Colab, or from a script.
# Expected repository layout: code/main_torch.py, code/TOP3_torch.py, code/egtm/...
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

importlib.invalidate_caches()

import egtm
print("Using egtm from:", egtm.__file__)
from egtm.ugtm_sklearn import eGTM
from egtm.ugtm_gtm import projection
from egtm.ugtm_core import createDistanceMatrix, createPMatrix, computelogLikelihood

from TOP3_torch import grid_search_top3, save_results


RANDOM_STATE = 42
DTYPE = torch.float64


def resolve_device():
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for k, v in cfg.items():
        if isinstance(v, list) and len(v) == 1:
            cfg[k] = v[0]

    return cfg


def load_dataset(cfg):
    df = pd.read_csv(cfg["csv_path"], index_col=0)

    if cfg.get("num_rows") is not None:
        df = df.iloc[: cfg["num_rows"], :]

    if cfg.get("num_columns") is not None:
        df = df.iloc[:, : cfg["num_columns"]]

    return df.values.astype("float64")


def run_experiment(config_path, output_dir="results"):
    cfg = load_yaml_config(config_path)

    cfg["resolved_device"] = resolve_device()
    cfg["device_name"] = cfg["resolved_device"].type

    X = load_dataset(cfg)
    print("Starting grid search...")
    df_top3 = grid_search_top3(X, cfg)
    print("Grid search finished.")

    save_results(df_top3, cfg, output_dir)

    return {
        "config": cfg,
        "df_top3": df_top3,
        "X": X,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run one Torch eGTM YAML experiment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file, e.g. code/yamls_CPU/2_gen_by_MAD_1230x5000/TF_1_gen_proc_sorted5000.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/results_torch",
        help="Directory where the result CSV will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(config_path=args.config, output_dir=args.output_dir)
