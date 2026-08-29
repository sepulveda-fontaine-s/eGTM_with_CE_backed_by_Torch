'''
    Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
    SPDX-License-Identifier: MIT
'''


import sys
import time
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.model_selection import ParameterGrid


# Make imports work when this file is executed from a notebook or a script.
# Expected repository layout: code/TOP3_torch.py and code/egtm/...
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

importlib.invalidate_caches()

import egtm
print("Using egtm from:", egtm.__file__)
from egtm.ugtm_sklearn import eGTM
from egtm.ugtm_gtm import projection
from egtm.ugtm_core import (
    createDistanceMatrix,
    createPMatrix,
    computelogLikelihood,
    computelogLikelihood_FF,
    _scale_like_sklearn,
)


torch.set_printoptions(precision=15, sci_mode=False)
np.set_printoptions(precision=15, suppress=False)

RANDOM_STATE = 42
DTYPE = torch.float64


def standardize_data(X):
    return _scale_like_sklearn(X)


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_normalized_mse(X_true, X_pred):
    X_true = as_numpy(X_true)
    X_pred = as_numpy(X_pred)

    mse = mean_squared_error(X_true, X_pred)
    var_x = np.var(X_true)

    if var_x <= 0:
        return np.inf

    return mse / var_x


def safe_silhouette_best_k(Z, k_sil, random_state=42):
    try:
        Z = as_numpy(Z)

        if Z.ndim != 2 or Z.shape[0] < 3:
            return np.nan, np.nan

        if not isinstance(k_sil, (list, tuple)):
            k_sil = [k_sil]

        valid_scores = []

        for k in k_sil:
            k = int(k)

            if k < 2:
                continue
            if k >= Z.shape[0]:
                continue

            labels = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
            ).fit_predict(Z)

            if len(np.unique(labels)) < 2:
                continue

            sil = silhouette_score(Z, labels)
            valid_scores.append((sil, k))

        if not valid_scores:
            return np.nan, np.nan

        best_sil, best_k_sil = max(valid_scores, key=lambda x: x[0])
        return best_sil, best_k_sil

    except Exception:
        return np.nan, np.nan


def build_model(params, cfg):
    return eGTM(
        k=params["k"],
        m=params["m"],
        s=params["s"],
        alpha_0=params["alpha_0"],
        sigma=1.0,
        eta=float(params.get("eta", 0.0)),
        random_state=RANDOM_STATE,
        niter=params.get("niter", 200),
        verbose=False,
        model="means",
        use_ce=bool(cfg["ce_regularization"]),
        device=cfg["resolved_device"],
        dtype=DTYPE,
    )


def compute_neg_avg_loglik(model, X, cfg):
    if bool(cfg["standarization"]):
        return compute_neg_avg_loglik_FT(model, X)
    return compute_neg_avg_loglik_FF(model, X)


def compute_neg_avg_loglik_FT(model, X):
    proj = projection(
        model.optimizedModel,
        X,
        device=model.optimizedModel.device,
        dtype=model.optimizedModel.dtype,
        standarization=True,
    )

    X_t = torch.as_tensor(X, dtype=proj.matY.dtype, device=proj.matY.device)
    matD = createDistanceMatrix(proj.matY, X_t, standarization=True)
    matP = createPMatrix(matD, proj.betaInv, proj.n_dimensions, standarization=True)

    neg_avg_loglik = float(
        computelogLikelihood(matP, proj.betaInv, proj.n_dimensions)
    )

    n_samples = X.shape[0]
    neg_loglik_total = neg_avg_loglik * n_samples
    return neg_avg_loglik, neg_loglik_total


def compute_neg_avg_loglik_FF(model, X):
    proj = projection(
        model.optimizedModel,
        X,
        device=model.optimizedModel.device,
        dtype=model.optimizedModel.dtype,
        standarization=False,
    )

    X_t = torch.as_tensor(X, dtype=proj.matY.dtype, device=proj.matY.device)
    matD = createDistanceMatrix(proj.matY, X_t, standarization=False)

    neg_avg_loglik = float(
        computelogLikelihood_FF(matD, proj.betaInv, proj.n_dimensions)
    )

    n_samples = X.shape[0]
    neg_loglik_total = neg_avg_loglik * n_samples
    return neg_avg_loglik, neg_loglik_total


def grid_search_top3(X, cfg):
    X = torch.as_tensor(X, dtype=DTYPE)

    if bool(cfg["standarization"]):
        X_fit = standardize_data(X)
    else:
        X_fit = X

    rows = []
    grid_start_time = time.time()
    print("Starting Grid Search\n")

    dataset_name = Path(cfg["csv_path"]).stem
    data_size = f"{X_fit.shape[0]}x{X_fit.shape[1]}"

    effective_param_grid = dict(cfg["param_grid"])
    k_sil_values = cfg.get("k_sil", effective_param_grid.pop("k_sil", [2, 3, 4, 5, 6]))

    use_ce = bool(cfg["ce_regularization"])

    if use_ce:
        if "eta" not in effective_param_grid:
            eta_key = "eta_std" if bool(cfg["standarization"]) else "eta_nonstd"
            effective_param_grid["eta"] = effective_param_grid.pop(eta_key)

        for key in ("eta_std", "eta_nonstd"):
            effective_param_grid.pop(key, None)
    else:
        for key in ("eta", "eta_std", "eta_nonstd"):
            effective_param_grid.pop(key, None)

    param_combinations = list(ParameterGrid(effective_param_grid))
    total_experiments = len(param_combinations)

    for exp_idx, params in enumerate(param_combinations, start=1):
        try:
            print(f"Running {exp_idx}/{total_experiments} experiments")
            gtm_start_time = time.time()

            model = build_model(params, cfg)

            if bool(cfg["standarization"]):
                model.fit(X_fit, standarization=True)
            else:
                model.fit(X_fit, standarization=False)

            gtm_time = time.time() - gtm_start_time

            proj = projection(
                model.optimizedModel,
                X_fit,
                device=model.optimizedModel.device,
                dtype=model.optimizedModel.dtype,
                standarization=bool(cfg["standarization"]),
            )

            Z = proj.matMeans
            silhouette_val, best_k_sil = safe_silhouette_best_k(Z, k_sil_values)

            neg_avg_loglik, neg_loglik_total = compute_neg_avg_loglik(model, X_fit, cfg)
            normalized_neg_avg_loglik = neg_avg_loglik

            X_rec = model.inverse_transform(proj.matR)
            normalized_mse = compute_normalized_mse(X_fit, X_rec)

            ce_value = model.optimizedModel.ce_value if use_ce else "not applicable"

            row = {
                "Dataset": dataset_name,
                "Data Size": data_size,
                "Standardization": bool(cfg["standarization"]),
                "Best k": params.get("k", np.nan),
                "Best m": params.get("m", np.nan),
                "Best s": params.get("s", np.nan),
                "Best alpha_0": params.get("alpha_0", np.nan),
                "Best eta": params["eta"] if use_ce else "not applicable",
                "Best k_sil": best_k_sil,
                "CE value": ce_value,
                "Normalized NLL": normalized_neg_avg_loglik,
                "Normalized MSE": normalized_mse,
                "Silhouette Score": silhouette_val,
                "Execution Time": np.nan,
                "Grid Search Time": np.nan,
                "GTM Time": gtm_time,
            }

            rows.append(row)

        except Exception:
            import traceback
            print(f"ERROR params={params}")
            traceback.print_exc()

    grid_time = time.time() - grid_start_time
    print("Grid Search finished\n")

    final_columns = [
        "Dataset",
        "Data Size",
        "Standardization",
        "Best k",
        "Best m",
        "Best s",
        "Best alpha_0",
        "Best eta",
        "Best k_sil",
        "CE value",
        "Normalized NLL",
        "Normalized MSE",
        "Silhouette Score",
        "Execution Time",
        "Grid Search Time",
        "GTM Time",
    ]

    if not rows:
        print("No successful parameter combinations.")
        return pd.DataFrame(columns=final_columns)

    df = pd.DataFrame(rows)
    df["Grid Search Time"] = grid_time
    df["Execution Time"] = df["Grid Search Time"] + df["GTM Time"]

    if len(df) == 1:
        return df[final_columns]

    best_ll = df.loc[df["Normalized NLL"].idxmin()]
    best_mse = df.loc[df["Normalized MSE"].idxmin()]
    best_sil = df.loc[df["Silhouette Score"].idxmax()] if df["Silhouette Score"].notna().any() else df.iloc[0]

    df_top3 = pd.DataFrame([best_ll, best_mse, best_sil]).reset_index(drop=True)
    return df_top3[final_columns]


def save_results(df_top3, cfg, output_dir="results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acronym = cfg.get("acronym", "GTM")
    num_rows = cfg.get("num_rows", "all")
    num_columns = cfg.get("num_columns", "all")
    dataset_name = Path(cfg["csv_path"]).stem

    filename = f"METRICS_torch_{cfg['device_name']}_{dataset_name}_{num_rows}_{num_columns}_{acronym}.csv"
    filepath = output_dir / filename

    print("FILEPATH BEFORE SAVE:", filepath)
    df_top3.to_csv(filepath, index=False)
