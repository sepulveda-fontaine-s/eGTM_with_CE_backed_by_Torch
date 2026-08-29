"""
Copyright (c) 2026 Salomé A. Sepúlveda-Fontaine
SPDX-License-Identifier: MIT

This file contains code derived in part from the original `ugtm` Python package.

Original project: ugtm: Generative Topographic Mapping with Python
Original author: Héléna A. Gaspar
Original repository: https://github.com/hagax8/ugtm
Original license: MIT License

The original `ugtm` MIT license notice is preserved in:
licenses/ugtm-MIT-LICENSE.txt

Modifications in this version include device-aware eGTM execution,
PyTorch-compatible numerical paths, CPU/GPU execution support, and
cross-entropy-based dynamic regularization experiments.
"""


"""Functions to run GTM models with Torch backend, preserving original ugtm logic."""



import numpy as np
import torch
from sklearn.decomposition import PCA



from . import ugtm_classes
from . import ugtm_core


DEFAULT_DTYPE = torch.float64


def _resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_torch(x, device=None, dtype=DEFAULT_DTYPE):
    device = _resolve_device(device)
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _to_numpy_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def initialize(data, k, m, s, random_state=1234, device=None, dtype=DEFAULT_DTYPE):
    r"""Initializes a GTM model.

    This is a strict torch port of the original ugtm initialization path.
    Shapes and control flow are preserved.
    """
    device = _resolve_device(device)
    data_t = _to_torch(data, device=device, dtype=dtype)
    data_np = _to_numpy_cpu(data_t)

    n_dimensions = data_np.shape[1]
    n_nodes = k * k
    n_rbf_centers = m * m

    x = np.linspace(-1, 1, k)
    matX = np.transpose(np.meshgrid(x, x)).reshape(k * k, 2)
    x = np.linspace(-1, 1, m)
    matM = np.transpose(np.meshgrid(x, x)).reshape(m * m, 2)

    if m == 1:
        matM = np.array([[0.0, 0.0]])
    if k == 1:
        matX = np.array([[0.0, 0.0]])

    matX_t = _to_torch(matX, device=device, dtype=dtype)
    matM_t = _to_torch(matM, device=device, dtype=dtype)

    rbfWidth = ugtm_core.computeWidth(matM_t, n_rbf_centers, s)
    matPhiMPlusOne = ugtm_core.createPhiMatrix(
        matX_t, matM_t, n_nodes, n_rbf_centers, rbfWidth
    )

    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(data_np)
    matU = (pca.components_.T * np.sqrt(pca.explained_variance_))[:, 0:2]
    betaInv = pca.explained_variance_[2]

    Uobj = ugtm_classes.ReturnU(matU, betaInv, device=device, dtype=dtype)
    matW = ugtm_core.createWMatrix(
        matX_t, matPhiMPlusOne, Uobj.matU, n_dimensions, n_rbf_centers
    )
    matY = ugtm_core.createYMatrixInit(data_t, matW, matPhiMPlusOne)
    betaInv = ugtm_core.evalBetaInv(matY, Uobj.betaInv, random_state=random_state)

    # print("rbfWidth:", rbfWidth)
    # print("PCA betaInv:", pca.explained_variance_[2])
    # print("evalBetaInv:", betaInv)
    # print("matW[0,:5]:", matW[0, :5])
    # print("matY[:,0]:", matY[:, 0])


    return ugtm_classes.InitialGTM(
        matX_t,
        matM_t,
        n_nodes,
        n_rbf_centers,
        rbfWidth,
        matPhiMPlusOne,
        matW,
        matY,
        betaInv,
        n_dimensions,
        device=device,
        dtype=dtype,
        #return_log_likelihoods=return_log_likelihoods,
        #choose_standarization=choose_standarization,
    )


def optimize(
    data,
    initialModel,
    alpha_0,
    niter,
    verbose=False,
    sigma=1.0,
    eta=1e-3,
    use_ce=False,
    device=None,
    dtype=DEFAULT_DTYPE,
    return_log_likelihoods=False,
    standarization=True,
):
    r"""Optimizes a GTM model.

    Strict torch port of the original ugtm EM loop.
    """
    device = _resolve_device(device if device is not None else getattr(initialModel, "device", None))
    data_t = _to_torch(data, device=device, dtype=dtype)

    matD = ugtm_core.createDistanceMatrix(initialModel.matY, data_t)
    matY = initialModel.matY
    betaInv = initialModel.betaInv

    i = 1
    diff = 1000.0
    converged = 0
    log_likelihoods = []

    while i < (niter + 1) and (converged < 4):
        
        matP = ugtm_core.createPMatrix(
            matD,
            betaInv,
            initialModel.n_dimensions,
            standarization=standarization,
        )
        matR = ugtm_core.createRMatrix(matP)



        matG = ugtm_core.createGMatrix(matR)


        effective_regul = alpha_0
        if use_ce:
            ce_val = ugtm_core._gaussian_cross_entropy_torch(
                data_t, sigma=sigma, device=device, dtype=dtype
            )
            ce_value = float(ce_val)

            effective_regul = alpha_0 * (1.0 + eta * ce_value)
        
            # if i == 1:
            #     print("DEBUG effective_regul =", effective_regul)
        else:
            ce_value = "not applicable"

        matW = ugtm_core.optimWMatrix(
            matR,
            initialModel.matPhiMPlusOne,
            matG,
            data_t,
            betaInv,
            effective_regul,
        )
        matY = ugtm_core.createYMatrix(matW, initialModel.matPhiMPlusOne)
        matD = ugtm_core.createDistanceMatrix(matY, data_t)
        betaInv = ugtm_core.optimBetaInv(matR, matD, initialModel.n_dimensions)

        if i == 1:
            loglike = ugtm_core.computelogLikelihood(
                matP, betaInv, initialModel.n_dimensions
            )
        else:
            loglikebefore = loglike
            loglike = ugtm_core.computelogLikelihood(
                matP, betaInv, initialModel.n_dimensions
            )
            diff = abs(float(loglikebefore) - float(loglike))
            if diff <= 0.0001:
                converged += 1
            else:
                converged = 0

        if verbose:
            print("Iter ", i, " Err: ", float(loglike))

        log_likelihoods.append(float(loglike))
        i += 1

    if verbose and converged >= 3:
        print("Converged: ", float(loglike))

    has_converged = converged >= 3

    matP = ugtm_core.createPMatrix(
            matD,
            betaInv,
            initialModel.n_dimensions,
            standarization=standarization,
        )
    matR = ugtm_core.createRMatrix(matP)
    
    
    matMeans = ugtm_core.meanPoint(matR, initialModel.matX)
    matModes = ugtm_core.modePoint(matR, initialModel.matX)

    optimized = ugtm_classes.OptimizedGTM(
        matW,
        matY,
        matP.T,
        matR.T,
        betaInv,
        matMeans,
        matModes,
        initialModel.matX,
        initialModel.n_dimensions,
        has_converged,
        device=device,
        dtype=dtype,
        #return_log_likelihoods=return_log_likelihoods,
        #choose_standarization=choose_standarization,
        )
    optimized.ce_value = ce_value
    optimized.log_likelihoods = log_likelihoods

    if return_log_likelihoods:
        return optimized, log_likelihoods
    return optimized


def projection(optimizedModel, new_data, device=None, dtype=DEFAULT_DTYPE, standarization=True):
    r"""Project test set on optimized GTM model. No pre-processing involved."""
    device = _resolve_device(device if device is not None else getattr(optimizedModel, "device", None))
    new_data_t = _to_torch(new_data, device=device, dtype=dtype)

    matD = ugtm_core.createDistanceMatrix(
        optimizedModel.matY,
        new_data_t,
        standarization=standarization,
    )
    matP = ugtm_core.createPMatrix(
        matD,
        optimizedModel.betaInv,
        optimizedModel.n_dimensions,
        standarization=standarization,
    )
    matR = ugtm_core.createRMatrix(matP)
    

    
    matMeans = ugtm_core.meanPoint(matR, optimizedModel.matX)
    matModes = ugtm_core.modePoint(matR, optimizedModel.matX)



    projected = ugtm_classes.OptimizedGTM(
        optimizedModel.matW,
        optimizedModel.matY,
        matP.T,
        matR.T,
        optimizedModel.betaInv,
        matMeans,
        matModes,
        optimizedModel.matX,
        optimizedModel.n_dimensions,
        optimizedModel.converged,
        device=device,
        dtype=dtype,
    )
    projected.ce_value = getattr(optimizedModel, "ce_value", None)
    return projected



def runGTM(
    data,
    k=16,
    m=4,
    s=0.3,
    alpha_0=0.1,
    doPCA=False,
    n_components=-1,
    missing=True,
    missing_strategy="median",
    random_state=1234,
    niter=200,
    verbose=False,
    sigma=1.0,
    eta=1e-3,
    use_ce=False,
    device=None,
    dtype=DEFAULT_DTYPE,
    return_log_likelihoods=False,
    standarization=True,
):
    r"""Run GTM (wrapper for initialize + optimize)."""
    if k == 0:
        k = int(np.sqrt(5 * np.sqrt(data.shape[0]))) + 2
    if m == 0:
        m = int(np.sqrt(k))

    work_data = data

    initialModel = initialize(
        work_data,
        k,
        m,
        s,
        random_state=random_state,
        device=device,
        dtype=dtype,
    )
    return optimize(
    work_data,
    initialModel,
    alpha_0,
    niter,
    verbose,
    sigma=sigma,
    eta=eta,
    use_ce=use_ce,
    device=device,
    dtype=dtype,
    return_log_likelihoods=return_log_likelihoods,
    standarization=standarization,
)
