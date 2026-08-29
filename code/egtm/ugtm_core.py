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

from __future__ import print_function

import math
import torch
import numpy as np
from scipy.spatial import distance

DEFAULT_DTYPE = torch.float64
_EPS = 1e-300


def _to_torch(x, device=None, dtype=DEFAULT_DTYPE):
    if isinstance(x, torch.Tensor):
        if device is None:
            device = x.device
        return x.to(device=device, dtype=dtype)
    if device is None:
        device = torch.device("cpu")
    return torch.as_tensor(x, device=device, dtype=dtype)


def _pairwise_sq_dists(A, B):
    A = _to_torch(A)
    B = _to_torch(B, device=A.device, dtype=A.dtype)
    A2 = torch.sum(A * A, dim=1, keepdim=True)
    B2 = torch.sum(B * B, dim=1, keepdim=True).T
    D = A2 + B2 - 2.0 * torch.matmul(A, B.T)
    return torch.clamp(D, min=0.0)


def _scale_like_sklearn(X):
    """Approximate sklearn.preprocessing.scale(..., with_mean=True, with_std=True)."""
    X = _to_torch(X)
    mean = torch.mean(X, dim=0, keepdim=True)
    std = torch.std(X, dim=0, unbiased=False, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (X - mean) / std


def createYMatrixInit(data, matW, matPhiMPlusOne):
    """Strict torch port of original ugtm createYMatrixInit.

    Returns shape: (n_dimensions, n_nodes)
    """
    matW = _to_torch(matW)
    matPhiMPlusOne = _to_torch(matPhiMPlusOne, device=matW.device, dtype=matW.dtype)
    data = _to_torch(data, device=matW.device, dtype=matW.dtype)

    shap1 = matW.shape[0]
    shap2 = matPhiMPlusOne.shape[0]
    the_means = torch.mean(data, dim=0)
    dm_mean_matrix = the_means[:, None].repeat(1, shap2).reshape(shap1, shap2)
    matY = torch.matmul(matW, matPhiMPlusOne.T)
    matY = matY + dm_mean_matrix
    return matY


def createPhiMatrix(matX, matM, numX, numM, sigma):
    """Strict torch port of original ugtm createPhiMatrix.

    Returns shape: (n_nodes, n_rbf_centers + 1)
    """
    matX = _to_torch(matX)
    matM = _to_torch(matM, device=matX.device, dtype=matX.dtype)
    sigma = _to_torch(sigma, device=matX.device, dtype=matX.dtype)

    result = torch.zeros((numX, numM + 1), device=matX.device, dtype=matX.dtype)
    for i in range(numX):
        for j in range(numM):
            coo1 = (matX[i][0] - matM[j][0]) * (matX[i][0] - matM[j][0])
            coo2 = (matX[i][1] - matM[j][1]) * (matX[i][1] - matM[j][1])
            dist = coo1 + coo2
            result[i, j] = torch.exp(-(dist) / (2 * sigma))
    result[:, numM] = 1.0
    return result


def computeWidth(matM, numM, sigma):
    """Strict torch port of original ugtm computeWidth."""
    matM = _to_torch(matM)
    sigma = _to_torch(sigma, device=matM.device, dtype=matM.dtype)

    result = torch.tensor(0.0, device=matM.device, dtype=matM.dtype)
    if numM <= 1:
        return sigma
    else:
        distances = torch.zeros((numM, numM), device=matM.device, dtype=matM.dtype)
        mins = torch.zeros((numM, 1), device=matM.device, dtype=matM.dtype)
        maxs = torch.zeros((numM, 1), device=matM.device, dtype=matM.dtype)
        for i in range(numM):
            for j in range(numM):
                coo1 = (matM[i][0] - matM[j][0]) * (matM[i][0] - matM[j][0])
                coo2 = (matM[i][1] - matM[j][1]) * (matM[i][1] - matM[j][1])
                distances[i, j] = coo1 + coo2
        for i in range(numM):
            nz = distances[i][distances[i] != 0]
            mins[i] = torch.min(nz)
        for i in range(numM):
            nz = distances[i][distances[i] != 0]
            maxs[i] = torch.max(nz)
        if sigma > 0.0:
            result = sigma * torch.mean(mins)
        else:
            result = torch.max(maxs)
        return result


def createWMatrix(matX, matPhiMPlusOne, matU, n_dimensions, n_rbf_centers):
    """Strict torch port of original ugtm createWMatrix.

    Returns shape: (n_dimensions, n_rbf_centers + 1)
    """
    matX = _to_torch(matX)
    matPhiMPlusOne = _to_torch(matPhiMPlusOne, device=matX.device, dtype=matX.dtype)
    matU = _to_torch(matU, device=matX.device, dtype=matX.dtype)

    normX = _scale_like_sklearn(matX)
    myProd = torch.matmul(matU, normX.T)
    tinv = torch.linalg.solve(torch.matmul(matPhiMPlusOne.T, matPhiMPlusOne), matPhiMPlusOne.T)
    result = torch.matmul(myProd, tinv.T)
    return result


def createYMatrix(matW, matPhiMPlusOne):
    """Strict torch port of original ugtm createYMatrix.

    Returns shape: (n_dimensions, n_nodes)
    """
    matW = _to_torch(matW)
    matPhiMPlusOne = _to_torch(matPhiMPlusOne, device=matW.device, dtype=matW.dtype)
    result = torch.matmul(matW, matPhiMPlusOne.T)
    return result


# def createDistanceMatrix(matY, data):
#     """Strict torch port of original ugtm createDistanceMatrix.

#     matY shape: (n_dimensions, n_nodes)
#     data shape: (n_individuals, n_dimensions)
#     returns: (n_nodes, n_individuals)
    
    

def createDistanceMatrix(matY, data, standarization=True):
    matY_t = _to_torch(matY)
    data_t = _to_torch(data, device=matY_t.device, dtype=matY_t.dtype)

    if standarization:
        return _pairwise_sq_dists(matY_t.T, data_t)
    else:
        matY_np = matY_t.detach().cpu().numpy()
        data_np = data_t.detach().cpu().numpy()
        D_np = distance.cdist(matY_np.T, data_np, metric="sqeuclidean")
        return torch.as_tensor(D_np, device=matY_t.device, dtype=matY_t.dtype)


def KERNELcreateDistanceMatrix(data, matL, matPhiMPlusOne):
    
    data = _to_torch(data)
    matL = _to_torch(matL, device=data.device, dtype=data.dtype)
    matPhiMPlusOne = _to_torch(matPhiMPlusOne, device=data.device, dtype=data.dtype)

    n_nodes = matPhiMPlusOne.shape[0]
    n_individuals = data.shape[0]
    result = torch.zeros((n_nodes, n_individuals), device=data.device, dtype=data.dtype)
    for i in range(n_nodes):
        LPhim = torch.matmul(matL, matPhiMPlusOne[i])
        thefloat = torch.matmul(torch.matmul(LPhim, data), LPhim)
        for j in range(n_individuals):
            result[i, j] = data[j, j] + thefloat - 2 * torch.dot(data[j], LPhim)
    return result


def exp_normalize(x):
    x = _to_torch(x)
    y = x - torch.max(x, dim=0, keepdim=True).values
    y = torch.exp(y)
    return y



def createPMatrix(matD, betaInv, n_dimensions, standarization=True):
    matD_t = _to_torch(matD)
    betaInv_t = _to_torch(betaInv, device=matD_t.device, dtype=matD_t.dtype)

    if standarization:
        beta = 1.0 / betaInv_t
        x = -(beta / 2.0) * matD_t
        x_max = torch.max(x, dim=0, keepdim=True).values
        matP = torch.exp(x - x_max)
        return matP
    else:
        matD_np = matD_t.detach().cpu().numpy()
        betaInv_np = float(betaInv_t.detach().cpu().item())

        beta = 1.0 / betaInv_np
        x = -(beta / 2.0) * matD_np
        x_max = np.max(x, axis=0, keepdims=True)
        matP_np = np.exp(x - x_max)

        return torch.as_tensor(matP_np, device=matD_t.device, dtype=matD_t.dtype)

def createRMatrix(matP):
    matP = _to_torch(matP)
    sums = torch.sum(matP, dim=0)
    matR = matP / sums[None, :]
    return matR


def createGMatrix(matR):
    matR = _to_torch(matR)
    sums = torch.sum(matR, dim=1)
    matG = torch.diag(sums)
    return matG


def optimWMatrix(matR, matPhiMPlusOne, matG, data, betaInv, alpha_0):
    matR = _to_torch(matR)
    matPhiMPlusOne = _to_torch(matPhiMPlusOne, device=matR.device, dtype=matR.dtype)
    matG = _to_torch(matG, device=matR.device, dtype=matR.dtype)
    data = _to_torch(data, device=matR.device, dtype=matR.dtype)
    betaInv = _to_torch(betaInv, device=matR.device, dtype=matR.dtype)
    alpha_0 = _to_torch(alpha_0, device=matR.device, dtype=matR.dtype)

    n_rbf_centersP = matPhiMPlusOne.shape[1]
    LBmat = torch.zeros((n_rbf_centersP, n_rbf_centersP), device=matR.device, dtype=matR.dtype)
    PhiGPhi = torch.matmul(torch.matmul(matPhiMPlusOne.T, matG), matPhiMPlusOne)
    for i in range(n_rbf_centersP):
        LBmat[i][i] = alpha_0 * betaInv
    PhiGPhiLB = PhiGPhi + LBmat
    Ginv = torch.linalg.inv(PhiGPhiLB)
    matW = torch.matmul(torch.matmul(torch.matmul(Ginv, matPhiMPlusOne.T), matR), data).T
    return matW


def optimLMatrix(matR, matPhiMPlusOne, matG, betaInv, alpha_0):
    matR = _to_torch(matR)
    matPhiMPlusOne = _to_torch(matPhiMPlusOne, device=matR.device, dtype=matR.dtype)
    matG = _to_torch(matG, device=matR.device, dtype=matR.dtype)
    betaInv = _to_torch(betaInv, device=matR.device, dtype=matR.dtype)
    alpha_0 = _to_torch(alpha_0, device=matR.device, dtype=matR.dtype)

    n_rbf_centersP = matPhiMPlusOne.shape[1]
    LBmat = torch.zeros((n_rbf_centersP, n_rbf_centersP), device=matR.device, dtype=matR.dtype)
    PhiGPhi = torch.matmul(torch.matmul(matPhiMPlusOne.T, matG), matPhiMPlusOne)
    for i in range(n_rbf_centersP):
        LBmat[i][i] = alpha_0 * betaInv
    PhiGPhiLB = PhiGPhi + LBmat
    Ginv = torch.linalg.inv(PhiGPhiLB)
    matW = torch.matmul(torch.matmul(Ginv, matPhiMPlusOne.T), matR).T
    return matW


def optimBetaInv(matR, matD, n_dimensions):
    matR = _to_torch(matR)
    matD = _to_torch(matD, device=matR.device, dtype=matR.dtype)
    n_individuals = matR.shape[1]
    betaInv = torch.sum(matR * matD) / (n_individuals * n_dimensions)
    return betaInv


def meanPoint(matR, matX):
    matR = _to_torch(matR)
    matX = _to_torch(matX, device=matR.device, dtype=matR.dtype)
    matMeans = torch.matmul(matR.T, matX)
    return matMeans


def modePoint(matR, matX):
    matR = _to_torch(matR)
    matX = _to_torch(matX, device=matR.device, dtype=matR.dtype)
    matModes = matX[torch.argmax(matR, dim=0), :]
    return matModes


# def computelogLikelihood(matP, betaInv, n_dimensions):
#     matP = _to_torch(matP)
#     betaInv = _to_torch(betaInv, device=matP.device, dtype=matP.dtype)
#     n_nodes = matP.shape[0]
#     n_individuals = matP.shape[1]
#     logLikelihood = torch.tensor(0.0, device=matP.device, dtype=matP.dtype)
#     prior = 1.0 / n_nodes
#     placeholder = 50
#     constante = torch.pow(((1.0 / betaInv) / (2.0 * math.pi)), min(n_dimensions / 2, placeholder))
#     tiny = torch.finfo(matP.dtype).tiny
#     logLikelihood = torch.sum(torch.log(torch.clamp(torch.sum(constante * matP, dim=0) * prior, min=tiny)))
#     logLikelihood = logLikelihood / n_individuals
#     return -logLikelihood


def computelogLikelihood(matP, betaInv, n_dimensions):
    matP = _to_torch(matP)
    betaInv = _to_torch(betaInv, device=matP.device, dtype=matP.dtype)

    n_nodes = matP.shape[0]
    n_individuals = matP.shape[1]
    prior = 1.0 / n_nodes

    exponent = min(n_dimensions / 2.0, 50.0)
    constante = ((1.0 / betaInv) / (2.0 * math.pi)) ** exponent

    s = constante * matP
    s = torch.sum(s, dim=0) * prior
    s = torch.clamp(s, min=1e-300)

    logLikelihood = torch.sum(torch.log(s))
    logLikelihood = logLikelihood / n_individuals

    return -logLikelihood
    

def computelogLikelihood_FF(matD, betaInv, n_dimensions):
    matD_t = _to_torch(matD)
    betaInv_t = _to_torch(betaInv, device=matD_t.device, dtype=matD_t.dtype)

    matD_np = matD_t.detach().cpu().numpy().astype(np.longdouble)
    betaInv_np = np.longdouble(float(betaInv_t.detach().cpu().item()))

    beta = np.longdouble(1.0) / betaInv_np
    x = np.longdouble(-(beta / 2.0) * matD_np)
    x = x - np.max(x, axis=0, keepdims=True)
    matP = np.exp(x, dtype=np.longdouble)

    n_nodes = matP.shape[0]
    prior = np.longdouble(1.0) / np.longdouble(n_nodes)
    exponent = min(n_dimensions / 2.0, 50)

    base = (float(1.0 / betaInv_np)) / (2.0 * np.pi)
    constante = np.longdouble(np.power(base, exponent))

    s = np.sum(constante * matP, axis=0) * prior
    s = np.maximum(s, np.finfo(np.longdouble).tiny)

    return float(-np.sum(np.log(s)) / matD_np.shape[1])



def evalBetaInv(matY, betaInv, random_state=1234):
    matY = _to_torch(matY)
    betaInv = _to_torch(betaInv, device=matY.device, dtype=matY.dtype)
    g = torch.Generator(device=matY.device)
    g.manual_seed(int(random_state))

    distances = torch.sqrt(_pairwise_sq_dists(matY.T, matY.T))
    myMin = torch.mean(distances) / 2.0
    myMin = myMin * myMin
    if (myMin < betaInv) or bool((betaInv == 0).item() if betaInv.numel() == 1 else torch.all(betaInv == 0).item()):
        betaInv = myMin
    if bool((betaInv == 0).item() if betaInv.numel() == 1 else torch.all(betaInv == 0).item()):
        print("bad initialization (0 variance), setting variance to random number...")
        betaInv = torch.abs(-1.0 + 2.0 * torch.rand(1, generator=g, device=matY.device, dtype=matY.dtype)).squeeze()
    return betaInv


def initBetaInvRandom(matD, n_nodes, n_individuals, n_dimensions):
    matD = _to_torch(matD)
    betaInv = torch.sum(matD * (1.0 / n_nodes)) / (n_individuals * n_dimensions)
    return betaInv




def _gaussian_cross_entropy_torch(data, sigma=1.0, device=None, dtype=DEFAULT_DTYPE):
    x = _to_torch(data, device=device, dtype=dtype)

    mu = torch.mean(x, dim=0)
    var = torch.var(x, dim=0, unbiased=False)

    sigma2 = torch.tensor(float(sigma) ** 2, device=x.device, dtype=x.dtype)
    sigma2 = torch.clamp(sigma2, min=torch.finfo(x.dtype).tiny)

    ce = 0.5 * torch.sum((var + mu * mu) / sigma2 + torch.log(2.0 * torch.pi * sigma2))
    return ce
