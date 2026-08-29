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

import torch


DEFAULT_DTYPE = torch.float64


def _to_torch(x, device=None, dtype=DEFAULT_DTYPE):
    if isinstance(x, torch.Tensor):
        if device is None:
            device = x.device
        return x.to(device=device, dtype=dtype)
    if device is None:
        device = torch.device("cpu")
    return torch.as_tensor(x, device=device, dtype=dtype)


class ReturnU(object):
    def __init__(self, matU, betaInv, device=None, dtype=DEFAULT_DTYPE):
        self.matU = _to_torch(matU, device=device, dtype=dtype)
        self.betaInv = _to_torch(betaInv, device=self.matU.device, dtype=dtype)
        self.device = self.matU.device
        self.dtype = self.matU.dtype


class InitialGTM(object):

    def __init__(
        self,
        matX,
        matM,
        n_nodes,
        n_rbf_centers,
        rbfWidth,
        matPhiMPlusOne,
        matW,
        matY,
        betaInv,
        n_dimensions,
        device=None,
        dtype=torch.float64,
    ):

        self.matX = _to_torch(matX, device=device, dtype=dtype)
        self.matM = _to_torch(matM, device=self.matX.device, dtype=dtype)

        self.n_rbf_centers = int(n_rbf_centers)
        self.n_nodes = int(n_nodes)

        self.rbfWidth = _to_torch(rbfWidth, device=self.matX.device, dtype=dtype)

        self.matPhiMPlusOne = _to_torch(matPhiMPlusOne, device=self.matX.device, dtype=dtype)

        self.matW = _to_torch(matW, device=self.matX.device, dtype=dtype)

        self.matY = _to_torch(matY, device=self.matX.device, dtype=dtype)

        self.betaInv = _to_torch(betaInv, device=self.matX.device, dtype=dtype)

        self.n_dimensions = int(n_dimensions)

        self.device = self.matX.device
        self.dtype = self.matX.dtype


class OptimizedGTM(object):

    def __init__(
        self,
        matW,
        matY,
        matP,
        matR,
        betaInv,
        matMeans,
        matModes,
        matX,
        n_dimensions,
        converged,
        device=None,
        dtype=torch.float64,
    ):
        self.matW = _to_torch(matW, device=device, dtype=dtype)
        self.matY = _to_torch(matY, device=self.matW.device, dtype=dtype)
        self.matP = _to_torch(matP, device=self.matW.device, dtype=dtype)
        self.matR = _to_torch(matR, device=self.matW.device, dtype=dtype)
        self.betaInv = _to_torch(betaInv, device=self.matW.device, dtype=dtype)
        self.matMeans = _to_torch(matMeans, device=self.matW.device, dtype=dtype)
        self.matModes = _to_torch(matModes, device=self.matW.device, dtype=dtype)
        self.matX = _to_torch(matX, device=self.matW.device, dtype=dtype)
        self.n_dimensions = int(n_dimensions)
        self.converged = bool(converged)
        self.device = self.matW.device
        self.dtype = self.matW.dtype