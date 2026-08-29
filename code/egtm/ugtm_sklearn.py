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


"""GTM transformer compatible with sklearn, Torch-native eGTM wrapper."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import torch
import numpy as np
from . import ugtm_gtm


class eGTM(BaseEstimator, TransformerMixin):
    """
    Torch-backed version of eGTM.

    Internal computation uses Torch and public transform outputs stay as Torch
    tensors, except for model="complete" which returns the projected object.
    """

    def __init__(
        self,
        k=16,
        m=4,
        s=0.3,
        alpha_0=0.1,
        random_state=1234,
        niter=200,
        verbose=False,
        model="means",
        sigma=1.0,
        eta=1e-3,
        use_ce=False,
        device=None,
        dtype=torch.float64,
    ):
        assert model in ("means", "modes", "responsibilities", "complete"), (
            "model must be either of 'means', 'modes', 'responsibilities', or 'complete'"
        )
        self.k = k
        self.m = m
        self.s = s
        self.alpha_0 = alpha_0
        self.random_state = random_state
        self.niter = niter
        self.verbose = verbose
        self.model = model

        # Torch / CE extensions
        self.sigma = sigma
        self.eta = eta
        self.use_ce = use_ce
        self.device = device
        self.dtype = dtype

    def _resolve_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensor(self, x):
        device = self._resolve_device()
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=self.dtype)
        return torch.as_tensor(x, device=device, dtype=self.dtype)

    def fit(self, X, y=None, standarization=True):
        X = check_array(X)
        X = self._to_tensor(X)

        self.initialModel = ugtm_gtm.initialize(
            X,
            self.k,
            self.m,
            self.s,
            random_state=self.random_state,
            device=self._resolve_device(),
            dtype=self.dtype,
        )

        optimize_out = ugtm_gtm.optimize(
            X,
            self.initialModel,
            self.alpha_0,
            self.niter,
            verbose=self.verbose,
            sigma=self.sigma,
            eta=self.eta,
            use_ce=self.use_ce,
            device=self._resolve_device(),
            dtype=self.dtype,
            standarization=standarization,
        )

        if isinstance(optimize_out, tuple):
            self.optimizedModel, self.log_likelihoods = optimize_out
        else:
            self.optimizedModel = optimize_out
            self.log_likelihoods = None

        return self

    def transform(self, X):
        check_is_fitted(self, ["optimizedModel"])
        X = check_array(X)
        X = self._to_tensor(X)

        self.projected = ugtm_gtm.projection(
            self.optimizedModel,
            X,
            device=self._resolve_device(),
            dtype=self.dtype,
        )

        outputs = {
            "complete": self.projected,
            "means": self.projected.matMeans,
            "modes": self.projected.matModes,
            "responsibilities": self.projected.matR,
        }
        return outputs[self.model]

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self, matR):
        check_is_fitted(self, ["optimizedModel", "initialModel"])
        matR = self._to_tensor(matR)
        weightedPhi = torch.matmul(matR, self.initialModel.matPhiMPlusOne)
        return torch.matmul(weightedPhi, self.optimizedModel.matW.transpose(0, 1))
