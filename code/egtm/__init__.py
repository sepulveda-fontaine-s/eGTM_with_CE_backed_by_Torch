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


"""ugtm: a python package for Generative Topographic Mapping (GTM)"""



from .ugtm_gtm import *
from .ugtm_sklearn import *
from .ugtm_classes import *
from .ugtm_core import *