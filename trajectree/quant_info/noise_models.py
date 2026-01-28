from scipy import sparse as sp
from scipy.linalg import expm

import numpy as np
from numpy import sqrt
import qutip as qt
from math import factorial
from functools import lru_cache

@lru_cache(maxsize=100)
def amplitude_damping(noise_parameter):

    ops = []
    ops.append(sp.csr_array([[1, 0], [0, sqrt(1 - noise_parameter)]]))
    ops.append(sp.csr_array([[0, sqrt(noise_parameter)], [0, 0]]))
    return ops

@lru_cache(maxsize=100)
def phase_damping(noise_parameter):

    ops = []
    ops.append(sp.csr_array([[1, 0], [0, sqrt(1 - noise_parameter)]]))
    ops.append(sp.csr_array([[0, 0], [0, sqrt(noise_parameter)]]))
    return ops  