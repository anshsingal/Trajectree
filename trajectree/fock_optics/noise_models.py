from scipy import sparse as sp
from scipy.linalg import expm
from .devices import rx, ry, rz, global_phase

import numpy as np
from numpy import sqrt
import qutip as qt
from math import factorial
from functools import lru_cache


def single_mode_bosonic_noise_channels(noise_parameter, N):
    """This function produces the Kraus operatorsd for the single mode bosonic noise channels. This includes pure loss and 
    pure gain channels. The pure gain channel is simply the transpose of the pure loss channel.
    
    Args:
        noise_parameter (float): The noise parameter, (loss for pure loss and gain for pure gain channels). For the pure loss channel, this 
                                 parameter is the dimensionless noise term: 1-transmissivity (of beamsplitter in beamsplitter model of attenuation).
                                 For a fiber, transmissivity = e**(-chi), where chi = l/l_att, where l is the length of the fiber and 
                                 l_att is the attenuation length. If the noise_parameter is greater than 1, it is assumed to be a gain channel.
        N (int): local Hilbert space dimension being considered.
    """
    a = qt.destroy(N).full()
    a_dag = qt.create(N).full()
    n = a_dag @ a
    
    # TODO: Theoretically, verify these
    normalization = 1
    gain_channel = False

    if noise_parameter > 1: 
        gain_channel = True
        normalization = np.sqrt(1/noise_parameter)
        noise_parameter = (noise_parameter-1)/(noise_parameter) # Convert gain to loss parameter

    kraus_ops = []
    for l in range(N): # you can lose anywhere from 0 to N-1 (=trunc) photons in the truncated Hilbert space. 
        kraus_ops.append(sp.csr_array(normalization * np.sqrt(1/factorial(l) * (noise_parameter/(1-noise_parameter))**l) * (np.linalg.matrix_power(a, l) @ expm(n/2 * np.log(1-noise_parameter)))))

    if gain_channel: 
        for l in range(N):
            kraus_ops[l] = kraus_ops[l].T.conjugate()

    return kraus_ops

@lru_cache(maxsize=100)
def depolarizing_operators(depolarizing_probability, N):
    ops = []
    ops.append(sqrt(1-depolarizing_probability) * sp.eye(N**2, format="csc"))
    ops.append(sqrt((depolarizing_probability)/3) * sp.csr_matrix(rx(np.pi, N, return_unitary=True)))
    ops.append(sqrt((depolarizing_probability)/3) * sp.csr_matrix(ry(np.pi, N, return_unitary=True)))
    ops.append(sqrt((depolarizing_probability)/3) * sp.csr_matrix(rz(np.pi, N, return_unitary=True)))
    return ops

def two_qubit_depolarizing_channel(depolarizing_probability, N):
    """This function produces the Kraus operators for the two qubit depolarizing channel.
    
    Args:
        depolarizing_probability (float): The depolarizing probability.
        N (int): local Hilbert space dimension being considered.
    """
    single_qubit_ops = [sp.csr_matrix(global_phase(0, N, return_unitary = True)), sp.csr_matrix(rx(np.pi, N, return_unitary=True)), sp.csr_matrix(ry(np.pi, N, return_unitary=True)), sp.csr_matrix(rz(np.pi, N, return_unitary=True))]
    ops = []
    ops.append(sqrt(1-(15/16)*depolarizing_probability) * sp.eye(N**4, format="csc"))
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            else:
                ops.append(sqrt(depolarizing_probability/16) * sp.kron(single_qubit_ops[i], single_qubit_ops[j]))
    return ops
