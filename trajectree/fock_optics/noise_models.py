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

    This implementation is based on the definitions in the paper: https://doi.org/10.1103/PhysRevA.97.032346
    
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

def _nck(n,k):
    """Compute the binomial coefficient "n choose k"."""
    if k < 0 or k > n:
        return 0
    return factorial(n) / (factorial(k) * factorial(n - k))

def general_coherent_bs_noise_model(bath_parameter, theta, N, bath_type='coherent'):
    """This function produces the Kraus operators for a general beamsplitter noise model with a pure coherent bath.
    
    Args:
        bath_parameter (float): The parameter of the bath (thermal or coherent).
        theta (float): The beamsplitter transmissivity.
        N (int): local Hilbert space dimension being considered.
    """
    a = qt.destroy(N).full()
    a_dag = qt.create(N).full()
    n = a_dag @ a

    basis = lambda i: qt.states.basis(N, i).full()
    if bath_type == 'coherent':
        bath_state = lambda n: np.sqrt(np.exp(-np.abs(bath_parameter)**2) / factorial(n)) * bath_parameter**n 
        
    kraus_ops = []

    for k in range(N):
        kraus_op = 0
        for n in range(N):
            for q in range(N):
                coeff = 0
                for r_2 in range(n):
                    coeff += bath_state(n) * np.sqrt(_nck(q, q+n-(k+r_2)) * _nck(n, r_2)) * (1j)**(n-r_2) * np.cos(theta)**(k+2*r_2-n) * np.sin(theta)**(q+2*n - k - 2*r_2)
                kraus_op += coeff * basis(q+n-k) @ basis(q).T
        kraus_ops.append(kraus_op)

    return kraus_ops


def general_mixed_bs_noise_model(dark_count_rate, eta, N):
    """This function produces the Kraus operators for a general beamsplitter noise model with mixed thermal bath.
    
    Args:
        dark_count_rate (float): Rate of detector dark counts per second
        theta (float): The beamsplitter transmissivity.
        N (int): local Hilbert space dimension being considered.
    """
    a = qt.destroy(N).full()
    a_dag = qt.create(N).full()
    n = a_dag @ a

    basis = lambda i: qt.states.basis(N, i).full()
    r = np.atanh(np.sqrt(dark_count_rate/(1-eta+eta*dark_count_rate))) # We can also calulate the mean photon number of the fictitious thermal bath as sinh(r)**2

    kraus_ops = []

    theta = np.arcsin(np.sqrt(eta))

    for n in range(N):
        bath_state = np.sqrt(np.cosh(r)**2 * np.tanh(r)**(2*n)) 
        for k in range(N):
            kraus_op = 0
            # flag = False
            for q in range(N):
                if q+n-k < N and q+n-k >= 0:
                    # flag = True
                    coeff = 0
                    for r_2 in range(n+1):
                        coeff += bath_state * np.sqrt(_nck(q, q+n-(k+r_2)) * _nck(n, r_2)) * (1j)**(n-r_2) * np.cos(theta)**(k+2*r_2-n) * np.sin(theta)**(q+2*n - k - 2*r_2)
                        # try:
                        kraus_op += coeff * sp.csr_array((basis(q+n-k) @ basis(q).T))
                        # except:
                        #     raise ValueError(f"Basis {q+n-k} is not possible for N={N}")
            # if flag:
            kraus_ops.append(kraus_op)

    return kraus_ops
                
            

@lru_cache(maxsize=100)
def depolarizing_operators(depolarizing_probability, N, bias = (1/3, 1/3, 1/3)):
    ops = []
    ops.append(sqrt(1-depolarizing_probability) * sp.eye(N**2, format="csc"))
    ops.append(sqrt(depolarizing_probability * bias[0]) * sp.csr_matrix(rx(np.pi, N, return_unitary=True)))
    ops.append(sqrt(depolarizing_probability * bias[1]) * sp.csr_matrix(ry(np.pi, N, return_unitary=True)))
    ops.append(sqrt(depolarizing_probability * bias[2]) * sp.csr_matrix(rz(np.pi, N, return_unitary=True)))
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
