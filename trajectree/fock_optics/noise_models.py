from scipy import sparse as sp
from scipy.linalg import expm

import numpy as np

import qutip as qt
from math import factorial

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