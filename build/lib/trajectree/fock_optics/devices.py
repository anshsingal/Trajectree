from scipy.linalg import expm

import numpy as np
from numpy import kron

from quimb.tensor import MatrixProductOperator as mpo #type: ignore

import qutip as qt

# Beamsplitter transformation
def create_BS_MPO(site1, site2, theta, total_sites, N, tag = 'BS'): 

    a = qt.destroy(N).full()
    a_dag = a.T
    I = np.eye(N)
    
    # This corresponds to the BS hamiltonian:

    hamiltonian_BS = -theta * ( kron(I, a_dag)@kron(a, I) - kron(I, a)@kron(a_dag, I) )
    unitary_BS = expm(hamiltonian_BS)

    # print("unitary_BS", unitary_BS)

    BS_MPO = mpo.from_dense(unitary_BS, dims = N, sites = (site1,site2), L=total_sites, tags=tag)
    # BS_MPO = BS_MPO.fill_empty_sites(mode = "full")
    return BS_MPO


def generalized_mode_mixer(site1, site2, theta, phi, psi, lamda, total_sites, N, tag = 'MM'): 

    a = qt.destroy(N).full()
    a_dag = a.T
    I = np.eye(N)
    
    # This corresponds to the BS hamiltonian: This is a different difinition from the one in 
    # create_BS_MPO. This is because of how the generalized beamsplitter is defined in DOI: 10.1088/0034-4885/66/7/203 . 
    hamiltonian_BS = theta * (kron(a_dag, I)@kron(I, a) + kron(a, I)@kron(I, a_dag))
    unitary_BS = expm(-1j * hamiltonian_BS)

    # print("unitary_BS\n", np.round(unitary_BS, 4))

    pre_phase_shifter = np.kron(phase_shifter(N, phi[0]/2), phase_shifter(N, phi[1]/2))
    post_phase_shifter = np.kron(phase_shifter(N, psi[0]/2), phase_shifter(N, psi[1]/2))
    global_phase_shifter = np.kron(phase_shifter(N, lamda[0]/2), phase_shifter(N, lamda[1]/2))

    # This construction for the generalized beamsplitter is based on the description in paper DOI: 10.1088/0034-4885/66/7/203
    generalized_BS = global_phase_shifter @  (pre_phase_shifter @ unitary_BS @ post_phase_shifter)

    # print("generalized_BS\n", np.round(generalized_BS, 4))

    BS_MPO = mpo.from_dense(generalized_BS, dims = N, sites = (site1,site2), L=total_sites, tags=tag)
    # BS_MPO = BS_MPO.fill_empty_sites(mode = "full")
    return BS_MPO


def phase_shifter(N, theta):
    diag = [np.exp(1j * theta * i) for i in range(N)]
    return np.diag(diag, k=0)
