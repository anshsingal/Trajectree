from scipy.linalg import expm

import numpy as np
from numpy import kron

from quimb.tensor import MatrixProductOperator as mpo #type: ignore
from functools import lru_cache

import qutip as qt

# Beamsplitter transformation
def create_BS_MPO(site1, site2, theta, total_sites, N, return_unitary = False, tag = 'BS'): 
    
    a = qt.destroy(N).full()
    a_dag = a.T
    I = np.eye(N)
    
    # This corresponds to the BS hamiltonian:

    hamiltonian_BS = -theta * ( kron(I, a_dag)@kron(a, I) - kron(I, a)@kron(a_dag, I) )
    unitary_BS = expm(hamiltonian_BS)

    if return_unitary: 
        return unitary_BS

    # print("unitary_BS", unitary_BS)

    BS_MPO = mpo.from_dense(unitary_BS, dims = N, sites = (site1,site2), L=total_sites, tags=tag)
    # BS_MPO = BS_MPO.fill_empty_sites(mode = "full")
    return BS_MPO


# def generalized_mode_mixer(site1, site2, theta, phi, psi, lamda, total_sites, N, tag = 'MM'): 
#     """
#     Deprticated, do not use!
#     """

#     a = qt.destroy(N).full()
#     a_dag = a.T
#     I = np.eye(N)
    
#     # This corresponds to the BS hamiltonian: This is a different difinition from the one in 
#     # create_BS_MPO. This is because of how the generalized beamsplitter is defined in DOI: 10.1088/0034-4885/66/7/203 . 
#     hamiltonian_BS = theta * (kron(a_dag, I)@kron(I, a) + kron(a, I)@kron(I, a_dag))
#     unitary_BS = expm(-1j * hamiltonian_BS)

#     # print("unitary_BS\n", np.round(unitary_BS, 4))

#     pre_phase_shifter = np.kron(phase_shifter(N, phi[0]/2), phase_shifter(N, phi[1]/2))
#     post_phase_shifter = np.kron(phase_shifter(N, psi[0]/2), phase_shifter(N, psi[1]/2))
#     global_phase_shifter = np.kron(phase_shifter(N, lamda[0]/2), phase_shifter(N, lamda[1]/2))

#     # This construction for the generalized beamsplitter is based on the description in paper DOI: 10.1088/0034-4885/66/7/203
#     generalized_BS = global_phase_shifter @  (pre_phase_shifter @ unitary_BS @ post_phase_shifter)

#     # print("generalized_BS\n", np.round(generalized_BS, 4))

#     BS_MPO = mpo.from_dense(generalized_BS, dims = N, sites = (site1,site2), L=total_sites, tags=tag)
#     # BS_MPO = BS_MPO.fill_empty_sites(mode = "full")
#     return BS_MPO


# def phase_shifter(N, theta):
#     """
#     Depricated, do not use!
#     """
#     diag = [np.exp(1j * theta * i) for i in range(N)]
#     return np.diag(diag, k=0)


def rx(omega, N, return_unitary = False, site1 = None, site2 = None, total_sites = None, tag = 'rx'):
    L_t, L_x, L_y, L_z = generate_angular_momentum_operators(N)
    unitary_rx = expm(-1j * omega * L_x)
    if return_unitary:
        return unitary_rx
    BS_MPO = mpo.from_dense(unitary_rx, dims = N, sites = (site1,site2), L=total_sites, tags=tag)
    return BS_MPO

def ry(theta, N, return_unitary = False, site1 = None, site2 = None, total_sites = None, tag = 'ry'):
    L_t, L_x, L_y, L_z = generate_angular_momentum_operators(N)
    unitary_ry = expm(-1j * theta * L_y)
    if return_unitary:
        return unitary_ry
    BS_MPO = mpo.from_dense(unitary_ry, dims = N, sites = (site1,site2), L=total_sites, tags=tag)
    return BS_MPO

def rz(phi, N, return_unitary = False, site1 = None, site2 = None, total_sites = None, tag = 'rz'):
    L_t, L_x, L_y, L_z = generate_angular_momentum_operators(N)
    unitary_rz = expm(-1j * phi * L_z)
    if return_unitary:
        return unitary_rz
    BS_MPO = mpo.from_dense(unitary_rz, dims = N, sites = (site1,site2), L=total_sites, tags=tag)
    return BS_MPO

def global_phase(lamda, N, return_unitary = False, site1 = None, site2 = None, total_sites = None, tag = 'global_phase'):
    L_t, L_x, L_y, L_z = generate_angular_momentum_operators(N)
    unitary_global_phase = expm(-1j * lamda * L_t)
    if return_unitary:
        return unitary_global_phase
    BS_MPO = mpo.from_dense(unitary_global_phase, dims = N, sites = (site1,site2), L=total_sites, tags=tag)
    return BS_MPO

def single_mode_phase(lamda, N):
    a = qt.destroy(N).full()
    a_dag = a.T
    I = np.eye(N)

    unitary_phase = expm(-1j * (lamda/2) * a_dag @ a)
    return unitary_phase

@lru_cache(maxsize=32)
def generate_angular_momentum_operators(N):
    """
    This generates the angular momentum operators for polarization two polarization modes using the Jordan-Schwinger representation. 
    See sec. 4.2 in DOI: 10.1088/0034-4885/66/7/203
    """
    a = qt.destroy(N).full()
    a_dag = a.T
    I = np.eye(N)

    a1 = kron(a, I)
    a2 = kron(I, a)
    a1_dag = a1.T
    a2_dag = a2.T
    
    L_t = 0.5 * (a1_dag @ a1 + a2_dag @ a2)
    L_x = 0.5 * (a1_dag @ a2 + a2_dag @ a1)
    L_y = 0.5j * (a2_dag @ a1 - a1_dag @ a2)
    L_z = 0.5 * (a1_dag @ a1 - a2_dag @ a2)

    return L_t, L_x, L_y, L_z

def generalized_mode_mixer(theta, phi, psi, lamda, N, tag = 'MM'): 
    unitary_BS = rz(phi, N, return_unitary=True) @ ry(theta, N, return_unitary=True) @ rz(psi, N, return_unitary=True) @ global_phase(lamda, N, return_unitary=True)
    return unitary_BS