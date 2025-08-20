from .utils import create_MPO
from .devices import create_BS_MPO

from scipy import sparse as sp
from scipy.linalg import expm

import numpy as np
from numpy.linalg import matrix_power
from numpy import kron, sqrt

from quimb.tensor.tensor_arbgeom import tensor_network_apply_op_vec #type: ignore
from quimb.tensor.tensor_1d_compress import enforce_1d_like #type: ignore

import qutip as qt
from math import factorial


def create_TMSV_OP_Dense(N, mean_photon_num):
    a = qt.destroy(N).full()
    a_dag = a.T
    truncation = (N-1)   

    op = expm(1j * mean_photon_num * (kron(a_dag, a_dag) + kron(a, a)))

    return op



########## Light Source ###########

def light_source(vacuum, N, mean_photon_num, num_modes, error_tolerance, TMSV_indices = ((0,2),(5,7)), compress = True, contract = True):

    psi = vacuum.copy()
    psi.add_tag("L0")
    site_tags = psi.site_tags

    # Creating TMSV ops:
    TMSV_op_dense = create_TMSV_OP_Dense(N, mean_photon_num)

    TMSV_MPO_H = create_MPO(site1 = TMSV_indices[0][0], site2 = TMSV_indices[0][1], total_sites = num_modes, op = TMSV_op_dense, N = N, tag = r"$TMSV_H$")
    # TMSV_MPO_H.draw()
    # print("sites present in light_source:", TMSV_MPO_H.sites)
    enforce_1d_like(TMSV_MPO_H, site_tags=site_tags, inplace=True)
    # print("sites present in light_source:", TMSV_MPO_H.sites)
    TMSV_MPO_H.add_tag("L1")

    TMSV_MPO_V = create_MPO(site1 = TMSV_indices[1][0], site2 = TMSV_indices[1][1], total_sites = num_modes, op = TMSV_op_dense, N = N, tag = r"$TMSV_V$")
    enforce_1d_like(TMSV_MPO_V, site_tags=site_tags, inplace=True)
    TMSV_MPO_V.add_tag("L1")

    # Creating PBS ops:
    U_PBS_H_Signal = create_BS_MPO(site1 = 2, site2 = 6, theta=np.pi/2, total_sites = num_modes, N = N, tag = r"$PBS_S$")
    enforce_1d_like(U_PBS_H_Signal, site_tags=site_tags, inplace=True)
    U_PBS_H_Signal.add_tag("L1")

    U_PBS_H_Idler = create_BS_MPO(site1 = 0, site2 = 4, theta=np.pi/2, total_sites = num_modes, N = N, tag = r"$PBS_I$")
    enforce_1d_like(U_PBS_H_Idler, site_tags=site_tags, inplace=True)
    U_PBS_H_Signal.add_tag("L1")

    # Create entangled state:
    psi = tensor_network_apply_op_vec(TMSV_MPO_H, psi, compress=compress, contract = contract, cutoff = error_tolerance)
    psi = tensor_network_apply_op_vec(TMSV_MPO_V, psi, compress=compress, contract = contract, cutoff = error_tolerance)
    psi = tensor_network_apply_op_vec(U_PBS_H_Idler, psi, compress=compress, contract = contract, cutoff = error_tolerance)
    psi = tensor_network_apply_op_vec(U_PBS_H_Signal, psi, compress=compress, contract = contract, cutoff = error_tolerance)

    psi.normalize()

    # print("trace is:", np.linalg.norm(psi.to_dense()))

    for _ in range(4):
        psi.measure(0, remove = True, renorm = True, inplace = True)

    # Not used for TN implermentation. Used for validating impelmentation with dense version
    TMSV_state = psi.to_dense()
    TMSV_state = np.reshape(TMSV_state.data, (-1, 1), order = 'C')
    TMSV_state = sp.csr_matrix(TMSV_state)
    TMSV_state.data = np.round(TMSV_state.data, 10)
    TMSV_state.eliminate_zeros()

    return psi, TMSV_state


# Generate truncation filter MPO 
# TODO: Make a function to renormalize a quantum state. How: find the projection of the quantum state onto itself and calculate the 
# probability. Next, take the square root of this number, divide it by the number nodes in the quantum state and multiply it with 
# all the states in the MPS. For density matrices, simply find the trace directly and do the same thing as the previous example except
# for not taking the square root.  The truncation filter would not work without the renormalization 
def create_truncation_filter_Dense(truncation):
    # This is only the projection operator. The states need to be normalized first. 
    N = truncation+1
    vacuum = np.zeros(N**2)
    vacuum[0] = 1

    a = qt.destroy(N).full()
    a_dag = a.T
    I = np.eye(N)

    # # debug
    # labels = generate_labels(1,N)

    op = 0
    for trunc in range(truncation, -1, -1):
        state = kron(matrix_power(a_dag, trunc), I) @ vacuum / sqrt(factorial(trunc) * factorial(0))
        op+=np.outer(state, state)
        coeffs = [trunc+1, 0]

        # # Debug
        # state_inds = state.nonzero()[0]
        # print("TMSV state:", [labels[i] for i in state_inds], "Val:", state[state_inds[0]])
        # print("coeffs", coeffs)

        for i in range(trunc):
            coeffs = [coeffs[0]-1, coeffs[1]+1]
            state = kron(a, a_dag) @ state / sqrt((coeffs[0]) * (coeffs[1]))
            op += np.outer(state, state)


            # # debug
            # state_inds = state.nonzero()[0]
            # print("TMSV state:", [labels[i] for i in state_inds], "Val:", state[state_inds[0]])
            # print("coeffs", coeffs)

    return op
