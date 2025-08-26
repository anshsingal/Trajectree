import numpy as np
from numpy import sqrt

from quimb.tensor import MatrixProductState as mps #type: ignore
from quimb.tensor import MatrixProductOperator as mpo #type: ignore
from quimb.tensor.tensor_arbgeom import tensor_network_apply_op_vec #type: ignore
from quimb.tensor.tensor_core import new_bond #type: ignore
from quimb.tensor.tensor_1d_compress import enforce_1d_like #type: ignore
from quimb.tensor.tensor_1d import TensorNetwork1DOperator #type: ignore

import qutip as qt
import re


###### SUPPORT FUNCTIONS ######

# Vacuum state creation
def fill_fn(shape):
    arr = np.zeros(shape)
    idx = tuple([0]*(len(shape)))
    arr[idx] = 1
    return arr
def create_vacuum_state(num_modes, N, bond_dim = 2):
    return mps.from_fill_fn(
                fill_fn,
                L=num_modes,
                bond_dim=bond_dim,
                phys_dim=N,
                cyclic=False,
                tags="In"
            )

def create_ladder_MPO(site, total_sites, N, tag="$Ladder$"):
    a = qt.destroy(N).full()
    a_dag = a.T
    TMSV_MPO = mpo.from_dense(a_dag, dims = N, sites = (site,), L=total_sites, tags=tag) 
    # return TMSV_MPO.fill_empty_sites(mode = "minimal")
    return TMSV_MPO

def create_MPO(site1, site2, total_sites, op, N, tag):
    MPO = mpo.from_dense(op, dims = N, sites = (site1,site2), L=total_sites, tags=tag)    
    return MPO

###### POVM OPERATORS #######


########## TMSV Operator ############




########## EXTEND MPS ###########

def extend_MPS(psi, psi_second = None):
    # print("inside extend_MPS")
    # psi_second.draw()
    # print(psi_second)
    
    psi.permute_arrays('lrp')

    # psi_second.draw()
    # print(psi_second)

    # This is supposed to be passed as the second MPS to extend the first MPS with. 
    if psi_second == None:
        psi_second = psi.copy()
    else:
        psi_second.permute_arrays('lrp')
    
    psi_num_modes = len(psi.site_tags)
    psi2_num_modes = len(psi_second.site_tags)

    psi_second.reindex({f"k{i}":f"k{i+psi_num_modes}" for i in range(psi2_num_modes)}, inplace = True)
    psi_second.retag({f"I{i}":f"I{i+psi_num_modes}" for i in range(psi2_num_modes)}, inplace = True)

    psi = psi.combine(psi_second)

    psi_last_tensor = psi.select_tensors(f"I{psi_num_modes-1}", which='any')[0]
    psi2_first_tensor = psi.select_tensors(f"I{psi_num_modes}", which='any')[0]

    new_bond(psi2_first_tensor, psi_last_tensor, axis1=0, axis2=1)

    # Simply find the tags for the input modes. 
    pattern = re.compile(r"I[0-9][0-9]*")
    tags = []
    for tag_list in [t.tags for t in psi]:
        for tag in tag_list:
            match = re.search(pattern, tag)
            if match:
                tags.append(match.string)
                break
            
    sorted_arrays = [array for array, _ in sorted( zip(psi.arrays, tags), key = lambda pair: int(pair[1][1:]) )]

    psi = mps(sorted_arrays)
    return psi


def calc_fidelity_swapping(state, reference_state, N, error_tolerance):
    reference_mps = create_bimode_bell_state(reference_state, N)
    projector_mpo = outer_product_mps(reference_mps)

    projector_mpo.reindex({"k0":"k0","k1":"k1","k2":"k4","k3":"k5"}, inplace = True)
    projector_mpo.reindex({"b0":"b0","b1":"b1","b2":"b4","b3":"b5"}, inplace = True)
    projector_mpo.retag({"I0":"I0","I1":"I1","I2":"I4","I3":"I5"}, inplace = True)

    # print("sites present in projector_mpo:", projector_mpo.sites)
    enforce_1d_like(projector_mpo, site_tags=state.site_tags, inplace=True)

    state = tensor_network_apply_op_vec(projector_mpo, state, compress=True, contract = True, cutoff = error_tolerance)
    # state.draw()
    return state.norm()**2

    
    
    # Calculate and return fidelity of the projected state. 


def create_bimode_bell_state(bell_state, N, error_tolerance = 1e-12):
    I = np.eye(N)

    a_dag = qt.create(N).full()
    a = qt.destroy(N).full()

    vacuum_state = np.zeros((N,1))
    vacuum_state[0] = 1
    vac_projector = np.outer(vacuum_state, vacuum_state)

    one_state = a_dag @ vacuum_state # For now, we're defining the 1 state as having only one photon. This could be changed to have any number of non-zero photons.
    # print("one_state:", one_state)   # This is because the ideal case is having exactly one photon for the 1 state. 
    one_projector = np.outer(one_state, one_state)                                 

    NOT_gate = vacuum_state @ one_state.conj().T + one_state @ vacuum_state.conj().T
    H_gate = (1/sqrt(2)) * ((vacuum_state - one_state) @ one_state.conj().T + (vacuum_state + one_state) @ vacuum_state.conj().T)
    C_NOT_close = np.kron(vac_projector, I) + np.kron(one_projector, NOT_gate)
    C_NOT_open = np.kron(one_projector, I) + np.kron(vac_projector, NOT_gate)

    NOT_MPO_0 = mpo.from_dense(NOT_gate, dims = N, sites = (0,), L=4, tags="a_dag")
    NOT_MPO_1 = mpo.from_dense(NOT_gate, dims = N, sites = (1,), L=4, tags="a_dag")
    H_MPO = mpo.from_dense(H_gate, dims = N, sites = (0,), L=4, tags="H")
    C_NOT_close_MPO_1 = mpo.from_dense(C_NOT_close, dims = N, sites = (0,1), L=4, tags="C_NOT_close_1")
    C_NOT_close_MPO_2 = mpo.from_dense(C_NOT_close, dims = N, sites = (1,2), L=4, tags="C_NOT_close_2")
    C_NOT_open_MPO = mpo.from_dense(C_NOT_open, dims = N, sites = (2,3), L=4, tags="C_create_open")
    
    vacuum = create_vacuum_state(4, N, bond_dim = 2)

    if bell_state == "psi_minus":
        psi = tensor_network_apply_op_vec(NOT_MPO_0, vacuum, compress=True, contract = True, cutoff = error_tolerance)
        psi = tensor_network_apply_op_vec(NOT_MPO_1, psi, compress=True, contract = True, cutoff = error_tolerance)
    elif bell_state == "psi_plus":
        psi = tensor_network_apply_op_vec(NOT_MPO_1, vacuum, compress=True, contract = True, cutoff = error_tolerance)
    elif bell_state == "phi_plus":
        psi = vacuum
    elif bell_state == "phi_minus":
        psi = tensor_network_apply_op_vec(NOT_MPO_0, vacuum, compress=True, contract = True, cutoff = error_tolerance)

    
    psi = tensor_network_apply_op_vec(H_MPO, psi, compress=True, contract = True, cutoff = error_tolerance)
    # read_quantum_state(psi, N, num_states = 2)
    psi = tensor_network_apply_op_vec(C_NOT_close_MPO_1, psi, compress=True, contract = True, cutoff = error_tolerance)
    # read_quantum_state(psi, N, num_states = 2)
    psi = tensor_network_apply_op_vec(C_NOT_close_MPO_2, psi, compress=True, contract = True, cutoff = error_tolerance)
    psi = tensor_network_apply_op_vec(C_NOT_open_MPO, psi, compress=True, contract = True, cutoff = error_tolerance)
    
    return psi


def outer_product_mps(psi):
    psi_H = psi.H
    psi_H.retag_({'In': 'Out'})
    psi_H.site_ind_id = 'b{}'
    rho = (psi_H | psi)
    for i in range(rho.L):
        rho ^= f"I{i}"   
    rho = TensorNetwork1DOperator(rho)
    rho._upper_ind_id = psi.site_ind_id
    rho._lower_ind_id = psi_H.site_ind_id
    rho = rho.fuse_multibonds()
    rho_MPO = rho.view_as_(mpo, cyclic = False, L = 8) # L is important. Its hard coded now, but must be configutrable based on the input state. 
    return rho_MPO