from trajectree.fock_optics.measurement import *
from trajectree.fock_optics.utils import *
from trajectree.fock_optics.light_sources import *
from trajectree.fock_optics.devices import *
from trajectree.trajectory import *

from trajectree.protocols.swap import perform_swapping_simulation

import numpy as np

def quantum_encoder(mean_photon_num, N, psi_control, control_indices, error_tolerance):
    """This function performs the quantum encoder, basically "copy" one state into two modes. Obviously, you aren't copying shit. You are simply entangling another 
    state (modes (0,1) or mode b in the paper) with the control state, creating a "copy". The bell state required to make the copy is added before the control MPS 
    ((0,1):(H,V){b,d} and (2,3):(H,V){a,c}). {a,c} modes are measured out at the end of the function and hence, only 2 modes are pre-added to the retured MPS.   
    
    """
    # Entangled state from EPS
    vacuum = create_vacuum_state(num_modes=8, N=N)
    bell_state, _ = light_source(vacuum, N, mean_photon_num, 8, error_tolerance, compress=True, contract=True)

    # psi_control.draw()
    # print(psi_control)

    psi = extend_MPS(bell_state, psi_control)

    # psi.draw()
    # print(psi)

    # PBS op: (The V mode is transmitted and not reflected)
    U_PBS_V = create_BS_MPO(site1 = 3, site2 = bell_state.L+control_indices[1], theta=np.pi/2, total_sites = psi.L, N = N, tag = r"$PBS$")
    enforce_1d_like(U_PBS_V, site_tags=psi.site_tags, inplace=True)
    psi = tensor_network_apply_op_vec(U_PBS_V, psi, compress=True, contract = True, cutoff = error_tolerance)

    # Measuring D_d
    # This is meant to change the basis from HV -> FS: (See https://doi.org/10.1103/PhysRevA.64.062311)
    U_PBS_FS = create_BS_MPO(site1 = 2, site2 = 3, theta=np.pi/4, total_sites = psi.L, N = N, tag = r"$rotator$")
    enforce_1d_like(U_PBS_FS, site_tags=psi.site_tags, inplace=True)
    psi = tensor_network_apply_op_vec(U_PBS_FS, psi, compress=True, contract = True, cutoff = error_tolerance)

    # Performing measurements:
    BSM_POVM_1_OPs = generate_sqrt_POVM_MPO(sites=[2], outcome = 1, total_sites=psi.L, efficiency=1, N=N, pnr = True)
    BSM_POVM_1_OPs.extend(generate_sqrt_POVM_MPO(sites=[3], outcome = 0, total_sites=psi.L, efficiency=1, N=N, pnr = True))

    for POVM_OP in BSM_POVM_1_OPs:
        psi = tensor_network_apply_op_vec(POVM_OP, psi, compress=True, contract = True, cutoff = error_tolerance)

    return psi


def destructive_CNOT(control_b_sites, target_sites, psi, N, error_tolerance):

    # print("control_b_sites:", control_b_sites)
    # print("target_sites", target_sites)

    # Rotaing bases of encoded control's b mode and the target mode:
    U_rotator_FS = create_BS_MPO(site1 = control_b_sites[0], site2 = control_b_sites[1], theta=np.pi/4, total_sites = psi.L, N = N, tag = r"$rotator$")
    enforce_1d_like(U_rotator_FS, site_tags=psi.site_tags, inplace=True)
    psi = tensor_network_apply_op_vec(U_rotator_FS, psi, compress=True, contract = True, cutoff = error_tolerance)

    U_rotator_FS = create_BS_MPO(site1 = target_sites[0], site2 = target_sites[1], theta=np.pi/4, total_sites = psi.L, N = N, tag = r"$rotator$")
    enforce_1d_like(U_rotator_FS, site_tags=psi.site_tags, inplace=True)
    psi = tensor_network_apply_op_vec(U_rotator_FS, psi, compress=True, contract = True, cutoff = error_tolerance)

    # Applying PBS in rotated basis (only the V modes are reflected and the H modes are transmitted. Hence, the V modes undergo pi/2 rotation and the H modes undergo no rotation):
    
    # Implementation using SWAP operator. This does not generalize to higher truncations. 
    # SWAP = qt.qip.operations.swap().full()
    # U_PBS_F = create_MPO(1, 3, psi.L, SWAP, N, r"$PBS$")
    U_PBS_F = create_BS_MPO(site1 = target_sites[1], site2 = control_b_sites[1], theta=np.pi/2, total_sites = psi.L, N = N, tag = r"$PBS$")
    enforce_1d_like(U_PBS_F, site_tags=psi.site_tags, inplace=True)
    psi = tensor_network_apply_op_vec(U_PBS_F, psi, compress=True, contract = True, cutoff = error_tolerance)

    # Undoing rotations:
    U_inverse_rotator_FS = create_BS_MPO(site1 = target_sites[0], site2 = target_sites[1], theta=-np.pi/4, total_sites = psi.L, N = N, tag = r"$rotator$")
    enforce_1d_like(U_inverse_rotator_FS, site_tags=psi.site_tags, inplace=True)
    psi = tensor_network_apply_op_vec(U_inverse_rotator_FS, psi, compress=True, contract = True, cutoff = error_tolerance)

    U_inverse_rotator_FS = create_BS_MPO(site1 = control_b_sites[0], site2 = control_b_sites[1], theta=-np.pi/4, total_sites = psi.L, N = N, tag = r"$rotator$")
    enforce_1d_like(U_inverse_rotator_FS, site_tags=psi.site_tags, inplace=True)
    psi = tensor_network_apply_op_vec(U_inverse_rotator_FS, psi, compress=True, contract = True, cutoff = error_tolerance)

    # Measuring the b mode (name d after the PBS)
    BSM_POVM_1_OPs = generate_sqrt_POVM_MPO(sites=[control_b_sites[0]], outcome = 1, total_sites=psi.L, efficiency=1, N=N, pnr = True)
    BSM_POVM_1_OPs.extend(generate_sqrt_POVM_MPO(sites=[control_b_sites[1]], outcome = 0, total_sites=psi.L, efficiency=1, N=N, pnr = True))

    for POVM_OP in BSM_POVM_1_OPs:
        psi = tensor_network_apply_op_vec(POVM_OP, psi, compress=True, contract = True, cutoff = error_tolerance)

    return psi

def CNOT(psi_control_modes, psi_target_modes, psi_control, psi_target, N, mean_photon_num, error_tolerance):
    """Pass psi_target as None if the same MPS has both the control and target modes.
    Args:
        psi_control_modes (list): List of control modes (H,V).
        psi_target_modes (list): List of target modes (H,V).
        psi_control (MPS): MPS for the control modes.
        psi_target (MPS): MPS for the target modes, can be None if the same MPS as target is used.
        N (int): Number of photons.
        mean_photon_num (float): Mean photon number for the EPS used in implemeting the CNOT gate.
        error_tolerance (float): Tolerance for numerical errors in tensor network operations.
    Returns:
        MPS: The resulting MPS after applying the CNOT operation.
    """

    psi_encoded_control = quantum_encoder(mean_photon_num, N, psi_control, psi_control_modes, error_tolerance)

    # read_quantum_state(psi_encoded_control,  N, num_states = 6)

    if not psi_target == None:
        psi = extend_MPS(psi_target, psi_encoded_control)
        psi_control_b_modes = [psi_target.L, psi_target.L+1] # [site + psi_target.L for site in psi_control_modes]
    
    else: 
        psi_control_b_modes = [0,1]
        psi_target_modes = [4+site for site in psi_target_modes] # We add 4 since the additional modes from the EPS are pre-added to the MPS. 
        psi = psi_encoded_control

    psi = destructive_CNOT(psi_control_b_modes, psi_target_modes, psi, N, error_tolerance)

    # read_quantum_state(psi, N, num_states = 6)

    norm = psi.normalize()
    for _ in range(4):
        psi.measure(0, remove = True, renorm = True, inplace = True)
    psi[-1].modify(data=psi[-1].data * norm**0.5)

    return psi


def H(psi, sites, N, error_tolerance):
    # TODO: This function does not work for N > 2.
    # This definition is based on the paper: https://arxiv.org/pdf/quant-ph/9706022
    H = generalized_mode_mixer(sites[0], sites[1], -np.pi/4, [0,-np.pi], [0,-np.pi], [0,0], psi.L, N)
    # H = generalized_mode_mixer(0, 1, np.pi/4, 0, 0, 0, 2, N)
    enforce_1d_like(H, site_tags=psi.site_tags, inplace=True)
    psi = tensor_network_apply_op_vec(H, psi, compress=True, contract = True, cutoff = error_tolerance)
    return psi
