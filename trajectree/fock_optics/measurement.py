from .devices import generalized_mode_mixer, create_BS_MPO, ry
from ..trajectory import quantum_channel
from .noise_models import single_mode_bosonic_noise_channels, general_mixed_bs_noise_model, depolarizing_operators

from scipy.linalg import sqrtm
from scipy import sparse as sp

import numpy as np
from numpy.linalg import matrix_power
from numpy import sqrt

from quimb.tensor import MatrixProductOperator as mpo #type: ignore
from quimb.tensor.tensor_arbgeom import tensor_network_apply_op_vec #type: ignore
from quimb.tensor.tensor_1d_compress import enforce_1d_like #type: ignore

import qutip as qt
from math import factorial

from functools import lru_cache


# This is the actual function that generates the POVM operator.
def create_threshold_POVM_OP_Dense(efficiency, outcome, N):
    a = qt.destroy(N).full()
    a_dag = a.T
    create0 = a_dag * sqrt(efficiency)
    destroy0 = a * sqrt(efficiency)
    series_elem_list = [((-1)**i) * matrix_power(create0, (i+1)) @ matrix_power(destroy0, (i+1)) / factorial(i+1) for i in range(N-1)] # (-1)^i * a_dag^(i+1) @ a^(i+1) / (i+1)! = (-1)^(i+2) * a_dag^(i+1) @ a^(i+1) / (i+1)! since goes from 0->n
    # print(series_elem_list[0])
    dense_op = sum(series_elem_list)

    if outcome == 0:
        dense_op = np.eye(dense_op.shape[0]) - dense_op
    # print(sqrtm(dense_op))
    return dense_op

@lru_cache(maxsize=20)
def factorial(x):
    n = 1
    for i in range(2, x+1):
        n *= i
    return n

@lru_cache(maxsize=20)
def comb(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

@lru_cache(maxsize=20)
def projector(n, N):
    state = np.zeros(N)
    state[n] = 1
    return np.outer(state, state)

# Testing stuff out here. 
def create_PNR_POVM_OP_Dense(eff, outcome, N, debug = False):
    a_dag = qt.create(N).full()
    vacuum = np.zeros(N)
    vacuum[0] = 1

    @lru_cache(maxsize=20)
    def create_povm_list(eff, N):
        povms = []
        # m is the outcome here
        for m in range(N-1):
            op = 0
            for n in range(m, N):
                op += comb(n,m) * eff**m * (1-eff)**(n-m) * projector(n, N)
            povms.append(op)

        povms.append(np.eye(N) - sum(povms))
        return povms
    
    povms = create_povm_list(eff, N)
    if debug:
        return povms[outcome], povms
    return povms[outcome]



def generate_sqrt_POVM_MPO(sites, outcome, total_sites, efficiency, N, pnr = False, tag = "POVM"):
    if pnr:
        # dense_op = sqrtm(create_PNR_POVM_OP_Dense(efficiency, outcome, N)).astype(np.complex128)
        dense_op = (create_PNR_POVM_OP_Dense(efficiency, outcome, N)).astype(np.complex128)
    else:
        # dense_op = sqrtm(create_threshold_POVM_OP_Dense(efficiency, outcome, N)).astype(np.complex128)
        dense_op = (create_threshold_POVM_OP_Dense(efficiency, outcome, N)).astype(np.complex128)

    sqrt_POVM_MPOs = []
    for i in sites:
        sqrt_POVM_MPOs.append(mpo.from_dense(dense_op, dims = N, sites = (i,), L=total_sites, tags=tag))

    return sqrt_POVM_MPOs


def bell_state_measurement(psi, N, site_tags, num_modes, efficiencies, dark_counts,  error_tolerance, beamsplitters = [[2,6],[3,7]], measurements = {0:(2,7), 1:(3,6)}, depolarizing_error = False, damping_error = True, pnr = False, det_outcome = 1, use_trajectory = False, return_MPOs = False, compress = True, contract = True):

    """Perform Bell state measrement or return the MPOs used in the measurement.
    Args:
        psi (mps): The input state to be measured.
        N (int): local Hilbert space dimension
        site_tags (list): The tags for the sites in the MPS.
        num_modes (int): The number of modes in the MPS.
        efficiencies list[float]: The efficiencies of the (pairs of) detectors in the BSM.
        error_tolerance (float): The error tolerance for the tensor network.
        measurements (dict): The sites for the measurements. Default is {1:(2,7), 0:(3,6)}.
        pnr (bool): Whether to use photon number resolving measurement. Default is False.
        pnr_outcome (int): The outcome for the photon number resolving measurement. Default is 1. When not using PNR, this can be anything other than 1 since threshold detectors don't distinguish between photon numbers. 
        return_MPOs (bool): Whether to return the MPOs used in the measurement. Default is False.
        compress (bool): Whether to compress the MPS after applying the MPOs. Default is True.
        contract (bool): Whether to contract the MPS after applying the MPOs. Default is True.
        
        Returns:
            mps: The measured state after the Bell state measurement.
            
    """

    U_BS_H = create_BS_MPO(site1 = beamsplitters[0][0], site2 = beamsplitters[0][1], theta=np.pi/4, total_sites = num_modes, N = N, tag = r"$U_{BS_H}$")
    enforce_1d_like(U_BS_H, site_tags=site_tags, inplace=True)
    U_BS_H.add_tag("L2")

    U_BS_V = create_BS_MPO(site1 = beamsplitters[1][0], site2 = beamsplitters[1][1], theta=np.pi/4, total_sites = num_modes, N = N, tag = r"$U_{BS_V}$")
    enforce_1d_like(U_BS_V, site_tags=site_tags, inplace=True)
    U_BS_V.add_tag("L3")

    # Note that these are not used if using trajectree to implement detector inefficiency. 
    BSM_POVM_1_OPs = generate_sqrt_POVM_MPO(sites=measurements[1], outcome = det_outcome, total_sites=num_modes, efficiency=efficiencies[0], N=N, pnr = pnr)
    BSM_POVM_1_OPs.extend(generate_sqrt_POVM_MPO(sites=measurements[0], outcome = 0, total_sites=num_modes, efficiency=efficiencies[1], N=N, pnr = pnr))

    if return_MPOs:
        returned_MPOs = [U_BS_H, U_BS_V]
        if use_trajectory:
            quantum_channel_list = [quantum_channel(N = N, num_modes = num_modes, formalism = "closed", unitary_op = BSM_MPO, name = "beam splitter") for BSM_MPO in returned_MPOs]
            
            if damping_error:
                damping_and_loss_channel0 = general_mixed_bs_noise_model(dark_count_rate = dark_counts[0], eta = efficiencies[0], N = N)
                damping_and_loss_channel1 = general_mixed_bs_noise_model(dark_count_rate = dark_counts[1], eta = efficiencies[1], N = N)
                # two_mode_kraus_ops_0 = [sp.kron(op1, op2) for op1 in damping_and_loss_channel0 for op2 in damping_and_loss_channel0]
                # two_mode_kraus_ops_1 = [sp.kron(op1, op2) for op1 in damping_and_loss_channel1 for op2 in damping_and_loss_channel1]
                quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((2,), damping_and_loss_channel0), name = "BSM detector")) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.
                quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((3,), damping_and_loss_channel0), name = "BSM detector")) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.
                quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((6,), damping_and_loss_channel1), name = "BSM detector")) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.
                quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((7,), damping_and_loss_channel1), name = "BSM detector")) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.

            if depolarizing_error != 0:
                depolarizing_channels = depolarizing_operators(depolarizing_probability = depolarizing_error, N = N)
                quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((2,3), depolarizing_channels), name = "BSM depol"))
                quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((6,7), depolarizing_channels), name = "BSM depol"))

            BSM_POVM_1_OPs = generate_sqrt_POVM_MPO(sites=measurements[1], outcome = det_outcome, total_sites=num_modes, efficiency=1, N=N, pnr = pnr)
            BSM_POVM_1_OPs.extend(generate_sqrt_POVM_MPO(sites=measurements[0], outcome = 0, total_sites=num_modes, efficiency=1, N=N, pnr = pnr))

            expectation_ops = [quantum_channel(N = N, num_modes = num_modes, formalism = "closed", unitary_op = DET_MPO, name = "Det POVM") for DET_MPO in BSM_POVM_1_OPs]
            # quantum_channel_list.extend(expectation_ops)
    
            return quantum_channel_list, expectation_ops 

        returned_MPOs.extend(BSM_POVM_1_OPs) # Collect all the MPOs in a list and return them. The operators are ordered as such: 

        quantum_channel_list = [quantum_channel(N = N, num_modes = num_modes, formalism = "closed", unitary_op = BSM_MPO, name = "BSM") for BSM_MPO in returned_MPOs]

        return quantum_channel_list

    psi = tensor_network_apply_op_vec(U_BS_H, psi, compress=compress, contract = contract, cutoff = error_tolerance)
    psi = tensor_network_apply_op_vec(U_BS_V, psi, compress=compress, contract = contract, cutoff = error_tolerance)

    for POVM_OP in BSM_POVM_1_OPs:
        POVM_OP.add_tag("L4")
        psi = tensor_network_apply_op_vec(POVM_OP, psi, compress=compress, contract = contract, cutoff = error_tolerance)

    return psi



def rotate_and_measure(psi, N, site_tags, num_modes, efficiency, error_tolerance, idler_angles, signal_angles, dark_counts = [3e-5,3e-5], rotations = {"signal":(4,5), "idler":(0,1)}, measurements = {1:(0,4), 0:(1,5)}, depolarizing_error = False, damping_error = True, pnr = False, det_outcome = 1, return_MPOs = False, return_quantum_channel = False, compress = True, contract = True, draw = False):
    # idler_angles = [0]
    # angles = [np.pi/4]

    # A dict is used when the user wants to rotate around multiple axes of the bloch sphere. Otherwise, if its a list, then 
    # we assume all measurements around the y axis.
    # if type(idler_angles) != dict:
    #     idler_angles = {"theta": idler_angles, "phi":[0]*len(idler_angles), "psi":[0]*len(idler_angles)}
    # if type(signal_angles) != dict:
    #     signal_angles = {"theta": signal_angles, "phi":[0]*len(signal_angles), "psi":[0]*len(signal_angles)}
    

    coincidence = []

    POVM_1_OPs = generate_sqrt_POVM_MPO(sites = measurements[1], outcome = det_outcome, total_sites=num_modes, efficiency=efficiency, N=N, pnr = pnr)
    POVM_0_OPs = generate_sqrt_POVM_MPO(sites = measurements[0], outcome = 0, total_sites=num_modes, efficiency=efficiency, N=N, pnr = pnr)

    meas_ops = POVM_1_OPs
    meas_ops.extend(POVM_0_OPs)

    for i in range(len(idler_angles)):
        coincidence_probs = []

        # rotator_node_1 = generalized_mode_mixer(site1 = rotations["idler"][0], site2 = rotations["idler"][1], theta = idler_angles[i], phi = 0, psi = 0, lamda = 0, total_sites = num_modes, N = N, tag = 'MM')
        rotator_node_1 = ry(idler_angles[i], N, site1 = rotations["idler"][0], site2 = rotations["idler"][1], total_sites = num_modes, tag = 'ry')
        
        enforce_1d_like(rotator_node_1, site_tags=site_tags, inplace=True)
        rotator_node_1.add_tag("L5")
        if not return_MPOs and not return_quantum_channel: # If the user wants the MPOs, we don't need to apply the rotator to the state.
            idler_rotated_psi = tensor_network_apply_op_vec(rotator_node_1, psi, compress=compress, contract = contract, cutoff = error_tolerance)


        for j in range(len(signal_angles)):
            # print("idler:", i, "signal:", j)
        
            # rotator_node_2 = create_BS_MPO(site1 = rotations["signal"][0], site2 = rotations["signal"][1], theta=angle, total_sites = num_modes, N = N, tag = r"$Rotator_S$")
            ##########################
            # We make this correction here since the rotator hamiltonian is 1/2(a_v b_h + a_h b_v), which does not show up in the bs unitary, whose function we are reusing to 
            # rotate the state.
            # rotator_node_2 = generalized_mode_mixer(site1 = rotations["signal"][0], site2 = rotations["signal"][1], theta = signal_angles[j], phi = 0, psi = 0, lamda = 0, total_sites = num_modes, N = N, tag = 'MM') 
            rotator_node_2 = ry(signal_angles[j], N, site1 = rotations["signal"][0], site2 = rotations["signal"][1], total_sites = num_modes, tag = 'ry')
            # print("checling node2 unitarity:", sp.csr_array(np.round(rotator_node_2.to_dense() @ rotator_node_2.to_dense().T.conj() - np.eye(N**2), 5)))
            
            enforce_1d_like(rotator_node_2, site_tags=site_tags, inplace=True)

            if return_MPOs:
                meas_ops = [rotator_node_1, rotator_node_2] + meas_ops # Collect all the MPOs in a list and return them
                return meas_ops

            if return_quantum_channel:
                quantum_channel_list = []
                quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "closed", unitary_op = rotator_node_1, name = "Idler Rotator"))
                quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "closed", unitary_op = rotator_node_2, name = "Signal Rotator"))

                if damping_error:
                    damping_and_loss_channel0 = general_mixed_bs_noise_model(dark_count_rate = dark_counts[0], eta = efficiency, N = N)
                    damping_and_loss_channel1 = general_mixed_bs_noise_model(dark_count_rate = dark_counts[1], eta = efficiency, N = N)
                    # two_mode_kraus_ops_0 = [sp.kron(op1, op2) for op1 in damping_and_loss_channel0 for op2 in damping_and_loss_channel0]
                    # two_mode_kraus_ops_1 = [sp.kron(op1, op2) for op1 in damping_and_loss_channel1 for op2 in damping_and_loss_channel1]
                    quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((rotations["idler"][0],), damping_and_loss_channel0), name = "PA detector")) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.
                    quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((rotations["idler"][1],), damping_and_loss_channel0), name = "PA detector")) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.
                    quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((rotations["signal"][0],), damping_and_loss_channel1), name = "PA detector")) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.
                    quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((rotations["signal"][1],), damping_and_loss_channel1), name = "PA detector")) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.

                if depolarizing_error:
                    depolarizing_channels = depolarizing_operators(depolarizing_probability = 0.5, N = N)
                    quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((0,1), depolarizing_channels), name = "BSM depol"))
                    quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((4,5), depolarizing_channels), name = "BSM depol"))
                
                POVM_1_OPs = generate_sqrt_POVM_MPO(sites=measurements[1], outcome = det_outcome, total_sites=num_modes, efficiency=1, N=N, pnr = pnr)
                POVM_1_OPs.extend(generate_sqrt_POVM_MPO(sites=measurements[0], outcome = 0, total_sites=num_modes, efficiency=1, N=N, pnr = pnr))

                expectation_ops = [quantum_channel(N = N, num_modes = num_modes, formalism = "closed", unitary_op = DET_MPO, name = "Det POVM") for DET_MPO in POVM_1_OPs]
                # quantum_channel_list.extend(expectation_ops)
    

                return quantum_channel_list, expectation_ops


            # Rotate and measure:
            rotator_node_2.add_tag("L5")
            rho_rotated = tensor_network_apply_op_vec(rotator_node_2, idler_rotated_psi, compress=compress, contract = contract, cutoff = error_tolerance)

            # read_quantum_state(psi)
            # read_quantum_state(rho_rotated)

            for POVM_OP in meas_ops:
                POVM_OP.add_tag("L6")
                rho_rotated = tensor_network_apply_op_vec(POVM_OP, rho_rotated, compress=compress, contract = contract, cutoff = error_tolerance)
        
            if draw:
                # only for drawing the TN. Not used otherwise
                fix = {(f"L{j}",f"I{num_modes - i-1}"):(3*j,i+5) for j in range(10) for i in range(10)}
                rho_rotated.draw(color = [r'$HH+VV$', r'$U_{BS_H}$', r"$U_{BS_V}$", 'POVM', r'$Rotator_I$', r'$Rotator_S$'], title = "Polarization entanglement swapping MPS", fix = fix, show_inds = True, show_tags = False)
                # rho_rotated.draw_tn()
            coincidence_probs.append((rho_rotated.norm())**2)
        coincidence.append(coincidence_probs)
    
    return np.array(coincidence)
