from ..fock_optics.noise_models import *
from ..fock_optics.measurement import *
from ..fock_optics.utils import *
from ..fock_optics.light_sources import *

from ..trajectory import *

import time 
import numpy as np
import copy

def generate_swapping_circuit(N, num_modes, site_tags, bsm_det_effs, bsm_dark_counts_gain, bsm_measurements,  channel_loss, error_tolerance):
    quantum_channel_list = []

    # Amplitude damping due to fibers
    damping_kraus_ops = single_mode_bosonic_noise_channels(noise_parameter = channel_loss, N = N)
    two_mode_kraus_ops = [sp.kron(op, op) for op in damping_kraus_ops]
    quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((2,3), two_mode_kraus_ops))) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.
    quantum_channel_list.append(quantum_channel(N = N, num_modes = num_modes, formalism = "kraus", kraus_ops_tuple = ((6,7), two_mode_kraus_ops))) # The tuples in this list are defined as (sites, kraus_ops). The sites are the sites where the Kraus ops are applied.

    # Quantum channel for the Bell state measurement
    # BSM_MPOs = bell_state_measurement(None, N, site_tags, num_modes, bsm_det_effs, error_tolerance, measurements = bsm_measurements, pnr = False, use_trajectory = True, return_MPOs = True, compress=True, contract=True)
    # BSM_quantum_channels = [quantum_channel(N = N, num_modes = num_modes, formalism = "closed", unitary_MPOs = BSM_MPO, name = "BSM") for BSM_MPO in BSM_MPOs]
    BSM_quantum_channels = bell_state_measurement(None, N, site_tags, num_modes, bsm_det_effs, bsm_dark_counts_gain, error_tolerance, measurements = bsm_measurements, pnr = False, use_trajectory = True, return_MPOs = True, compress=True, contract=True)
    quantum_channel_list.extend(BSM_quantum_channels)

    return quantum_channel_list

def analyze_entanglement(quantum_channel_list, N, site_tags, num_modes, efficiency, error_tolerance, idler_angles, signal_angles):
    PA_MPOs = rotate_and_measure(None, N, site_tags, num_modes, efficiency, error_tolerance, idler_angles, signal_angles, return_MPOs = True)
    PA_quantum_channels = [quantum_channel(N = N, num_modes = num_modes, formalism = "closed", unitary_MPOs = PA_MPO) for PA_MPO in PA_MPOs]
    print("num pa quantum channels:", len(PA_quantum_channels))
    quantum_channel_list.extend(PA_quantum_channels)


def create_swapping_initial_state(num_modes, N, mean_photon_num, error_tolerance):
    # Create Vacuum state:
    vacuum = create_vacuum_state(num_modes=num_modes, N=N)

    # Entangled state from EPS
    psi, TMSV_state = light_source(vacuum, N, mean_photon_num, num_modes, error_tolerance, compress=True, contract=True)

    psi = extend_MPS(psi)
    return psi

def perform_swapping_simulation(N, num_modes, num_simulations, params, error_tolerance = 1e-10):

    psi = create_swapping_initial_state(num_modes, N, params["chi"], error_tolerance)

    quantum_channels = generate_swapping_circuit(N, num_modes, psi.site_tags, [params["BSM_det_loss_1"], params["BSM_det_loss_2"]], [params["BSM_dark_counts_1"], params["BSM_dark_counts_2"]], params["BSM_meas"], params["channel_loss"], error_tolerance)

    if params["if_analyze_entanglement"]:
        analyze_entanglement(quantum_channels, N, psi.site_tags, num_modes, params["PA_det_loss"], error_tolerance, params["alpha_list"], params["delta_list"])

    t_eval = trajectory_evaluator(quantum_channels)

    fidelities = []
    probabilities = []

    for i in range(num_simulations): 
        start = time.time()
        psi_iter = copy.deepcopy(t_eval.perform_simulation(psi, error_tolerance, normalize = False))

        probabilities.append(psi_iter.normalize())
        
        if params["calc_fidelity"]:
            fidelity = np.abs(calc_fidelity_swapping(psi_iter, "psi_minus", N, error_tolerance))
            fidelities.append(fidelity)
    
        time_taken = time.time() - start
        # print("time taken for simulation", i, ":", time_taken)

    print("completed set", "cache_hits:", t_eval.cache_hit, "cache_partial_hits:", t_eval.cache_partial_hit, "cache_misses:", t_eval.cache_miss,  "time taken:", time_taken)

    return fidelities, probabilities, t_eval

