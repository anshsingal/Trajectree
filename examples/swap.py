from trajectree.sequence.swap import perform_swapping_simulation
import numpy as np
from matplotlib import pyplot as plt

# Set simulation params
N = 4
error_tolerance = 1e-7

params = {
    "PA_det_eff": 0.96,
    "BSM_det_loss_1":  0.045,
    "BSM_det_loss_2": 0.135,
    "BSM_dark_counts_1": 1,
    "BSM_dark_counts_2": 1,
    "alpha_list": np.array([np.pi/2]),
    "delta_list": np.array([np.pi/2]),
    "channel_loss": 1e-5,
    "chi": 0.24,
    "BSM_meas": {0:(2,3), 1:(6,7)},

    "if_analyze_entanglement": True,
    "calc_fidelity": False,
}

num_modes = 8
# Create vacuum state
 
# idler_angles = np.linspace(0, np.pi, 1)
# signal_angles = np.linspace(0, 4*np.pi, 30)

num_simulations = 400
visibilities = []

cache_sizes = [6]

for i in np.linspace(0.1, 0.5, 10):
    prob_same_phase = 0
    prob_diff_phase = 0

    # params['chi'] = i
    params["BSM_dark_counts_1"] = 1+1e-3
    params["BSM_dark_counts_2"] = 1+1e-3

    params["chi"] = i

    params["alpha_list"] = np.array([np.pi/2])

    fidelities, probabilities, t_eval = perform_swapping_simulation(N, num_modes, num_simulations, params = params, error_tolerance = error_tolerance)
    print("probabilities:", probabilities)
    print("next case")
    print("fidelities:", fidelities)
    prob_same_phase += np.mean(probabilities)

    params["alpha_list"] = np.array([3*np.pi/2])

    fidelities, probabilities, t_eval = perform_swapping_simulation(N, num_modes, num_simulations, params = params, error_tolerance = error_tolerance)
    print("probabilities:", probabilities)
    print("fidelities:", fidelities)
    prob_diff_phase += np.mean(probabilities)

    visibilities.append((prob_same_phase - prob_diff_phase) / (prob_same_phase + prob_diff_phase))


print(visibilities)
np.save("visibilities.npy", visibilities)
print(t_eval.graph)