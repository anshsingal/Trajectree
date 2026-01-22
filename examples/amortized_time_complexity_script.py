from trajectree.sequence.swap import perform_swapping_simulation, create_swapping_simulation
import numpy as np
from matplotlib import pyplot as plt

# Set simulation params
def run_experiment(max_cache_nodes=-1):
    N = 4
    error_tolerance = 1e-13

    params = {
        "PA_det_eff": (0.04),
        "BSM_det_loss_1":  (0.045),
        "BSM_det_loss_2": (0.135),
        "BSM_dark_counts_1": 3e-5,
        "BSM_dark_counts_2": 3e-5,
        "alpha_list": np.array([np.pi/2]),
        "delta_list": np.array([np.pi/2]),
        "channel_loss": 1e-7,
        "chi": 0.0587, # 0.06, # 0.24,
        "BSM_meas": {0:(2,3), 1:(6,7)},

        "if_analyze_entanglement": True,
        "calc_fidelity": False,
        "damping_error": True,
        "depolarizing_error": False,
        "max_cache_nodes": max_cache_nodes
    }

    num_modes = 8
    # Create vacuum state
    
    # idler_angles = np.linspace(0, np.pi, 1)
    # signal_angles = np.linspace(0, 4*np.pi, 30)

    num_simulations = 1000 # 20

    cache_size = 10
    iter = 0
    max_iter = 5
    threshold = 1e-4
    times = []

    while iter < max_iter:        
        psi_same, t_eval_same = create_swapping_simulation(N, num_modes, params, cache_size, error_tolerance = 1e-10)


        # np.random.seed(iter)
        _, _, times_same = perform_swapping_simulation(psi_same, t_eval_same, num_simulations, verbose = False)
        times.append(times_same)
        
        iter += 1
        print("iter:", iter)

    times_avg = np.mean(np.array(times).T, axis = 1)

    avg_times = [np.mean(times_avg[:i]) for i in range(1, len(times_avg))]

    np.save(f"amortized_sample_complexity_{max_cache_nodes}.npy", avg_times)

    # return times

run_experiment(max_cache_nodes=5)
run_experiment(max_cache_nodes=10)
run_experiment(max_cache_nodes=20)
run_experiment(max_cache_nodes=30)
run_experiment(max_cache_nodes=40)
run_experiment(max_cache_nodes=50)
run_experiment(max_cache_nodes=100)
run_experiment(max_cache_nodes=500)
run_experiment(max_cache_nodes=1000)