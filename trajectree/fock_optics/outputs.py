from scipy import sparse as sp

import numpy as np

from matplotlib import pyplot as plt

# Generating labels for reading state. 
def generate_labels(num_systems, N):
    dim = N**2
    labels = []
    state_labels = []
    for i in range(dim):
        state_labels.append(f"{i//N}H{i%N}V")
    # print("sates:", self.state_labels)
    for i in range(dim**num_systems):
        new_label = ""
        for j in range(num_systems-1, -1, -1):
            # print("appending to labels:", f"{self.state_labels[(i//self.dim**j)%self.dim]}_{chr(65+j)} ")
            new_label += f"{state_labels[(i//dim**j)%dim]}_{chr(65+j)} "
        labels.append(new_label[:-1])
    return labels

def read_quantum_state(TN_state, N, num_states = 4, return_dense = False, precision = 10):
    dense_state = TN_state.to_dense()
    if return_dense: return dense_state
    dense_state = np.reshape(dense_state.data, (-1, 1), order = 'C')
    dense_state = sp.csr_matrix(dense_state)
    dense_state.data = np.round(dense_state.data, precision)
    dense_state.eliminate_zeros()

    print_quantum_state(N, dense_state, num_states)

def print_quantum_state(N, dense_state, num_states = 4):
    labels = generate_labels(num_states,N)
    state = dense_state.nonzero()[0]
    print("Corresponding Basis terms:")
    for k in state: print(labels[k],"-",k,"-",dense_state[k].data)



def plot_coincidences(coincidence, idler_angles, signal_angles, title = ''):
    visibilities = []
    for i in range(len(coincidence)):
        visibility = (max(coincidence[i]) - min(coincidence[i])) / (max(coincidence[i]) + min(coincidence[i]))
        visibilities.append(visibility)
        # print(visibility, coincidence[i])

    idler_angles = np.array(list(map(float, idler_angles)))/np.pi

    plt.figure()
    plt.grid(True)
    for i in range(len(idler_angles)):
        # print(fringe_real[i])
        plt.plot(signal_angles, coincidence[i], label=r'{:.2f}$\pi$'.format(idler_angles[i]))
    plt.title(title)
    plt.ylabel("Coincidence probability")
    plt.xlabel(r"$\alpha$ (rad)")    
    plt.legend(title = "$\delta$")

    plt.figure()
    plt.grid(True)
    plt.plot(idler_angles*np.pi, visibilities)
    plt.title("Visiblilities")
    plt.ylabel("Visibility")
    plt.xlabel(r"$\delta$")    
