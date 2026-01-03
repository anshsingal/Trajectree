import numpy as np
from quimb.tensor import MatrixProductOperator as mpo #type: ignore
from quimb.tensor.tensor_arbgeom import tensor_network_apply_op_vec #type: ignore
from .fock_optics.outputs import read_quantum_state
from treelib import Tree


class quantum_channel:
    def __init__(self, N, num_modes, formalism, kraus_ops_tuple = None, unitary_MPOs = None, name = "quantum_channel"):
        self.N = N
        self.name = name
        self.num_modes = num_modes
        self.formalism = formalism
        if self.formalism == 'kraus':
            # Calculate the MPOs of the Kraus operators
            self.kraus_MPOs = quantum_channel.find_quantum_channels_MPOs(kraus_ops_tuple, N, num_modes)
        elif self.formalism == 'closed':
            self.unitary_MPOs = unitary_MPOs

    def get_MPOs(self):
        if self.formalism == 'closed':
            return self.unitary_MPOs
        elif self.formalism == 'kraus':
            return self.kraus_MPOs

    @staticmethod
    def find_quantum_channels_MPOs(ops_tuple, N, num_modes):
        (sites, ops) = ops_tuple
        quantum_channels = quantum_channel.calc_mpos(ops, N, sites, num_modes)
        return quantum_channels

    # Just a function which calcualte the MPOs of the Kraus ops
    @staticmethod
    def calc_mpos(ops, N, sites, num_modes):
        MPOs = []
        for op in ops:
            # print("matrix norm:", np.linalg.norm(op.todense()))
            MPO = mpo.from_dense(op.todense(), dims = N, sites = sites, L=num_modes, tags="op")
            MPOs.append(MPO)
        return MPOs


class trajectree_node:
    def __init__(self, weights, trajectories, trajectory_indices):
        self.weights = weights
        self.trajectories = trajectories
        self.trajectory_indices = trajectory_indices

class trajectory_evaluator():
    def __init__(self, quantum_channels, cache_size = 7):
        self.quantum_channels = quantum_channels
        self.kraus_channels = []
        for quantum_channel in self.quantum_channels:
            if quantum_channel.formalism == 'kraus':
                self.kraus_channels.append(quantum_channel)

        self.trajectree = [{} for i in range(len(self.kraus_channels)+1)] # +1 because you also cache the end of the simulation so you prevent doing the final unitary operations multiple times. 
        self.traversed_nodes = ()
        self.cache_size = cache_size

        # Visualizing the Trajectree:
        self.graph = Tree()
        # self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes))  # root node

        # for debugging only:
        self.cache_hit = 0
        self.cache_miss = 0
        self.cache_partial_hit = 0


    def apply_kraus(self, psi, kraus_MPOs, error_tolerance, normalize = True):
        trajectory_weights = np.array([])
        trajectories = np.array([])

        # read_quantum_state(psi, N=3)
        for kraus_MPO in kraus_MPOs:

            trajectory = tensor_network_apply_op_vec(kraus_MPO, psi, compress=True, contract = True, cutoff = error_tolerance)
            
            # trajectory.draw()

            # this weight is almost arbitrary (can be greater than 1). This is because the Kraus operators themselves are not unitary. 
            trajectory_weight = np.real(trajectory.H @ trajectory)
            # print("trajectory weight:", trajectory_weight)
            # print("trajectory:")
            # read_quantum_state(trajectory, N=3)
            # print()

            # if trajectory_weight < 1e-25: # Using 1e-25 arbitrarily. Trajectories with weight less than this are pruned.  
            #     continue

            if normalize:
                # After this, the trajectory is always normalized.
                if trajectory_weight < 1e-25:
                    trajectory = None
                    trajectory_weight = 0
                else: 
                    trajectory /= np.sqrt(trajectory_weight)

            trajectory_weights = np.append(trajectory_weights, trajectory_weight)
            trajectories = np.append(trajectories, trajectory)
        assert len(np.nonzero(trajectory_weights)[0]) > 0, f"All trajectories have zero weight. The input state had magnitude: {psi.H @ psi}"   
        # print("trajectories:")
        # for i in range(len(trajectories)):
        #     read_quantum_state(trajectories[i], N=3)

        return trajectories, trajectory_weights


    def cache_trajectree_node(self, trajectory_weights, trajectories):
        sorted_indices = np.argsort(trajectory_weights)

        # print("trajectory_weights", trajectory_weights)

        cached_trajectory_indices = sorted_indices[-self.cache_size:]
        cached_trajectories = np.array(trajectories)[cached_trajectory_indices]

        new_node = trajectree_node(trajectory_weights, cached_trajectories, cached_trajectory_indices)
        self.trajectree[len(self.traversed_nodes)][self.traversed_nodes] = new_node
        if len(self.traversed_nodes) == 0:
            self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes))  # root node
        else:
            self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes), parent = str(self.traversed_nodes[:-1]))

        self.last_cached_node = new_node

        return cached_trajectory_indices


    def discover_trajectree_node(self, psi, kraus_MPOs, error_tolerance, normalize = True, selected_trajectory_index = None):
        # read_quantum_state(psi, N=3)
        trajectories, trajectory_weights = self.apply_kraus(psi, kraus_MPOs, error_tolerance, normalize)

        cached_trajectory_indices = self.cache_trajectree_node(trajectory_weights, trajectories) # cached_trajectory_indices is returned only for debugging. 

        # print("trajectory weights:", trajectory_weights)

        if selected_trajectory_index == None:
            try:
                selected_trajectory_index = np.random.choice(a = len(trajectory_weights), p = trajectory_weights/sum(trajectory_weights))
            except:
                self.graph.show()
                raise Exception(f"traversed nodes: {self.traversed_nodes} trajectory_weights: {trajectory_weights} invalid")

        
        self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)

        return trajectories[selected_trajectory_index]


    def query_trajectree(self, psi, kraus_MPOs, error_tolerance, cache = True, selected_trajectory_index = None, normalize = True):
        self.skip_unitary = False
        self.cache_unitary = False
        # print("entering trajectree magnitude:", psi.H @ psi)
        
        if cache == False:
            trajectories, trajectory_weights = self.apply_kraus(psi, kraus_MPOs, error_tolerance, normalize)
            selected_trajectory_index = np.random.choice(a = len(trajectory_weights), p = trajectory_weights/sum(trajectory_weights))
            # psi = tensor_network_apply_op_vec(self.kraus_channels[len(self.traversed_nodes)].get_MPOs()[selected_trajectory_index], psi, compress=True, contract = True, cutoff = error_tolerance)
            self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)
            return trajectories[selected_trajectory_index]

        if self.traversed_nodes in self.trajectree[len(self.traversed_nodes)]: # Check if the dictionary at level where the traversal is now, i.e., len(self.traversed_nodes)
                                                                               # has the path that the present traversal has taken. 
            node = self.trajectree[len(self.traversed_nodes)][self.traversed_nodes] # If found, index that node into the node object to call the probabilities and trajectories cached inside it.
            if selected_trajectory_index == None:
                # print("cached weights:", node.weights, "at:", self.traversed_nodes)
                selected_trajectory_index = np.random.choice(a = len(node.weights), p = node.weights/sum(node.weights)) # The cached nodes have all the weights, but not all the trajectories cache. So, we can select
                                                                                                  # what trajecory our traversal takes and later see if the actual trajectory has been cached or needs to be retrieved. 
                # print("cached weights:", node.weights, "at:", self.traversed_nodes, "selected trajectory:", selected_trajectory_index)
            self.cache_unitary = False # If the node has been found, we do not cache the unitary. The unitary is either already cached or we don't need to cache it at all.

            if selected_trajectory_index in node.trajectory_indices: # See if the selected trajectory's MPS has been cached or not. 
                self.skip_unitary = True # If we're skipping the unitary entirely, it just does not matter whether we cache the unitary or not.
                self.cache_hit += 1
                psi = node.trajectories[np.where(node.trajectory_indices == selected_trajectory_index)[0][0]]
            else: 
                self.skip_unitary = False # If the trajectory has not been cached, we will have to apply the unitary to it.
                self.cache_partial_hit += 1
                psi = tensor_network_apply_op_vec(self.kraus_channels[len(self.traversed_nodes)].get_MPOs()[selected_trajectory_index], psi, compress=True, contract = True, cutoff = error_tolerance) # If not, simply calculate that trajectory. 
                                                                                                                                                                                             # You don't need to cache it since we have already cached what we had to.  
                # print("cache partial hit. new state:" )
                # read_quantum_state(psi, N=3)
                if normalize:
                    # After this, the trajectory is always normalized. 
                    psi /= np.sqrt(node.weights[selected_trajectory_index])
            self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)


        else: # If the node has not been discovered, we'll have to find all weights and cache the results.
            # print("exploring new node at:", self.traversed_nodes)
            # read_quantum_state(psi, N=3) 
            self.skip_unitary = False
            self.cache_unitary = True
            self.cache_miss += 1
            psi = self.discover_trajectree_node(psi, kraus_MPOs, error_tolerance, normalize, selected_trajectory_index = selected_trajectory_index)

        # print("exitting trajectree magnitude:", psi.H @ psi)
        # if psi == None:
        #     # raise Exception(f"traversed nodes: {self.traversed_nodes} trajectory_weights: {trajectory_weights} invalid")
        #     self.graph.show()
        #     raise Exception(f"Psi is None in query trajectree. traversed nodes: {self.traversed_nodes}")
        return psi

    def apply_unitary_MPOs(self, psi, unitary_MPOs, error_tolerance):
        return tensor_network_apply_op_vec(unitary_MPOs, psi, compress=True, contract = True, cutoff = error_tolerance)


    def calculate_density_matrix(self, psi, error_tolerance):
        dm = 0
        trajectree_indices_list = [[]]
        for quantum_channel in self.quantum_channels:
            if quantum_channel.formalism == 'kraus':
                trajectree_indices_list = [[*i, j] for i in trajectree_indices_list for j in range(len(quantum_channel.get_MPOs()))]
        for trajectree_indices in trajectree_indices_list:
            psi_new_dense = self.perform_simulation(psi, error_tolerance, cache = True, trajectree_indices = trajectree_indices, normalize = False).to_dense()
            dm += psi_new_dense @ psi_new_dense.conj().T
        return dm

    def update_cached_node(self, unitary_MPOs, last_cached_node, error_tolerance):
        for kraus_idx in range(len(last_cached_node.trajectories)):
            if last_cached_node.trajectories[kraus_idx] is not None:
                last_cached_node.trajectories[kraus_idx] = self.apply_unitary_MPOs(last_cached_node.trajectories[kraus_idx], unitary_MPOs, error_tolerance)
                if np.real(last_cached_node.trajectories[kraus_idx].H @ last_cached_node.trajectories[kraus_idx]) < 1e-25:
                    last_cached_node.trajectories[kraus_idx] = None


    # NOTE: USE NORMALIZE = TRUE FOR TRAJECTORY SIMULATIONS. USE NORMALIZE = FALSE FOR DENSITY MATRIX CALCULATIONS.
    def perform_simulation(self, psi, error_tolerance, cache = True, trajectree_indices = None, normalize = True):
        self.traversed_nodes = ()
        self.skip_unitary = False
        self.cache_unitary = False
        for quantum_channel in self.quantum_channels:
            # print("operation:", quantum_channel.name, "formalism:", quantum_channel.formalism, "traversed nodes:", self.traversed_nodes)
            if quantum_channel.formalism == 'kraus':
                kraus_MPOs = quantum_channel.get_MPOs()
                # print("kraus op:", quantum_channel.name, "number of kraus ops:", len(kraus_MPOs))
                # print("before kraus ops:")
                # read_quantum_state(psi, N=3)
                if not trajectree_indices == None: # If the list of trajectoery indices is provided, we will use that to traverse the trajectree. The random number generators will not be used.
                    psi = self.query_trajectree(psi, kraus_MPOs, error_tolerance, cache, trajectree_indices.pop(0), normalize)
                else: # In this branch, you actually select the trajectory redomly and perform realistic simulations. 
                    # read_quantum_state(psi, N=3)
                    psi = self.query_trajectree(psi, kraus_MPOs, error_tolerance, cache = cache, normalize = normalize)
                # print("after kraus ops:")
                # read_quantum_state(psi, N=3)

            elif quantum_channel.formalism == 'closed' and not self.skip_unitary:
                # print("closed op:", quantum_channel.name)
                unitary_MPOs = quantum_channel.get_MPOs()
                                
                if not cache: # If we aren't aching the trajectories at all, simply apply the unitary MPOs to the state.
                    psi = self.apply_unitary_MPOs(psi, unitary_MPOs, error_tolerance)
                    continue

                last_cached_node = self.trajectree[len(self.traversed_nodes)-1][self.traversed_nodes[:-1]]
                
                if self.cache_unitary:
                    self.update_cached_node(unitary_MPOs, last_cached_node, error_tolerance)

                # This is where we are checking if the psi is cached or not. If it is, simply use the last cached node 
                # node to update psi. If not, apply the unitary MPOs to psi.
                traj_idx = np.where(last_cached_node.trajectory_indices == self.traversed_nodes[-1])    
                if traj_idx[0].size > 0:
                    psi = last_cached_node.trajectories[traj_idx[0][0]]
                else:
                    psi = self.apply_unitary_MPOs(psi, unitary_MPOs, error_tolerance)
                    if np.real(psi.H @ psi) < 1e-25:
                        psi = None
            
            else:
                # print("unitary skipped:", self.traversed_nodes)
                pass
            if psi == None:
                # print("Nothing returned")
                return 0
                # raise Exception("Psi is None")
            # read_quantum_state(psi, N=6)
            # print("next operation:")

        return psi
