import numpy as np
from quimb.tensor import MatrixProductOperator as mpo #type: ignore
from quimb.tensor.tensor_arbgeom import tensor_network_apply_op_vec #type: ignore
from .fock_optics.outputs import read_quantum_state
from treelib import Tree
import heapq
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
f_handler = logging.FileHandler('log.log')
logger.addHandler(f_handler)


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
        self.trajectory_indices = np.array(trajectory_indices)
        self.accesses = 1
    
    def __lt__(self, other):
        return self.accesses < other.accesses

class trajectory_evaluator():
    def __init__(self, quantum_channels, cache_size = 7, max_cache_nodes = -1):
        self.quantum_channels = quantum_channels
        self.kraus_channels = []
        for quantum_channel in self.quantum_channels:
            if quantum_channel.formalism == 'kraus':
                self.kraus_channels.append(quantum_channel)

        self.trajectree = [{} for i in range(len(self.kraus_channels)+1)] # +1 because you also cache the end of the simulation so you prevent doing the final unitary operations multiple times. 
        self.traversed_nodes = ()
        self.cache_size = cache_size
        self.cache_heap = []
        self.max_cache_nodes = max_cache_nodes

        # Visualizing the Trajectree:
        self.graph = Tree()
        self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes))
        # self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes))  # root node

        # for debugging only:
        # self.cache_hit = 0
        # self.cache_miss = 0
        # self.cache_partial_hit = 0


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



    def discover_trajectree_node(self, psi, kraus_MPOs, error_tolerance, normalize = True, selected_trajectory_index = None):
        # read_quantum_state(psi, N=3)
        trajectories, trajectory_weights = self.apply_kraus(psi, kraus_MPOs, error_tolerance, normalize)

        # Instead of caching trajectree node right here, we save the node with all the trajectories and weights, and later decide which trajectories to cache after any closed operations. 
        # cached_trajectory_indices = self.cache_trajectree_node(trajectory_weights, trajectories) # cached_trajectory_indices is returned only for debugging. 
        new_node = trajectree_node(trajectory_weights, trajectories, np.arange(len(trajectory_weights)))
        self.trajectree[len(self.traversed_nodes)][self.traversed_nodes] = new_node

        if selected_trajectory_index == None:
            try:
                selected_trajectory_index = np.random.choice(a = len(trajectory_weights), p = trajectory_weights/sum(trajectory_weights))
                logger.info("selected index while discovering was: %d", selected_trajectory_index)
            except:
                self.graph.show()
                raise Exception(f"traversed nodes: {self.traversed_nodes} trajectory_weights: {trajectory_weights} invalid")

        
        self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)

        return trajectories[selected_trajectory_index]


    def query_trajectree(self, psi, kraus_MPOs, error_tolerance, selected_trajectory_index = None, normalize = True):      
        if self.cache_size == 0:
            trajectories, trajectory_weights = self.apply_kraus(psi, kraus_MPOs, error_tolerance, normalize)
            selected_trajectory_index = np.random.choice(a = len(trajectory_weights), p = trajectory_weights/sum(trajectory_weights))
            logger.info("selected index while not caching was: %d", selected_trajectory_index)
            self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)
            return trajectories[selected_trajectory_index]

        # Check if the node has been discovered previously.
        if self.traversed_nodes in self.trajectree[len(self.traversed_nodes)]: 
            self.new_node_discovered = False # If the node has been found, we do not cache the unitary. The unitary is either already cached or we don't need to cache it at all.

            node = self.trajectree[len(self.traversed_nodes)][self.traversed_nodes] # If found, index that node into the node object to call the probabilities and trajectories cached inside it.
            if selected_trajectory_index == None:
                selected_trajectory_index = np.random.choice(a = len(node.weights), p = node.weights/sum(node.weights)) # The cached nodes have all the weights, but not all the trajectories cache. So, we can select
                logger.info("selected index while discovered node found was: %d", selected_trajectory_index)                      # what trajecory our traversal takes and later see if the actual trajectory has been cached or needs to be retrieved. 

            # Check if cached trajectory is present:
            if selected_trajectory_index in node.trajectory_indices:  
                self.cached_trajectory_found = True # If we're skipping the unitary entirely, it just does not matter whether we cache the unitary or not.
                psi = node.trajectories[np.where(node.trajectory_indices == selected_trajectory_index)[0][0]]
                # self.cache_hit += 1
            else: 
                self.cached_trajectory_found = False # If the trajectory has not been cached, we will have to apply the unitary to it.
                psi = tensor_network_apply_op_vec(self.kraus_channels[len(self.traversed_nodes)].get_MPOs()[selected_trajectory_index], psi, compress=True, contract = True, cutoff = error_tolerance) # If not, simply calculate that trajectory. You don't need to cache it since we have already cached what we had to.  
                if normalize:
                    # After this, the trajectory is always normalized. 
                    psi /= np.sqrt(node.weights[selected_trajectory_index])
                if node in self.cache_heap:
                    node.trajectories.append(psi)
                    node.trajectory_indices = np.append(node.trajectory_indices, selected_trajectory_index)
                # self.cache_partial_hit += 1

            self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)

            # Here, we check if the node needs to be cached or not. 
            node.accesses += 1
            if len(node.trajectory_indices) == 0: # If the node has been discovered but not cached
                if node.accesses > self.cache_heap[0].accesses: # the new node has been accesses more often than the top of the heap, so, we can replace the top of the heap with the new node.
                    # You'll need to add a functionality here to start caching all the trajectories which have been added to the heap. 
                    old_node = heapq.heapreplace(self.cache_heap, node)
                    del old_node.trajectories
                    old_node.trajectories = [] 
                    old_node.trajectory_indices = np.array([])
            heapq.heapify(self.cache_heap)


        else: # If the node has not been discovered, we'll have to find all weights and cache the results.
            self.new_node_discovered = True
            self.cached_trajectory_found = False
            # self.cache_miss += 1
            psi = self.discover_trajectree_node(psi, kraus_MPOs, error_tolerance, normalize, selected_trajectory_index = selected_trajectory_index)
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
            psi_new_dense = self.perform_simulation(psi, error_tolerance, trajectree_indices = trajectree_indices, normalize = False).to_dense()
            dm += psi_new_dense @ psi_new_dense.conj().T
        return dm

    def update_cached_node(self, unitary_MPOs, last_discovered_node, error_tolerance):
        for kraus_idx in range(len(last_discovered_node.trajectories)):

            if last_discovered_node.trajectories[kraus_idx] is not None:
                last_discovered_node.trajectories[kraus_idx] = self.apply_unitary_MPOs(last_discovered_node.trajectories[kraus_idx], unitary_MPOs, error_tolerance)
                last_discovered_node.weights[kraus_idx] = np.real(last_discovered_node.trajectories[kraus_idx].H @ last_discovered_node.trajectories[kraus_idx])
                
                if last_discovered_node.weights[kraus_idx] < 1e-25:
                    last_discovered_node.trajectories[kraus_idx] = None
                    last_discovered_node.weights[kraus_idx] = 0

    def cache_trajectree_node(self):
        last_discovered_node = self.trajectree[len(self.traversed_nodes)-1][self.traversed_nodes[:-1]]

        sorted_indices = np.argsort(last_discovered_node.weights)

        if len(self.cache_heap) < self.max_cache_nodes or self.max_cache_nodes == -1:
            cached_trajectory_indices = sorted_indices[-self.cache_size:]
            cached_trajectories = np.array(last_discovered_node.trajectories)[cached_trajectory_indices]
            
            last_discovered_node.trajectories = cached_trajectories
            last_discovered_node.trajectory_indices = cached_trajectory_indices
            heapq.heappush(self.cache_heap, last_discovered_node)
        else:
            last_discovered_node.trajectories = np.array([])
            last_discovered_node.trajectory_indices = np.array([])

        self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes), parent = str(self.traversed_nodes[:-1]))

    # NOTE: USE NORMALIZE = TRUE FOR TRAJECTORY SIMULATIONS. USE NORMALIZE = FALSE FOR DENSITY MATRIX CALCULATIONS.
    def perform_simulation(self, psi, error_tolerance, trajectree_indices = None, normalize = True):
        self.traversed_nodes = ()
        self.cached_trajectory_found = False
        self.new_node_discovered = False

        for quantum_channel in self.quantum_channels:
            if quantum_channel.formalism == 'kraus':
                if self.new_node_discovered: # If we had previously discovered a new node, cache the previously discovered node. 
                    self.cache_trajectree_node()  
                
                # Reset the flags for the next kraus operation. 
                self.cached_trajectory_found = False
                self.new_node_discovered = False

                kraus_MPOs = quantum_channel.get_MPOs()
                if not trajectree_indices == None: # If the list of trajectoery indices is provided, we will use that to traverse the trajectree. The random number generators will not be used.
                    psi = self.query_trajectree(psi, kraus_MPOs, error_tolerance, trajectree_indices.pop(0), normalize)
                else: # In this branch, you actually select the trajectory redomly and perform realistic simulations. 
                    psi = self.query_trajectree(psi, kraus_MPOs, error_tolerance, normalize = normalize)

            elif quantum_channel.formalism == 'closed' and not self.cached_trajectory_found:
                unitary_MPOs = quantum_channel.get_MPOs()
                                
                if self.cache_size == 0: # If we aren't aching the trajectories at all, simply apply the unitary MPOs to the state.
                    psi = self.apply_unitary_MPOs(psi, unitary_MPOs, error_tolerance)
                    continue

                if self.new_node_discovered:
                    last_discovered_node = self.trajectree[len(self.traversed_nodes)-1][self.traversed_nodes[:-1]] # Here, we have len(self.traversed_nodes)-1 because we want the last level explored in the tree in this run
                                                                                                           # and, we have self.traversed_nodes[:-1] because self.traversed_nodes corresponds to the node after applying the kruas operation. 
                    self.update_cached_node(unitary_MPOs, last_discovered_node, error_tolerance)
                    traj_idx = np.where(last_discovered_node.trajectory_indices == self.traversed_nodes[-1])    
                    psi = last_discovered_node.trajectories[traj_idx[0][0]]
                else:
                    psi = self.apply_unitary_MPOs(psi, unitary_MPOs, error_tolerance)
                    if np.real(psi.H @ psi) < 1e-25:
                        psi = None

            if psi == None:
                return 0
        return psi
