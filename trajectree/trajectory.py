import numpy as np
from quimb.tensor import MatrixProductOperator as mpo #type: ignore
from quimb.tensor.tensor_arbgeom import tensor_network_apply_op_vec #type: ignore
from .fock_optics.outputs import read_quantum_state
# from treelib import Tree
from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from qutip import basis, expand_operator, Qobj
import heapq
import logging
import copy
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
f_handler = logging.FileHandler('log.log')
logger.addHandler(f_handler)


class quantum_channel:
    def __init__(self, N, num_modes, formalism, kraus_ops_tuple = None, unitary_op = None, backend = "tensor", name = "quantum_channel"):
        self.N = N
        self.name = name
        self.num_modes = num_modes
        self.formalism = formalism
        self.backend = backend
        if self.formalism == 'kraus':
            if self.backend == "tensor":
                # Calculate the MPOs of the Kraus operators
                self.kraus_ops = quantum_channel.find_quantum_channels_MPOs(kraus_ops_tuple, N, num_modes)
            elif self.backend == 'statevector':
                self.kraus_ops = self.create_qutip_ops(kraus_ops_tuple)

        elif self.formalism == 'closed':
            if self.backend == "tensor":
                self.unitary_op = unitary_op
            elif self.backend == "statevector":
                self.unitary_op = expand_operator(oper = unitary_op[1], dims = [self.N] * self.num_modes, targets = unitary_op[0])


    def get_ops(self):
        if self.formalism == 'closed':
            return self.unitary_op
        elif self.formalism == 'kraus':
            return self.kraus_ops

    def create_qutip_ops(self, ops_tuple):
        (targets, ops) = ops_tuple
        return_ops = []
        for op in ops:
            return_ops.append(expand_operator(oper = Qobj(op.toarray()), dims = [self.N] * self.num_modes, targets = targets))
        return return_ops

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
    def __init__(self, quantum_channels, cache_size = 7, max_cache_nodes = -1, backend = "tensor", calc_expectation = False, observable_ops = []):
        self.quantum_channels = quantum_channels
        self.kraus_channels = []
        for quantum_channel in self.quantum_channels:
            if quantum_channel.formalism == 'kraus':
                self.kraus_channels.append(quantum_channel)

        self.backend = backend
        self.trajectree = [{} for i in range(len(self.kraus_channels)+1)] # +1 because you also cache the end of the simulation so you prevent doing the final unitary operations multiple times. 
        self.traversed_nodes = ()
        self.cache_size = cache_size
        self.cache_heap = []
        self.max_cache_nodes = max_cache_nodes

        self.calc_expectation = calc_expectation
        self.observable_ops = observable_ops


        # Visualizing the Trajectree:
        self.graph = nx.DiGraph()
        # self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes))  # root node

        # for debugging only:
        self.cache_hit = 0
        self.cache_miss = 0
        self.cache_partial_hit = 0

    def apply_op(self, psi, op, error_tolerance):
        if self.backend == 'tensor':
            return tensor_network_apply_op_vec(op, psi, compress=True, contract = True, cutoff = error_tolerance)
        if self.backend == 'statevector':
            return op * psi

    def calc_magnitude(self, psi):
        if self.backend == 'tensor':
            return np.real(psi.H @ psi)
        if self.backend == 'statevector':
            return np.real(psi.norm())

    def calc_inner_product(self, psi1, psi2):
        if self.backend == 'tensor':
            return np.real(psi1.H @ psi2)
        if self.backend == 'statevector':
            return np.real(psi1.dag() * psi2)


    def apply_kraus(self, psi, kraus_ops, error_tolerance, normalize = True):
        trajectory_weights = np.array([])
        trajectories = np.array([])

        # read_quantum_state(psi, N=3)
        for kraus_op in kraus_ops:

            trajectory = self.apply_op(psi, kraus_op, error_tolerance)
            
            # trajectory.draw()

            # this weight is almost arbitrary (can be greater than 1). This is because the Kraus operators themselves are not unitary. 
            trajectory_weight = self.calc_magnitude(trajectory)
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
        assert len(np.nonzero(trajectory_weights)[0]) > 0, f"All trajectories have zero weight. The input state had magnitude: {self.calc_magnitude(psi)}"   
        # print("trajectories:")
        # for i in range(len(trajectories)):
        #     read_quantum_state(trajectories[i], N=3)

        return trajectories, trajectory_weights


    def cache_trajectree_node(self, trajectory_weights, trajectories):
        sorted_indices = np.argsort(trajectory_weights)

        # print("trajectory_weights", trajectory_weights)

        if len(self.cache_heap) < self.max_cache_nodes or self.max_cache_nodes == -1:
            push_node = True
            cached_trajectory_indices = sorted_indices[-self.cache_size:]
        else:
            push_node = False
            cached_trajectory_indices = []

        cached_trajectories = np.array(trajectories)[cached_trajectory_indices]

        new_node = trajectree_node(trajectory_weights, cached_trajectories, cached_trajectory_indices)
        self.trajectree[len(self.traversed_nodes)][self.traversed_nodes] = new_node
        if len(self.traversed_nodes) == 0: # root node
            # self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes))  # Tree lib implementation 
            self.graph.add_node(self.traversed_nodes)  # NetworkX implementation
        else:
            # self.graph.create_node(str(self.traversed_nodes), str(self.traversed_nodes), parent = str(self.traversed_nodes[:-1]))
            self.graph.add_edge(self.traversed_nodes[:-1], self.traversed_nodes)

        self.last_cached_node = new_node

        if push_node:
            heapq.heappush(self.cache_heap, new_node)

        return cached_trajectory_indices


    def discover_trajectree_node(self, psi, kraus_ops, error_tolerance, normalize = True, selected_trajectory_index = None):
        # read_quantum_state(psi, N=3)
        trajectories, trajectory_weights = self.apply_kraus(psi, kraus_ops, error_tolerance, normalize)

        cached_trajectory_indices = self.cache_trajectree_node(trajectory_weights, trajectories) # cached_trajectory_indices is returned only for debugging. 

        # print("trajectory weights:", trajectory_weights)

        if selected_trajectory_index == None:
            try:
                selected_trajectory_index = np.random.choice(a = len(trajectory_weights), p = trajectory_weights/sum(trajectory_weights))
                logger.info("selected index while discovering was: %d", selected_trajectory_index)
            except:
                self.show_graph()
                raise Exception(f"traversed nodes: {self.traversed_nodes} trajectory_weights: {trajectory_weights} invalid")

        
        self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)

        return trajectories[selected_trajectory_index]

    def show_graph(self, use_graphviz = False, node_descriptions = {}):
        """
            A typical node description would look like: node_descriptions = {(0,0,1): "Root Node", (0,0,1,2): "Child Node"}
        """
        if not use_graphviz: # You can either use the default layouts in NetworkX:  
            pos = nx.bfs_layout(self.graph, (), align = "vertical", scale = 1) 
        else: # For a pretty tree like graph, you need to have Graphviz installed:
            pos = graphviz_layout(self.graph, prog="dot") # You need Graphviz installed for this to work

        plt.figure(figsize=(8, 20))
        nx.draw(self.graph, pos, with_labels=False, node_color='blue', node_size=50, alpha=0.5)
        nx.draw_networkx_labels(self.graph, pos, labels=node_descriptions, font_size=12, font_color='black')
    
    
    def query_trajectree(self, psi, kraus_ops, error_tolerance, selected_trajectory_index = None, normalize = True):
        self.skip_unitary = False
        self.cache_unitary = False
        # print("entering trajectree magnitude:", self.calc_magnitude(psi))
        if self.cache_size == 0:
            trajectories, trajectory_weights = self.apply_kraus(psi, kraus_ops, error_tolerance, normalize)
            selected_trajectory_index = np.random.choice(a = len(trajectory_weights), p = trajectory_weights/sum(trajectory_weights))
            logger.info("selected index while not caching was: %d", selected_trajectory_index)
            # psi = self.apply_op(psi, self.kraus_channels[len(self.traversed_nodes)].get_ops()[selected_trajectory_index], error_tolerance)
            self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)
            return trajectories[selected_trajectory_index]

        if self.traversed_nodes in self.trajectree[len(self.traversed_nodes)]: # Check if the dictionary at level where the traversal is now, i.e., len(self.traversed_nodes)
                                                                               # has the path that the present traversal has taken. 
            logger.info(f"selected_trajectory_index: {selected_trajectory_index}")
            node = self.trajectree[len(self.traversed_nodes)][self.traversed_nodes] # If found, index that node into the node object to call the probabilities and trajectories cached inside it.
            if selected_trajectory_index == None:
                # print("cached weights:", node.weights, "at:", self.traversed_nodes)
                selected_trajectory_index = np.random.choice(a = len(node.weights), p = node.weights/sum(node.weights)) # The cached nodes have all the weights, but not all the trajectories cache. So, we can select
                logger.info(f"selected index while discovered node found: {selected_trajectory_index} with prob {node.weights[selected_trajectory_index]}")                      # what trajecory our traversal takes and later see if the actual trajectory has been cached or needs to be retrieved. 
                # print("cached weights:", node.weights, "at:", self.traversed_nodes, "selected trajectory:", selected_trajectory_index)
            self.cache_unitary = False # If the node has been found, we do not cache the unitary. The unitary is either already cached or we don't need to cache it at all.

            if selected_trajectory_index in node.trajectory_indices: # See if the selected trajectory's MPS has been cached or not. 
                self.skip_unitary = True # If we're skipping the unitary entirely, it just does not matter whether we cache the unitary or not.
                self.cache_hit += 1
                psi = node.trajectories[np.where(node.trajectory_indices == selected_trajectory_index)[0][0]]
                logger.info(f"selected outcome: {psi}")
            else: 
                self.skip_unitary = False # If the trajectory has not been cached, we will have to apply the unitary to it.
                self.cache_partial_hit += 1
                psi = self.apply_op(psi, self.kraus_channels[len(self.traversed_nodes)].get_ops()[selected_trajectory_index], error_tolerance) # If not, simply calculate that trajectory. 
                                                                                                                                               # You don't need to cache it since we have already cached what we had to.  
                # print("cache partial hit. new state:" )
                # read_quantum_state(psi, N=3)
                if normalize:
                    # After this, the trajectory is always normalized. 
                    psi /= np.sqrt(node.weights[selected_trajectory_index])
                if node in self.cache_heap and len(node.trajectory_indices) < self.cache_size:
                    node.trajectories = np.append(node.trajectories, psi)
                    # node.trajectories.append(psi)
                    node.trajectory_indices = np.append(node.trajectory_indices, selected_trajectory_index)

            self.traversed_nodes = self.traversed_nodes + (selected_trajectory_index,)

            # Here, we check if the node needs to be cached or not. 
            node.accesses += 1
            if len(node.trajectory_indices) == 0: # If the node has been discovered but not cached
                if node.accesses > self.cache_heap[0].accesses: # the new node has been accesses more often than the top of the heap, so, we can replace the top of the heap with the new node.
                    # print("in here!!!")
                    old_node = heapq.heapreplace(self.cache_heap, node)
                    del old_node.trajectories
                    old_node.trajectories = [] 
                    old_node.trajectory_indices = np.array([])
            heapq.heapify(self.cache_heap)


        else: # If the node has not been discovered, we'll have to find all weights and cache the results.
            # print("exploring new node at:", self.traversed_nodes)
            # read_quantum_state(psi, N=3) 
            self.skip_unitary = False
            self.cache_unitary = True
            self.cache_miss += 1
            psi = self.discover_trajectree_node(psi, kraus_ops, error_tolerance, normalize, selected_trajectory_index = selected_trajectory_index)

        # print("exitting trajectree magnitude:", self.calc_magnitude(psi))
        # if psi == None:
        #     # raise Exception(f"traversed nodes: {self.traversed_nodes} trajectory_weights: {trajectory_weights} invalid")
        #     self.graph.show()
        #     raise Exception(f"Psi is None in query trajectree. traversed nodes: {self.traversed_nodes}")
        return psi


    def calculate_density_matrix(self, psi, error_tolerance):
        dm = 0
        trajectree_indices_list = [[]]
        for quantum_channel in self.quantum_channels:
            if quantum_channel.formalism == 'kraus':
                trajectree_indices_list = [[*i, j] for i in trajectree_indices_list for j in range(len(quantum_channel.get_ops()))]
        for trajectree_indices in trajectree_indices_list:
            psi_new_dense = self.perform_simulation(psi, error_tolerance, trajectree_indices = trajectree_indices, normalize = False).to_dense()
            dm += psi_new_dense @ psi_new_dense.conj().T
        return dm

    def unitary_cached_trajectories(self, unitary_op, last_cached_node, error_tolerance):
        for kraus_idx in range(len(last_cached_node.trajectories)):
            if last_cached_node.trajectories[kraus_idx] is not None:
                last_cached_node.trajectories[kraus_idx] = self.apply_op(last_cached_node.trajectories[kraus_idx], unitary_op, error_tolerance)
                if self.calc_magnitude(last_cached_node.trajectories[kraus_idx]) < 1e-25: 
                    last_cached_node.trajectories[kraus_idx] = None

    def expectation_cached_trajectories(self, last_cached_node, temp_trajectories):
        for kraus_idx in range(len(last_cached_node.trajectories)):

            if last_cached_node.trajectories[kraus_idx] is not None:
                ev = self.calc_inner_product(temp_trajectories[kraus_idx], last_cached_node.trajectories[kraus_idx])
                if ev == None: ev = 0
                last_cached_node.trajectories[kraus_idx] = ev
                # if self.calc_magnitude(last_cached_node.trajectories[kraus_idx]) < 1e-25: 
                #     last_cached_node.trajectories[kraus_idx] = None



    # def nonunitary_cached_trajectories(self, ops, last_cached_node, error_tolerance):
    #     # for kraus_idx in range(len(last_discovered_node.trajectories)):
    #     for kraus_idx in range(len(last_cached_node.trajectories)):

    #         if last_cached_node.trajectories[kraus_idx] is not None:
    #             last_cached_node.trajectories[kraus_idx] = self.apply_op(last_cached_node.trajectories[kraus_idx], ops, error_tolerance)
    #             last_cached_node.weights[kraus_idx] = self.calc_magnitude(last_cached_node.trajectories[kraus_idx])
                
    #             if last_cached_node.weights[kraus_idx] < 1e-25:
    #                 last_cached_node.trajectories[kraus_idx] = None
    #                 last_cached_node.weights[kraus_idx] = 0

    def get_trajectree_node(self, address):
        return self.trajectree[len(address)][address]

    # NOTE: USE NORMALIZE = TRUE FOR TRAJECTORY SIMULATIONS. USE NORMALIZE = FALSE FOR DENSITY MATRIX CALCULATIONS.
    def perform_simulation(self, psi, error_tolerance, trajectree_indices = None, normalize = True):
        self.traversed_nodes = ()
        self.skip_unitary = False
        self.cache_unitary = False
        for quantum_channel in self.quantum_channels:
            # print("operation:", quantum_channel.name, "formalism:", quantum_channel.formalism, "traversed nodes:", self.traversed_nodes)
            if quantum_channel.formalism == 'kraus':
                kraus_ops = quantum_channel.get_ops()
                # print("kraus op:", quantum_channel.name, "number of kraus ops:", len(kraus_ops))
                # print("before kraus ops:")
                # read_quantum_state(psi, N=3)
                if not trajectree_indices == None: # If the list of trajectoery indices is provided, we will use that to traverse the trajectree. The random number generators will not be used.
                    psi = self.query_trajectree(psi, kraus_ops, error_tolerance, trajectree_indices.pop(0), normalize)
                else: # In this branch, you actually select the trajectory redomly and perform realistic simulations. 
                    # read_quantum_state(psi, N=3)
                    psi = self.query_trajectree(psi, kraus_ops, error_tolerance, normalize = normalize)
                # print("after kraus ops:")
                # read_quantum_state(psi, N=3)

            # In this case, the weights are not updated since the operations are just unitary. 
            elif quantum_channel.formalism == 'closed' and not self.skip_unitary:
                # print("closed op:", quantum_channel.name)
                unitary_op = quantum_channel.get_ops()
                                
                if self.cache_size == 0: # If we aren't aching the trajectories at all, simply apply the unitary ops to the state.
                    psi = self.apply_op(psi, unitary_op, error_tolerance)
                    continue

                last_cached_node = self.trajectree[len(self.traversed_nodes)-1][self.traversed_nodes[:-1]]
                
                if self.cache_unitary:
                    self.unitary_cached_trajectories(unitary_op, last_cached_node, error_tolerance)


                # This is where we are checking if the psi is cached or not. If it is, simply use the last cached node 
                # node to update psi. If not, apply the unitary ops to psi.
                traj_idx = np.where(last_cached_node.trajectory_indices == self.traversed_nodes[-1])    
                if traj_idx[0].size > 0:
                    psi = last_cached_node.trajectories[traj_idx[0][0]]
                else:
                    psi = self.apply_op(psi, unitary_op, error_tolerance)
                    if self.calc_magnitude(psi) < 1e-25:
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

        if self.calc_expectation:
            # print("state before expectation calculation:", psi)
            if not self.skip_unitary:
            
                # This is where we are checking if the psi is cached or not. If it is, simply use the last cached node 
                # node to update psi. If not, apply the unitary ops to psi.
                last_cached_node = self.get_trajectree_node(self.traversed_nodes[:-1])
                traj_idx = np.where(last_cached_node.trajectory_indices == self.traversed_nodes[-1])    

                if traj_idx[0].size > 0:
                    temp_trajectories = copy.deepcopy(last_cached_node.trajectories)
                else:
                    psi_temp = copy.deepcopy(psi)

                for quantum_channel in self.observable_ops:
                    observable_op = quantum_channel.get_ops()

                    if self.cache_unitary:
                        self.unitary_cached_trajectories(observable_op, last_cached_node, error_tolerance)

                    if traj_idx[0].size > 0:
                        psi = last_cached_node.trajectories[traj_idx[0][0]]
                    else:
                        psi = self.apply_op(psi, observable_op, error_tolerance)
                        if self.calc_magnitude(psi) < 1e-25:
                            psi = None

                if traj_idx[0].size > 0:
                    self.expectation_cached_trajectories(last_cached_node, temp_trajectories)
                    psi = last_cached_node.trajectories[traj_idx[0][0]] 
                elif psi != None:
                    psi = self.calc_inner_product(psi_temp, psi)
                # if psi != None:
                #     last_cached_node.trajectories[traj_idx[0][0]] = temp_state.H @ psi
                # else:
                #     last_cached_node.trajectories[traj_idx[0][0]] = 0
        if psi == None: return 0
        return psi
