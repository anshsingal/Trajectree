from quimb.tensor import MatrixProductOperator as mpo
from trajectree.trajectory import quantum_channel, trajectory_evaluator
import numpy as np
from qutip_qip.operations import H, CNOT, S, T, X
from .noise_models import amplitude_damping, phase_damping
from trajectree.fock_optics.utils import create_vacuum_state
import time
from scipy import sparse as sp


class Circuit:
    def __init__(self, num_qubits=-1):
        self.num_qubits = num_qubits
        self.quantum_channel_list = []

        if num_qubits > 0:
            trajectree_init = [sp.eye(2)]
            self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = num_qubits, formalism = "kraus", kraus_ops_tuple = ((0,), trajectree_init), name = "trajectree_init"))
            
            self.psi = create_vacuum_state(num_qubits, N=2)

    def create_trajectree(self, cache_size=1, max_cache_nodes=-1):
        self.t_eval = trajectory_evaluator(self.quantum_channel_list, cache_size = cache_size, max_cache_nodes = max_cache_nodes)

    def H_gate(self, idx, tag = "H gate"):
        dense_op = H(0).get_compact_qobj().full()
        H_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
        self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_MPOs = H_MPO, name = tag))

    def CNOT_gate(self, control_idx, target_idx, tag = "CNOT gate"):
        dense_op = CNOT(0,1).get_compact_qobj().full()
        CNOT_MPO = mpo.from_dense(dense_op, dims = 2, sites = (control_idx, target_idx), L=self.num_qubits, tags=tag)
        self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_MPOs = CNOT_MPO, name = tag))

    def S_gate(self, idx, tag = "S gate"):
        dense_op = S(0).get_compact_qobj().full()
        S_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
        self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_MPOs = S_MPO, name = tag))

    def T_gate(self, idx, tag = "T gate"):
        dense_op = T(0).get_compact_qobj().full()
        T_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
        self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_MPOs = T_MPO, name = tag))

    def X_gate(self, idx, tag = "X gate"):
        dense_op = X(0).get_compact_qobj().full()
        X_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
        self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_MPOs = X_MPO, name = tag))

    def amplitude_damping(self, noise_parameter, idx, tag = "amplitude_damping"):
        damping_channels = amplitude_damping(noise_parameter = noise_parameter)
        self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "kraus", kraus_ops_tuple = ((idx,), damping_channels), name = tag))

    def phase_damping(self, noise_parameter, idx, tag = "phase_damping"):
        damping_channels = phase_damping(noise_parameter = noise_parameter)
        self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "kraus", kraus_ops_tuple = ((idx,), damping_channels), name = tag))
    
    def perform_trajectree_simulation(self, num_simulations, error_tolerance = 1e-10):
        times = []
        for _ in range(num_simulations): 
            start = time.time()
            psi_iter = self.t_eval.perform_simulation(self.psi, error_tolerance, normalize = True)
        
            time_taken = time.time() - start
            # if verbose:
            #     if i in progress:
            #         print(f"Completed {progress.index(i)+1}0% of simulations")
            times.append(time_taken)
        
        return times

    def qiskit_to_trajectree(self, qc, noise_parameter):
        self.__init__(qc.num_qubits) 

        for circuit_instr in qc.data:
            instr = circuit_instr.operation
            qargs = circuit_instr.qubits
            cargs = circuit_instr.clbits
            gate_name = instr.name
            
            if gate_name == 'x':
                self.X_gate(qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)

            elif gate_name == 'h':
                self.H_gate(qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)

            elif gate_name == 's':
                self.S_gate(qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)

            elif gate_name == 't':
                self.T_gate(qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)

            elif gate_name == 'cx':
                self.CNOT_gate(qargs[0]._index, qargs[1]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[1]._index)

        self.create_trajectree()