from quimb.tensor import MatrixProductOperator as mpo
from trajectree.trajectory import quantum_channel, trajectory_evaluator
import numpy as np
from qutip_qip.operations import H, CNOT, S, T, X, Z, CZ, RY, RZ
from .noise_models import amplitude_damping, phase_damping
from trajectree.fock_optics.utils import create_vacuum_state
from qutip import basis, tensor
import time
from scipy import sparse as sp
import copy
from functools import lru_cache



class Circuit:
    def __init__(self, num_qubits=-1, backend = 'tensor'):
        self.num_qubits = num_qubits
        self.quantum_channel_list = []
        self.expectation_ops = []
        self.backend = backend

        if num_qubits > 0:
            trajectree_init = [sp.eye(2)]
            self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = num_qubits, formalism = "kraus", kraus_ops_tuple = ((0,), trajectree_init), backend = self.backend, name = "trajectree_init"))
            if self.backend ==  'tensor':
                self.psi = create_vacuum_state(num_qubits, N=2)
            elif self.backend == 'statevector':
                self.psi = tensor([basis(2, 0)] * self.num_qubits)

    def create_trajectree(self, cache_size=2, max_cache_nodes=-1):
        self.t_eval = trajectory_evaluator(self.quantum_channel_list, cache_size = cache_size, max_cache_nodes = max_cache_nodes, backend = self.backend, calc_expectation = True)

    def H_gate(self, idx, is_expectation = False, tag = "H gate"):
        if self.backend == 'statevector':
            quantum_channel_H = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([idx], H(0).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == "tensor":
            dense_op = H(0).get_compact_qobj().full()
            H_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
            quantum_channel_H = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = H_MPO, backend = self.backend, name = tag)
        
        if is_expectation:
            self.expectation_ops.append(quantum_channel_H)
        else:
            self.quantum_channel_list.append(quantum_channel_H)

    def CNOT_gate(self, control_idx, target_idx, tag = "CNOT gate"):
        if self.backend == 'statevector':
            quantum_channel_CX = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([control_idx, target_idx], CNOT(0,1).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == 'tensor':
            dense_op = CNOT(0,1).get_compact_qobj().full()
            CNOT_MPO = mpo.from_dense(dense_op, dims = 2, sites = (control_idx, target_idx), L=self.num_qubits, tags=tag)
            quantum_channel_CX = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = CNOT_MPO, backend = self.backend, name = tag)

        self.quantum_channel_list.append(quantum_channel_CX)

    
    def CZ_gate(self, control_idx, target_idx, tag = "CZ gate"):
        if self.backend == 'statevector':
            quantum_channel_CZ = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([control_idx, target_idx], CZ(0,1).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == 'tensor':
            dense_op = CZ(0,1).get_compact_qobj().full()
            CZ_MPO = mpo.from_dense(dense_op, dims = 2, sites = (control_idx, target_idx), L=self.num_qubits, tags=tag)
            quantum_channel_CZ = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = CZ_MPO, backend = self.backend, name = tag)

        self.quantum_channel_list.append(quantum_channel_CZ)


    def S_gate(self, idx, is_expectation = False, tag = "S gate"):
        if self.backend == 'statevector':
            quantum_channel_S = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([idx], S(0).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == 'tensor':
            dense_op = S(0).get_compact_qobj().full()
            S_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
            quantum_channel_S = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = S_MPO, backend = self.backend, name = tag)

        if is_expectation:
            self.expectation_ops.append(quantum_channel_S)
        else:
            self.quantum_channel_list.append(quantum_channel_S)

    
    def RY_gate(self, theta, idx, is_expectation = False, tag = "RY gate"):
        if self.backend == 'statevector':
            quantum_channel_RY = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([idx], RY(0, theta).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == 'tensor':
            dense_op = RY(0, theta).get_compact_qobj().full()
            RY_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
            quantum_channel_RY = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = RY_MPO, backend = self.backend, name = tag)

        if is_expectation:
            self.expectation_ops.append(quantum_channel_RY)
        else:
            self.quantum_channel_list.append(quantum_channel_RY)

    def RZ_gate(self, theta, idx, is_expectation = False, tag = "RZ gate"):
        if self.backend == 'statevector':
            quantum_channel_RZ = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([idx], RZ(0, theta).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == 'tensor':
            dense_op = RZ(0, theta).get_compact_qobj().full()
            RZ_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
            quantum_channel_RZ = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = RZ_MPO, backend = self.backend, name = tag)

        if is_expectation:
            self.expectation_ops.append(quantum_channel_RZ)
        else:
            self.quantum_channel_list.append(quantum_channel_RZ)

    def T_gate(self, idx, is_expectation = False, tag = "T gate"):
        if self.backend == 'statevector':
            quantum_channel_T = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([idx], T(0).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == 'tensor':
            dense_op = T(0).get_compact_qobj().full()
            T_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
            quantum_channel_T = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = T_MPO, backend = self.backend, name = tag)
        
        if is_expectation:
            self.expectation_ops.append(quantum_channel_T)
        else:
            self.quantum_channel_list.append(quantum_channel_T)


    def X_gate(self, idx, is_expectation = False, tag = "X gate"):
        if self.backend == "statevector":
            quantum_channel_X = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([idx], X(0).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == "tensor":
            dense_op = X(0).get_compact_qobj().full()
            X_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
            quantum_channel_X = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = X_MPO, backend = self.backend, name = tag)
        
        if is_expectation:
            self.expectation_ops.append(quantum_channel_X)
        else:
            self.quantum_channel_list.append(quantum_channel_X)


    def Z_gate(self, idx, is_expectation = False, tag = "X gate"):
        if self.backend == 'statevector':
            quantum_channel_Z = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = ([idx], Z(0).get_compact_qobj()), backend = self.backend, name = tag)
        elif self.backend == 'tensor':
            dense_op = Z(0).get_compact_qobj().full()
            Z_MPO = mpo.from_dense(dense_op, dims = 2, sites = (idx,), L=self.num_qubits, tags=tag)
            quantum_channel_Z = quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "closed", unitary_op = Z_MPO, backend = self.backend, name = tag)
        
        if is_expectation:
            self.expectation_ops.append(quantum_channel_Z)
        else:
            self.quantum_channel_list.append(quantum_channel_Z)

    def amplitude_damping(self, noise_parameter, idx, tag = "amplitude_damping"):
        qc = Circuit.generate_amplitude_damping_quantum_channel(self.num_qubits, self.backend, noise_parameter, idx, tag = tag)
        self.quantum_channel_list.append(qc)

    @lru_cache(maxsize=100)
    @staticmethod
    def generate_amplitude_damping_quantum_channel(num_qubits, backend, noise_parameter, idx, tag = "amplitude_damping"):
        damping_channels = amplitude_damping(noise_parameter = noise_parameter)
        return quantum_channel(N = 2, num_modes = num_qubits, formalism = "kraus", kraus_ops_tuple = ((idx,), damping_channels), backend = backend, name = tag)
        

    def phase_damping(self, noise_parameter, idx, tag = "phase_damping"):
        damping_channels = phase_damping(noise_parameter = noise_parameter)
        self.quantum_channel_list.append(quantum_channel(N = 2, num_modes = self.num_qubits, formalism = "kraus", kraus_ops_tuple = ((idx,), damping_channels), backend = self.backend, name = tag))
    
    def perform_trajectree_simulation(self, num_simulations, error_tolerance = 1e-12, verbose = False):
        times = []
        evs = []
        for _ in range(num_simulations): 
            start = time.time()
            evs.append(self.t_eval.perform_simulation(self.psi, error_tolerance, normalize = True))
        
            time_taken = time.time() - start
            # print("time taken:", time_taken)
            if verbose:
                print("time taken:", time_taken)
            #     if i in progress:
            #         print(f"Completed {progress.index(i)+1}0% of simulations")
            times.append(time_taken)
        # print("done with simulations")
        
        return evs, times

    def qiskit_to_trajectree(self, qc, noise_parameter):
        self.__init__(qc.num_qubits, backend = self.backend) 

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

            elif gate_name == 'z':
                self.Z_gate(qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)

            elif gate_name == 'cx':
                self.CNOT_gate(qargs[0]._index, qargs[1]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[1]._index)

            elif gate_name == 'cz':
                self.CZ_gate(qargs[0]._index, qargs[1]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[1]._index)

            elif gate_name == 'ry':
                self.RY_gate(instr.params[0], qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)

            elif gate_name == 'rz':
                self.RZ_gate(instr.params[0], qargs[0]._index)
                self.amplitude_damping(noise_parameter=noise_parameter, idx = qargs[0]._index)

        self.create_trajectree()

    def expectation(self, ops):
        for i in range(len(ops)):
            if ops[i] == 'x':
                self.X_gate(i, is_expectation = True)

            elif ops[i] == 'h':
                self.H_gate(i, is_expectation = True)

            elif ops[i] == 's':
                self.S_gate(i, is_expectation = True)

            elif ops[i] == 't':
                self.T_gate(i, is_expectation = True)

            elif ops[i] == 'z':
                self.Z_gate(i, is_expectation = True)
            else:
                continue
        self.t_eval.observable_ops = self.expectation_ops
        self.t_eval.calc_expectation = True

