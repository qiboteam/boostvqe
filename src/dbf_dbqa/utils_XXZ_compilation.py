from qibo.backends import construct_backend
from qibo import hamiltonians, Circuit, gates, set_backend
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from copy import deepcopy
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

class XXZ_compilation_line(hamiltonians.SymbolicHamiltonian):
    def __init__(self, L, delta=1.0, boundary="OBC", gateset="CNOT"):

        even_pairs = [(i, (i+1)%(L)) for i in range(0, L-1, 2)]
        odd_pairs = [(i, i+1) for i in range(1, L-1, 2)]
        if boundary == "PBC":
            odd_pairs.append((0, L-1))
        pairs = even_pairs + odd_pairs

        super().__init__(sum([X(i)*X(j) + Y(i)*Y(j) + delta*Z(i)*Z(j) for (i, j) in pairs]))

        self.H_even = SymbolicHamiltonian(sum([X(i)*X(j) + Y(i)*Y(j) + delta*Z(i)*Z(j) for (i, j) in even_pairs]))
        self.H_odd = SymbolicHamiltonian(sum([X(i)*X(j) + Y(i)*Y(j) + delta*Z(i)*Z(j) for (i, j) in odd_pairs]))
        
        self.gateset = gateset
        self.boundary = boundary
        self.even_pairs = even_pairs
        self.odd_pairs = odd_pairs
        self.pairs = pairs
        self.delta = delta

        self.ts_order = 2
    
    def circuit(self,dt):
        c_even = Circuit(self.nqubits)
        c_odd = Circuit(self.nqubits)
        
        if self.ts_order == 1:
            for (i, j) in self.even_pairs:
                self.add_XXZ_term_dt(c_even, i, j, dt)            
            for (i, j) in self.odd_pairs:
                self.add_XXZ_term_dt(c_odd, i, j, dt)
            return c_even + c_odd
        elif self.ts_order == 2:
            for (i, j) in self.even_pairs:
                self.add_XXZ_term_dt(c_even, i, j, dt)            
            for (i, j) in self.odd_pairs:
                self.add_XXZ_term_dt(c_odd, i, j, dt/2)
            return c_odd+c_even + c_odd


    def add_XXZ_term_dt(self,qc,i,j,dt):
        if self.gateset == "CNOT":

            alpha = -dt
            beta = -dt
            gamma = -self.delta * dt

            qc.add(gates.RZ(i, -np.pi / 2))
            qc.add(gates.CNOT(i, j))
            qc.add(gates.RZ(j, -2 * gamma + np.pi / 2))
            qc.add(gates.RY(i, -np.pi / 2 + 2 * alpha))
            qc.add(gates.CNOT(j, i))
            qc.add(gates.RY(i, -2 * beta + np.pi / 2))
            qc.add(gates.CNOT(i, j))
            qc.add(gates.RZ(j, np.pi / 2))

            return qc 
        
        elif self.gateset == "RZZ":
            qc.add(gates.RZZ(i, j, dt))        

            qc.add(gates.H(i)), qc.add(gates.H(j))
            qc.add(gates.RZZ(i, j, dt))
            qc.add(gates.H(i)), qc.add(gates.H(j))        

            qc.add(gates.SDG(i)), qc.add(gates.SDG(j))
            qc.add(gates.H(i)), qc.add(gates.H(j))
            qc.add(gates.RZZ(i, j, dt*self.delta))
            qc.add(gates.H(i)), qc.add(gates.H(j))  
            qc.add(gates.S(i)), qc.add(gates.S(j))
        
            return qc   
        
    def circuit_gnd_approximation_by_adjacent_singlets(self, qc=None):
        """
        Prepare tensor product of singlet states
        """
        if qc is None:
            qc = Circuit(self.nqubits)
        for (a, b) in self.even_pairs:
            qc.add(gates.X(a))
            qc.add(gates.H(a))
            qc.add(gates.X(b))
            qc.add(gates.CNOT(a, b))
        return qc
    
    def XXZ_HVA_circuit(self,t_even,t_odd, qc = None):
        if qc is None:
            qc = self.circuit_gnd_approximation_by_adjacent_singlets()
        for (i,j) in self.odd_pairs:
            self.add_XXZ_term_dt(qc, i, j, t_odd)          
        for (i,j) in self.even_pairs:
            self.add_XXZ_term_dt(qc, i, j, t_even)
                         
        return qc
    
    def find_XXZ_HVA_circuit(self, max_evals = 1000, nlayers = 1, initial_params=None, warm_start_qc = None):
        def HVA_cost_function(parameters):
            qc = warm_start_qc
            for n in range(nlayers):    
                qc = self.XXZ_HVA_circuit(parameters[2*n], parameters[2*n+1], qc=qc)
            return self.expectation(
                self.backend.execute_circuit(circuit=qc).state())

        if initial_params is None:
            initial_params = [0.25] * (nlayers * 2)

        print('Initial loss:', HVA_cost_function(initial_params))
        from scipy.optimize import minimize

        result = minimize(
            HVA_cost_function,
            initial_params,
            method="COBYLA",
            options={"disp": True, "maxiter": max_evals},
            tol=1e-2,
        )

        print(result.fun)
        print(result.x)
        
        parameters = result.x
        qc = warm_start_qc
        for n in range(nlayers):    
            qc = self.XXZ_HVA_circuit(parameters[2*n], parameters[2*n+1], qc=qc)
        return qc, result.fun, result.x
    
    def eigenergies_ED(self):
        """ Calculate the eigenenergies of the XXZ Hamiltonian using exact diagonalization.
        @TODO migrate from qrisp implementation to reduce import load"""""
        from qrisp.operators import X, Y, Z
        from scipy.sparse.linalg import eigsh
        import networkx as nx
        # Define Hamiltonian
        def create_heisenberg_hamiltonian(G):
            H = sum(X(i)*X(j)+Y(i)*Y(j)+delta*Z(i)*Z(j) for (i,j) in G.edges())
            return H

        L = self.nqubits
        delta = self.delta

        G = nx.Graph()
        G.add_edges_from([(k,(k+1)%L) for k in range(L-1)]) 

        H = create_heisenberg_hamiltonian(G)

        M = H.to_sparse_matrix()
        eigenvalues, eigenvectors = eigsh(M, k=2, which='SA')


        E0_val = eigenvalues[0]
        E1_val = eigenvalues[1]
        E0_vec = eigenvectors[:, 0]
        E1_vec = eigenvectors[:, 1]
        return E0_val, E1_val