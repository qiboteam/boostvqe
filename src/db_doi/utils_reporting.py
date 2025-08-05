from qibo import gates
def print_gate_count_report(circ):
    """
    Prints a report of the gate counts in the circuit.
    """
    print("Gate count report:")
    print("-------------------")
    print("Number of qubits:", circ.nqubits)
    print("Circuit depth:", circ.depth)
    print("Total gate count:", circ.ngates)
    
    from collections import Counter
    gate_names = [g.__class__.__name__ for g in circ.queue]
    counts = Counter(gate_names)
    
    print("Gate counts:")
    for name, freq in counts.items():
        print(f"  {name}: {freq}")
    
    unitary_1q_count = sum(1 for gate in circ.queue if isinstance(gate, gates.Unitary) and len(gate.qubits) == 1)
    print(f"Number of gates with generic Unitary type acting on exactly 1 qubit: {unitary_1q_count}")
    
    two_qubit_count = sum(1 for gate in circ.queue if len(gate.qubits) == 2)
    print(f"Number of gates acting on exactly 2 qubits: {two_qubit_count}")

def take_verbose_step(self, magnetic_hamiltonian):
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    step_grid = np.linspace(0.0051, 0.05, 15)
    s_opt, s_min, losses, _ = self.choose_step(
        magnetic_hamiltonian, step_grid=step_grid)
    print("Step grid:", step_grid)
    print("Losses:", losses)
    print(f"Optimal step: {s_opt}")
    print(f"Minimum loss: {s_min}")
    plt.figure()
    plt.plot(step_grid, losses, marker='o')
    plt.xlabel('Step size')
    plt.ylabel('Loss')
    plt.title('Loss vs Step size for gci_hva.choose_step')
    plt.show()

    # Circuit analysis    
    print("-----\nCircuit analysis before:")
    print_gate_count_report(self.preparation_circuit)
    
    self(s_opt, magnetic_hamiltonian)
    print("Circuit analysis after:")
    print_gate_count_report(self.preparation_circuit)    

def run_param_rc(fontsize = 30):
        import matplotlib.pyplot as plt
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize
        plt.rcParams['font.size'] = fontsize
        #set_matplotlib_formats('pdf', 'png')
        plt.rcParams['savefig.dpi'] = 75
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 8
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['legend.labelspacing'] = .3
        plt.rcParams['legend.columnspacing']= .3
        plt.rcParams['legend.handletextpad']= .1
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['font.serif'] = "cm"

import pickle

__HVA__training_version = 1.0

def simulation_data_path():
    return f"../simulation_results/Qibo_HVA/training_run_{__HVA__training_version}/"

def save_data(data, filename):  
    # Open a file for writing
    with open(filename+'.pickle', 'wb') as file:
        # Write the object to the file
        pickle.dump(data, file)

def load_data(filename):
    with open(filename+'.pickle', 'rb') as file:
        # Load the object from the file
        data = pickle.load(file)
    return data