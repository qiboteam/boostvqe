import json
import pathlib

import numpy as np
from qibo.hamiltonians import Hamiltonian
from qibo.models import Circuit

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.png"
ROOT_FOLDER = "results"


def generate_path(args):
    return f"./results/{args.optimizer}_{args.nqubits}q_{args.nlayers}l"


def create_folder(path: str):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dump(path: str, results: np.array, output_dict: dict):
    np.save(file=f"{path}/{PARAMS_FILE}", arr=results)
    with open(f"{path}/{OPTIMIZATION_FILE}", "w") as file:
        json.dump(output_dict, file, indent=4)


def json_load(path: str):
    f = open(path)
    return json.load(f)
