import json
from pathlib import Path

import numpy as np

OPTIMIZATION_FILE = "optimization_results.json"
PARAMS_FILE = "parameters_history.npy"
PLOT_FILE = "energy.png"
ROOT_FOLDER = "results"
FLUCTUATION_FILE = "fluctuations"
LOSS_FILE = "energies"
HAMILTONIAN_FILE = "hamiltonian_matrix.npy"


def generate_path(args):
    return f"./{args.output_folder}/{args.optimizer}_{args.nqubits}q_{args.nlayers}l"


def create_folder(path: str):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dump(path: str, results: np.array, output_dict: dict):
    np.save(file=f"{path}/{PARAMS_FILE}", arr=results)
    json_file = Path(f"{path}/{OPTIMIZATION_FILE}")
    dump_json(json_file, output_dict)


def dump_json(path: Path, data):
    path.write_text(json.dumps(data, indent=4))


def json_load(path: str):
    f = open(path)
    return json.load(f)
