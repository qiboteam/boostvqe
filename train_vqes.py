import argparse
import json
import logging

from boostvqe import ansatze
from boostvqe.boost import dbqa_vqe


def main():
    parser = argparse.ArgumentParser(description="Run DBQA VQE optimization.")

    # Required arguments
    parser.add_argument("--output_folder", type=str, help="Output folder for results.")
    parser.add_argument("--circuit_ansatz", type=str, help="Circuit ansatz.")
    parser.add_argument(
        "--nqubits",
        type=int,
        help="Number of particles of the problem. It will fix the number of qubits.",
    )
    parser.add_argument(
        "--nlayers", type=int, help="Number of layers of the VQE ansatz."
    )

    # Optional arguments with defaults matching the function signature
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        help="Quantum backend to use (default: numpy).",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Platform for backend (e.g., 'CUDA').",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Powell",
        help="Optimizer for VQE (default: Powell).",
    )
    parser.add_argument(
        "--optimizer_options",
        type=json.loads,
        default={},
        help="Optimizer options as a JSON string.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="Tolerance for optimization convergence.",
    )
    parser.add_argument(
        "--decay_rate_lr", type=float, default=1.0, help="Decay rate for learning rate."
    )
    parser.add_argument(
        "--nboost", type=int, default=0, help="Number of boost iterations."
    )
    parser.add_argument(
        "--boost_frequency", type=int, default=10, help="Frequency of boosting calls."
    )
    parser.add_argument(
        "--dbi_steps", type=int, default=1, help="Number of DBI iterations."
    )
    parser.add_argument(
        "--store_h", action="store_true", help="Store Hamiltonian at each iteration."
    )
    parser.add_argument(
        "--hamiltonian",
        type=str,
        default="XXZ",
        help="Type of Hamiltonian (default: XXZ).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--nshots", type=int, default=None, help="Number of shots.")
    parser.add_argument(
        "--mode", type=str, default="single_commutator", help="DBI generator mode."
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load or construct the circuit from the given file path
    circuit = getattr(ansatze, args.circuit_ansatz)(args.nqubits, args.nlayers)

    # Run the DBQA VQE function
    dbqa_vqe(
        circuit=circuit,
        output_folder=args.output_folder,
        backend=args.backend,
        platform=args.platform,
        optimizer=args.optimizer,
        optimizer_options=args.optimizer_options,
        tol=args.tol,
        decay_rate_lr=args.decay_rate_lr,
        nboost=args.nboost,
        boost_frequency=args.boost_frequency,
        dbi_steps=args.dbi_steps,
        store_h=args.store_h,
        hamiltonian=args.hamiltonian,
        seed=args.seed,
        nshots=args.nshots,
        nlayers=args.nlayers,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
