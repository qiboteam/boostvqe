#!/bin/bash
#SBATCH --output=output_%A_%a.out      # Output file for each array task
#SBATCH --error=error_%A_%a.err        # Error file for each array task
#SBATCH --array=0-59                   # Total tasks: 6 nqubits * 10 seeds = 60

# Define variables for the parameters
CIRCUIT_ANSAZ="hw_preserving"
NLAYERS=5
BACKEND="qiboml"
PLATFORM="tensorflow"
OPTIMIZER="sgd"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adam\", \"learning_rate\": 0.05, \"nmessage\": 1, \"nepochs\": 400 }"

# Define the nqubits and seed values
NQUBITS_VALUES=(2 4 6 8 10 12)
SEED_VALUES=(0 1 2 3 4 5 6 7 8 9)

# Calculate indices for nqubits and seed based on SLURM_ARRAY_TASK_ID
NQUBITS_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 10))

# Set nqubits and seed based on the calculated indices
NQUBITS=${NQUBITS_VALUES[NQUBITS_INDEX]}
SEED=${SEED_VALUES[SEED_INDEX]}

# Set a specific job name for each task with the current nqubits
#SBATCH --job-name=vqe_${NQUBITS}

# Run the Python script with the specified arguments, including nqubits and seed
python train_vqes.py    --output_folder "results/test_nqubits_${NQUBITS}" --circuit_ansatz "$CIRCUIT_ANSAZ" \
                        --nqubits "$NQUBITS" --nlayers "$NLAYERS" --backend $BACKEND \
                        --platform $PLATFORM --optimizer $OPTIMIZER \
                        --optimizer_options "$OPTIMIZER_OPTIONS" \
                        --seed $SEED
