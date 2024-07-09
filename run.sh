#!/bin/bash
#SBATCH --job-name=10q_sgd
#SBATCH --output=sgd_%A_%a.log
#SBATCH --array=0-39  # Total number of combinations minus one

# Define arrays of seeds and layers
SEEDS=(1 2 3 4 42)
LAYERS=(3 4 5 6 7 8 9 10)

# Calculate the number of layers and seeds
NUM_SEEDS=${#SEEDS[@]}
NUM_LAYERS=${#LAYERS[@]}

# Calculate the total number of combinations
TOTAL_COMBINATIONS=$((NUM_SEEDS * NUM_LAYERS))

# Check if the job array index is within bounds
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_COMBINATIONS} ]; then
  echo "Error: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds number of combinations."
  exit 1
fi

# Calculate the seed and layer for the current job
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_LAYERS))
LAYER_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_LAYERS))

SEED=${SEEDS[$SEED_INDEX]}
NLAYERS=${LAYERS[$LAYER_INDEX]}

# Parameters
NQUBITS=10
DBI_STEPS=0
NBOOST=0
BOOST_FREQUENCY=2000
NSHOTS=100

OPTIMIZER="sgd"
BACKEND="tensorflow"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adam\", \"learning_rate\": 0.05, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"
DECAY_RATE_LR=1.

# Run the main.py script with the current parameters
python3 main.py --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/j1j2_hdw_eff --backend $BACKEND \
                --dbi_step $DBI_STEPS --seed $SEED \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST \
                --optimizer_options "$OPTIMIZER_OPTIONS" \
                --hamiltonian "J1J2" --ansatz "hdw_efficient"
