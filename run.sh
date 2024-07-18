#!/bin/bash
#SBATCH --job-name=10qVQE_jj
#SBATCH --output=10qVQE_%A_%a.log
#SBATCH --array=0-249  # Creates 250 jobs in total

NQUBITS=10

DBI_STEPS=0
NBOOST=0
BOOST_FREQUENCY=2000

NSHOTS=100

# Calculate the seed and number of layers based on SLURM_ARRAY_TASK_ID
# 50 seeds for each of the 5 configurations (3, 4, 5, 6, 7 layers)
SEED=$((42 + SLURM_ARRAY_TASK_ID / 5))
NLAYERS=$((3 + SLURM_ARRAY_TASK_ID % 5))

OPTIMIZER="sgd"
BACKEND="tensorflow"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adam\", \"learning_rate\": 0.05, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"
DECAY_RATE_LR=1.

# Adjust output folder to include both seed and number of layers
python3 main.py --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/J1J2_HW_many_seeds --backend $BACKEND \
                --dbi_step $DBI_STEPS --seed $SEED \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST \
                --optimizer_options "$OPTIMIZER_OPTIONS" \
                --hamiltonian "J1J2" --ansatz "hw_preserving"
