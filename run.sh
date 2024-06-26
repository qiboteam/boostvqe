#!/bin/bash
#SBATCH --job-name=tlfim
#SBATCH --output=tlfim_shots.log

NQUBITS=10
NLAYERS=3

DBI_STEPS=0
NBOOST=0
BOOST_FREQUENCY=100

NSHOTS=500
SEED=42

OPTIMIZER="sgd"
BACKEND="tensorflow"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adam\", \"learning_rate\": 0.01, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"
DECAY_RATE_LR=0.05

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/tlfim_shots --backend $BACKEND \
                --dbi_step $DBI_STEPS --seed $SEED \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST \
                --optimizer_options "$OPTIMIZER_OPTIONS" \
                --hamiltonian "TLFIM" --nshots $NSHOTS 