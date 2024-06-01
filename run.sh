#!/bin/bash
#SBATCH --job-name=adamch
#SBATCH --output=bp_regime.log

NQUBITS=5
NLAYERS=1

DBI_STEPS=0
NBOOST=0
BOOST_FREQUENCY=100

NSHOTS=1000
SEED=42

OPTIMIZER="sgd"
BACKEND="tensorflow"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adam\", \"learning_rate\": 0.1, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"
DECAY_RATE_LR=0.05

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/debugging_decay --backend $BACKEND \
                --dbi_step $DBI_STEPS --seed $SEED \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST \
                --optimizer_options "$OPTIMIZER_OPTIONS" \
                --decay_rate_lr $DECAY_RATE_LR
