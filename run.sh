#!/bin/bash
#SBATCH --job-name=adamch
#SBATCH --output=bp_regime.log

NQUBITS=11
NLAYERS=20

DBI_STEPS=0
NBOOST=0
BOOST_FREQUENCY=500

NSHOTS=1000
SEED=42

OPTIMIZER="sgd"
BACKEND="tensorflow"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adam\", \"learning_rate\": 0.005, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"
DECAY_RATE_LR=1.

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/big_architectures_small_lr --backend $BACKEND \
                --dbi_step $DBI_STEPS --seed $SEED \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST \
                --optimizer_options "$OPTIMIZER_OPTIONS" \
                --decay_rate_lr $DECAY_RATE_LR
