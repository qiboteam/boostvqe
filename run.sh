#!/bin/bash
#SBATCH --job-name=adamch
#SBATCH --output=bp_regime.log

NQUBITS=10
NLAYERS=10

DBI_STEPS=2
NBOOST=2
BOOST_FREQUENCY=100

NSHOTS=1000
SEED=42

OPTIMIZER="sgd"
BACKEND="tensorflow"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adam\", \"learning_rate\": 0.01, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/debugging --backend $BACKEND \
                --dbi_step $DBI_STEPS --seed $SEED \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST \
                --optimizer_options "$OPTIMIZER_OPTIONS"
