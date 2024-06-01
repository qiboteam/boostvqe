#!/bin/bash
#SBATCH --job-name=boostvqe
#SBATCH --output=boostvqe.log

NQUBITS=4
NLAYERS=2

DBI_STEPS=2
NBOOST=2
BOOST_FREQUENCY=10

NSHOTS=10000
TOL=1e-8
ACC=0.5

OPTIMIZER="sgd"
BACKEND="tensorflow"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adagrad\", \"learning_rate\": 0.1, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/debugging --backend $BACKEND --tol $TOL \
                --dbi_step $DBI_STEPS --seed 42 \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST \
                --optimizer_options "$OPTIMIZER_OPTIONS"
