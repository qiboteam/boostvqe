#!/bin/bash
#SBATCH --job-name=boostvqe
#SBATCH --output=boostvqe.log

NQUBITS=7
NLAYERS=3

DBI_STEPS=0
NBOOST=0
BOOST_FREQUENCY=1000

NSHOTS=1000
TOL=1e-8

OPTIMIZER="sgd"
BACKEND="tensorflow"
OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adagrad\", \"learning_rate\": 0.05, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/sgd_exact --backend $BACKEND --tol $TOL \
                --dbi_step $DBI_STEPS --seed 42 \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST \
                --optimizer_options "$OPTIMIZER_OPTIONS"
