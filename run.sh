#!/bin/bash
#SBATCH --job-name=boostvqe
#SBATCH --output=boostvqe.log

NQUBITS=2

NLAYERS=1
DBI_STEPS=0
NBOOST=0
NSHOTS=100000
OPTIMIZER="sgd"
BACKEND="tensorflow"
TOL=1e-8
BOOST_FREQUENCY=500

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/debugging --backend $BACKEND --tol $TOL \
                --dbi_step $DBI_STEPS --seed 42 \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST
