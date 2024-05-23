#!/bin/bash
#SBATCH --job-name=boostvqe
#SBATCH --output=boostvqe.log

NQUBITS=4
NLAYERS=1
DBI_STEPS=0
NBOOST=0
NSHOTS=10000
OPTIMIZER="Powell"
TOL=1e-16
BOOST_FREQUENCY=500000

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/pure --backend numpy --tol $TOL \
                --dbi_step $DBI_STEPS --seed 42 \
                --boost_frequency $BOOST_FREQUENCY --nboost $NBOOST
