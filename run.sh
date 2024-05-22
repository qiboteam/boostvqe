#!/bin/bash
#SBATCH --job-name=dbi7q1l
#SBATCH --output=dbi_hybrid_7q_1l.log

NQUBITS=7
NLAYERS=1
NSHOTS=10000
OPTIMIZER="Powell"
TOL=0.00001
BOOST_FREQUENCY=50
ACC=0.1

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/routine --backend numpy --tol $TOL \
                --dbi_step 2 --seed 42 \
                --boost_frequency $BOOST_FREQUENCY --accuracy $ACC --nboost 1
