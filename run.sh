#!/bin/bash
#SBATCH --job-name=dbi8q1l
#SBATCH --output=dbi_8q1l.log

NQUBITS=8
NLAYERS=1
DBI_STEPS=2
NBOOST=1
NSHOTS=10000
OPTIMIZER="BFGS"
TOL=0.00001
BOOST_FREQUENCY=10
ACC=0.1

python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                --output_folder results/pure_vqe --backend numpy --tol $TOL \
                --dbi_step $DBI_STEPS --seed 42 \
                --boost_frequency $BOOST_FREQUENCY --accuracy $ACC --nboost $NBOOST

                
