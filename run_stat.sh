#!/bin/bash
#SBATCH --job-name=vqedbi
#SBATCH --output=hybrid.log

NQUBITS=8
DBI_STEPS=2
NBOOST=1
NSHOTS=10000
OPTIMIZER="Powell"
TOL=0.0000000001
BOOST_FREQUENCY=20
ACC=0.1


for NLAYERS in $(seq 1 1 5); do
    python main.py  --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                    --output_folder "results/hybrid" --backend numpy --tol $TOL \
                    --dbi_step $DBI_STEPS --seed 42 \
                    --boost_frequency $BOOST_FREQUENCY --accuracy $ACC --nboost $NBOOST
done

