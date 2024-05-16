#!/bin/bash
#SBATCH --job-name=nruns.log
#SBATCH --output=boostvqe5_20l_Powell.out

OPTIMIZER="Powell"
TOL=0.0001
NSHOTS=10000
BOOST_FREQUENCY=100
DBI_STEPS=1


for NQUBITS in 4 5; do
    for NLAYERS in 1 2; do
        for i in $(seq 1 25 101); do
            SEED=$i
            python main.py --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                           --output_folder results/shots --backend numpy --tol $TOL \
                           --dbi_step $DBI_STEPS --shot_train --nshots $NSHOTS --seed $SEED \
                           --boost_frequency $BOOST_FREQUENCY
        done
    done
done
