#!/bin/bash
#SBATCH --job-name=boostvqe_exact
#SBATCH --output=boostvqe_exact.log

OPTIMIZER="Powell"
TOL=0.00001
# NSHOTS=5000
BOOST_FREQUENCY=100
DBI_STEPS=2


# for NQUBITS in $(seq 6 1 11); do
#     for NLAYERS in $(seq 1 1 10); do
#         for i in $(seq 1 5 101); do
#             SEED=$i
#             python main.py --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
#                            --output_folder results/shots_run1 --backend numpy --tol $TOL \
#                            --dbi_step $DBI_STEPS --shot_train --nshots $NSHOTS --seed $SEED \
#                            --boost_frequency $BOOST_FREQUENCY
#         done
#     done
# done


for NQUBITS in $(seq 6 1 11); do
    for NLAYERS in $(seq 1 1 10); do
        for i in $(seq 1 5 101); do
            SEED=$i
            python main.py --nqubits $NQUBITS --nlayers $NLAYERS --optimizer $OPTIMIZER \
                           --output_folder results/exact_run1 --backend numpy --tol $TOL \
                           --dbi_step $DBI_STEPS --seed $SEED \
                           --boost_frequency $BOOST_FREQUENCY
        done
    done
done
