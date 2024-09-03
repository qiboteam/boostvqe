#!/bin/bash
#SBATCH --job-name=boost
#SBATCH --output=boost_powell.log

OPTIMIZATION_METHOD="Powell"
OPTIMIZATION_CONFIG="{ \"maxiter\": 5}"
QIBO_LOG_LEVEL=4 python3 single_commutator_boosting.py   --path "results/XXZ_hw/3_5_layers/sgd_10q_3l_1" \
                                        --epoch 100 --steps 3 --optimization_method $OPTIMIZATION_METHOD \
                                        --optimization_config "$OPTIMIZATION_CONFIG"
