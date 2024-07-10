#!/bin/bash
#SBATCH --job-name=boost
#SBATCH --output=boost.log

OPTIMIZATION_METHOD="cma"
OPTIMIZATION_CONFIG="{ \"maxiter\": 5}"
QIBO_LOG_LEVEL=4 python3 single_commutator_boosting.py   --path "j1j2_hw_preserving/3_7_layers/sgd_10q_7l_1" \
                                        --epoch 100 --steps 3 --optimization_method $OPTIMIZATION_METHOD \
                                        --optimization_config "$OPTIMIZATION_CONFIG"
