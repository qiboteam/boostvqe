#!/bin/bash
#SBATCH --job-name=boost
#SBATCH --output=boost.log

OPTIMIZATION_METHOD="cma"
OPTIMIZATION_CONFIG="{ \"maxiter\": 5}"
QIBO_LOG_LEVEL=4 python3 single_commutator_boosting.py   --path "XXZ_5seeds/XXZ_5seeds/sgd_10q_7l_42/" \
                                        --epoch 3000 --steps 1 --optimization_method $OPTIMIZATION_METHOD \
                                        --optimization_config "$OPTIMIZATION_CONFIG"
