#!/bin/bash
#SBATCH --job-name=boost
#SBATCH --output=boost.log

OPTIMIZATION_METHOD="sgd"
OPTIMIZATION_CONFIG="{ \"gd_epochs\": 2}"

python3 compiling.py --backend numpy --path "./results/XXZ_5seeds/sgd_10q_6l_27/" \
                     --epoch 500 --steps 2 --optimization_method $OPTIMIZATION_METHOD \
                     --optimization_config "$OPTIMIZATION_CONFIG"
