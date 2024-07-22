#!/bin/bash
#SBATCH --job-name=gci_shots
#SBATCH --output=gci_shots.log

OPTIMIZATION_METHOD="cma"
OPTIMIZATION_CONFIG="{ \"maxiter\": 5}"

python3 compiling.py --backend numpy --path "./results/moreonXXZ_shots/sgd_10q_8l_27/" \
                     --epoch 200 --steps 2 --optimization_method $OPTIMIZATION_METHOD \
                     --optimization_config "$OPTIMIZATION_CONFIG" --nshots 1000 \
