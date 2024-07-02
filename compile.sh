#!/bin/bash
#SBATCH --job-name=boost
#SBATCH --output=boost.log

OPTIMIZATION_METHOD="cma"
OPTIMIZATION_CONFIG="{ \"maxiter\": 2}"
/home/users/matteo.robbiati/boostvqe/results/moreonXXZ/test
python3 compiling.py --backend numpy --path "./results/XXZ_5seeds/sgd_10q_6l_27/" \
                     --epoch 500 --steps 2 --optimization_method $OPTIMIZATION_METHOD \
                     --optimization_config "$OPTIMIZATION_CONFIG"
