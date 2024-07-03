#!/bin/bash
#SBATCH --job-name=boost
#SBATCH --output=boost.log

python3 single_commutator_boosting.py   --path "./results/moreonXXZ/compile_targets/sgd_10q_7l_13" \
                                        --epoch 3000 --steps 1 
