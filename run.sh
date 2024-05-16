#!/bin/bash
#SBATCH --job-name=vqe13q
#SBATCH --output=dbi4bp_13q_powell.log


python main.py --nqubits 13 --nlayers 30 --optimizer Powell \
                --output_folder results/BP_check13q --backend numpy --tol 0.00001 \
                --dbi_step 2 --seed 42 \
                --boost_frequency 100
