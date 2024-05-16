#!/bin/bash
#SBATCH --job-name=bvqe5q20l
#SBATCH --output=boostvqe5_20l_Powell.out


python main.py --nqubits 5 --seed 4242 --nlayers 4 --optimizer Powell --output_folder proper_simulation --backend numpy  --boost_frequency 100
