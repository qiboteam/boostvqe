#!/bin/bash

n_start=2
n_end=6

for ((n=$n_start; n<=$n_end; n++)); do
    python train.py --nqubits 3 --nlayers $n --optimizer BFGS --nthreads 16 
done
