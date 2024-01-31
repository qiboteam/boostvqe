#!/bin/bash

n_start=2
n_end=6

for ((n=$n_start; n<=$n_end; n++)); do
    python train_vqe.py --nqubits 3 --nlayers $n --optimizer Powell --nthreads 16 --output_folder results
done
