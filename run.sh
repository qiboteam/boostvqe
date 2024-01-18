#!/bin/bash

python main.py --nqubits 8 --nlayers 3 --optimizer Powell --output_folder results --backend numpy --optimize_dbi_step true --boost_frequency 20 --nboost 3 --dbi_steps 10
