#!/bin/bash

python main.py --nqubits 6 --nlayers 1 --optimizer Powell --output_folder results --backend numpy  --boost_frequency 10 --nboost 2 --dbi_steps 3
