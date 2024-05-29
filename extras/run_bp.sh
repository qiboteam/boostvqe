#!/bin/bash
#SBATCH --job-name=bpwXXZ
#SBATCH --output=bp_diagnostic.log

python bp_diagnostic.py
