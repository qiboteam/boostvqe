#!/bin/bash
#SBATCH --job-name=bpcheck
#SBATCH --output=bp_diagnostic.out

python bp_diagnostic.py
