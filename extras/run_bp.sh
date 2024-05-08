#!/bin/bash
#SBATCH --job-name=bp_diagnostic
#SBATCH --output=bp_diagnostic.out

python bp_diagnostic.py
