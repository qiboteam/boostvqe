#!/bin/bash
#SBATCH --job-name=j1j2_sgc
#SBATCH --output=lighttest_%A_%a.log  
#SBATCH --array=0-999

# Base directory containing the target folders
base_dir="./results/J1J2_HW_many_seeds/3_6_layers"

# Read all directories into an array
dirs=($base_dir/*/)
num_dirs=${#dirs[@]}

echo "Total number of target directories: $num_dirs"

# Define the specific epochs you want to run
epoch_points=(100 200 500 1000 2000)
epochs_per_dir=${#epoch_points[@]}

# Calculate directory index and epoch index
dir_index=$(($SLURM_ARRAY_TASK_ID / $epochs_per_dir))
epoch_index=$(($SLURM_ARRAY_TASK_ID % $epochs_per_dir))

if [ $dir_index -lt $num_dirs ]; then
    dir=${dirs[$dir_index]}
    echo $dir
    if [ -d "$dir" ]; then
        # Get the epoch start from the array of defined epochs
        epoch_start=${epoch_points[$epoch_index]}
        echo "Running job for directory $(basename "$dir") at epoch $epoch_start"
        # Run the Python script with the dynamically set parameters
        python3 single_commutator_boosting.py   --path "$dir" \
                                                --epoch $epoch_start --steps 3 \
                                                --optimization_method "cma" \
                                                --optimization_config "{ \"maxiter\": 50}" \

    fi
else
    echo "Directory index out of range"
fi
