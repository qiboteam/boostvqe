#!/bin/bash

# SLURM directives
#SBATCH --job-name=VQE12qxDBQA
#SBATCH --output=boost_%A_%a.log
#SBATCH --array=0-29  # Adjusted dynamically based on the number of tasks.

OPTIMIZATION_METHOD="cma"
OPTIMIZATION_CONFIG="{ \"maxiter\": 50}"

# Define the epoch values
EPOCHS=(100 200 300)

# Dynamically calculate the folders
FOLDERS=($(ls -d ./results/scaling_data/test_nqubits_12/*/))
NUM_FOLDERS=${#FOLDERS[@]}
NUM_EPOCHS=${#EPOCHS[@]}

if [ $NUM_FOLDERS -eq 0 ]; then
    echo "No folders found in ./results/scaling_data/test_nqubits_12. Exiting."
    exit 1
fi

# Total number of tasks
TOTAL_TASKS=$((NUM_FOLDERS * NUM_EPOCHS))

# Adjust SLURM array size dynamically
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_TASKS" ]; then
    echo "Task ID exceeds available tasks. Exiting."
    exit 1
fi

# Determine the folder and epoch for this task
FOLDER_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_EPOCHS))
EPOCH_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_EPOCHS))

FOLDER=${FOLDERS[$FOLDER_INDEX]}
EPOCH=${EPOCHS[$EPOCH_INDEX]}

echo "Processing folder: $FOLDER with epoch: $EPOCH"

# Run the Python script for this folder and epoch
python3 load_vqe_and_rotate.py      --backend numpy \
                                    --path "$FOLDER" \
                                    --epoch $EPOCH --steps 3 --optimization_method $OPTIMIZATION_METHOD \
                                    --optimization_config "$OPTIMIZATION_CONFIG"
