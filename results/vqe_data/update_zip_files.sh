#!/bin/bash

# Directory containing the zip files
#DIR="/path/to/your/directory"
#DIR = $(pwd)

# Change to the directory
cd "$(pwd)" || exit

# Loop through each zip file in the directory
for zip_file in *.zip; do
  # Extract the parameters from the zip file name
  # Assuming the naming convention is consistent and the parts are separated by underscores
  IFS='_' read -r optimizer q_value l_value s_value lr_value <<< "${zip_file%.zip}"

  # Extract numeric values from the parameters
  NQUBITS="${q_value%q}"
  NLAYERS="${l_value%l}"
  SEED="${s_value%s}"

  # Initialize the learning rate
  if [[ "$lr_value" == *lr ]]; then
    LEARNING_RATE="${lr_value%lr}"
    HAS_LR=true
  else
    LEARNING_RATE="0.01"  # Default learning rate if not specified
    HAS_LR=false
  fi

  # Define other parameters (adjust as necessary)
  DBI_STEPS=0
  NBOOST=0
  BOOST_FREQUENCY=1000
  NSHOTS=1000
  TOL="1e-8"
  BACKEND="tensorflow"
  OPTIMIZER_OPTIONS="{ \"optimizer\": \"Adagrad\", \"learning_rate\": $LEARNING_RATE, \"nmessage\": 1, \"nepochs\": $BOOST_FREQUENCY }"

  # Construct the content of the new file
  content="#!/bin/bash
#SBATCH --job-name=boostvqe
#SBATCH --output=boostvqe.log

NQUBITS=$NQUBITS
NLAYERS=$NLAYERS

DBI_STEPS=$DBI_STEPS
NBOOST=$NBOOST
BOOST_FREQUENCY=$BOOST_FREQUENCY

NSHOTS=$NSHOTS
TOL=$TOL

OPTIMIZER=\"$optimizer\"
BACKEND=\"$BACKEND\"
OPTIMIZER_OPTIONS=\"$OPTIMIZER_OPTIONS\"

python main.py  --nqubits \$NQUBITS --nlayers \$NLAYERS --optimizer \$OPTIMIZER \\
                --output_folder results/\$OPTIMIZER --backend \$BACKEND --tol \$TOL \\
                --dbi_step \$DBI_STEPS --seed $SEED \\
                --boost_frequency \$BOOST_FREQUENCY --nboost \$NBOOST \\
                --optimizer_options \"\$OPTIMIZER_OPTIONS\""

  # Construct the name of the new file to be added
  if $HAS_LR; then
    new_file="${optimizer}_${NQUBITS}q_${NLAYERS}l_${SEED}s_${LEARNING_RATE}lr.sh"
  else
    new_file="${optimizer}_${NQUBITS}q_${NLAYERS}l_${SEED}s.sh"
  fi
  
  # Create the new file with the content
  echo "$content" > "$new_file"
  
  # Add the new file to the zip file
  zip -u "$zip_file" "$new_file"
  echo "Added $new_file to $zip_file"
  
  # Remove the temporary new file after adding it to the zip
  rm "$new_file"
done