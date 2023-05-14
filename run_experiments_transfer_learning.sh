#!/usr/bin/env sh

# Define array of methods
method_paths=(
    #gnn/schnet.yaml
    #gnn-ft/schnet.yaml
    #gap/gap.yaml
    mekrr/schnet.yaml
    #mekrr-mw/schnet.yaml
)

### Private dirs

# Define array of private directories
private_dirs=(
      1_to_2
      1_to_3
      2_to_3
      2_to_4
      3_to_4
)

# Loop over all private directories for all methods to run scripts for energies and forces
for private_dir in "${private_dirs[@]}"; do
  for method_path in "${method_paths[@]}"; do

    echo ">>> LOG $private_dir - $method_path"
    if [["$method_path" == *"mekrr"*]]; then
        # Run script for energies
        python transfer_learning/main.py \
            --config-yml transfer_learning/configs/s2ef/private/transfer_learning/"$private_dir"/energy/"$method_path" \
            --run-dir runs\
            --cpu
    else
        python transfer_learning/main.py \
            --config-yml transfer_learning/configs/s2ef/private/transfer_learning/"$private_dir"/energy/"$method_path" 
            --run-dir runs
    fi

  done
done
