#!/usr/bin/env sh

# Define array of methods
method_paths=(
    gnn/schnet.yaml
    gnn-ft/schnet.yaml
    mekrr/schnet.yaml
    gap/gap.yaml
    #mekrr-mw/schnet.yaml
)

### First private dirs

# Define array of private directories
private_dirs=(
    1_abinitio-md_Fe45N2
    2_abinitio-metad_Fe45N2
#    2_abinitio-metad_Fe45N2-nonrandom
    3_active-metad_Fe45N2
    4_active-metad_Fe72N2
)

# Loop over all private directories for all methods to run scripts for energies and forces
for private_dir in "${private_dirs[@]}"; do
  for method_path in "${method_paths[@]}"; do

    if [["$method_path" == *"mekrr"*]]; then
        # Run script for energies
        python transfer_learning/main.py \
            --config-yml transfer_learning/configs/s2ef/private/"$private_dir"/energy/"$method_path" \
            --run-dir runs\
            --cpu
    else
        python transfer_learning/main.py \
            --config-yml transfer_learning/configs/s2ef/private/"$private_dir"/energy/"$method_path" \
            --run-dir runs
    fi
    # # Run script for forces
    # python transfer_learning/main.py \
    #   --config-yml transfer_learning/configs/s2ef/private/"$private_dir"/forces/"$method_path" \
    #   --run-dir runs

  done
done

### Now formate dataset
for method_path in "${method_paths[@]}"; do

    if [["$method_path" == *"mekrr"*]]; then
        # Run script for energies
        python transfer_learning/main.py \
            --config-yml transfer_learning/configs/s2ef/formate/energy/"$method_path" \
            --run-dir runs\
            --cpu
    else
        python transfer_learning/main.py \
            --config-yml transfer_learning/configs/s2ef/formate/energy/energy/"$method_path" \
            --run-dir runs
    fi

    # # Run script for forces
    # python transfer_learning/main.py \
        #   --config-yml transfer_learning/configs/s2ef/formate/forces/"$method_path" \
        #   --run-dir runs
done
