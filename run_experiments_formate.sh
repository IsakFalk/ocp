#!/usr/bin/env sh

# Define array of methods
method_paths=(
    mekrr/schnet.yaml
    #gnn/schnet.yaml
    #gnn-ft/schnet.yaml
    #gap/gap.yaml
    #mekrr-mw/schnet.yaml
)

### Private dirs

for method_path in "${method_paths[@]}"; do

     echo ">>> LOG $method_path"
     if [[ "$method_path" == *"mekrr"* ]]; then
         # Run script for energies
         CUDA_VISIBLE_DEVICES=-1 python transfer_learning/main.py \
                      --config-yml transfer_learning/configs/s2ef/formate/energy/"$method_path" \
             --run-dir runs \
             --cpu
     else
         python transfer_learning/main.py \
                      --config-yml transfer_learning/configs/s2ef/formate/energy/"$method_path" \
             --run-dir runs
     fi
 done