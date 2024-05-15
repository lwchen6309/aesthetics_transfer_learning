#!/bin/bash
run_script="train_nima_lapis_traitsample.py"
# run_script="train_histonet_lapis_traitsample.py"


list=("male" "female") # "other/would prefer not to disclose"
for value in "${list[@]}"; do
    python $run_script --trait demo_gender --value $value --is_eval --no_log
done

list=("primary education" "secondary education" "Bachelor's or equivalent" "Master's or equivalent" "Doctorate")
for value in "${list[@]}"; do
    python $run_script --trait demo_edu --value "$value" --is_eval --no_log
done

# Define an array of traits
# traits=("VAIAK1" "VAIAK2" "VAIAK3" "VAIAK4" "VAIAK5" "VAIAK6" "VAIAK7" "2VAIAK1" "2VAIAK2" "2VAIAK3" "2VAIAK4")
# values=("0.0" "1.0" "2.0" "3.0" "4.0" "5.0" "6.0")
# # Loop through each trait
# for trait in "${traits[@]}"; do
#     # Nested loop through each value
#     for value in ${values[@]}; do
#         python $run_script --trait $trait --value $value
#     done
# done