#!/bin/bash
run_script="train_nima_lapis.py"
# run_script="compare_traitsample_lapis.py"
# resume="models_pth/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_sleek-pine-468.pth"
# training_args=" --resume $resume --num_epochs 5 --lr 5e-7 --trait_joint"
training_args="--no_log"


list=("male" "female")
for value in "${list[@]}"; do
    python $run_script --trait demo_gender --value $value $training_args 
done

list=("primary education" "secondary education" "Bachelor's or equivalent" "Master's or equivalent" "Doctorate")
for value in "${list[@]}"; do
    python $run_script --trait demo_edu --value "$value" $training_args
done

list=("18-27" "28-38" "39-49" "50-60" "61-71")
for value in "${list[@]}"; do
    python $run_script --trait age --value "$value" $training_args
done

# Define an array of traits
traits=("VAIAK1" "VAIAK2" "VAIAK3" "VAIAK4" "VAIAK5" "VAIAK6" "VAIAK7" "2VAIAK1" "2VAIAK2" "2VAIAK3" "2VAIAK4")
values=("0.0" "1.0" "2.0" "3.0" "4.0" "5.0" "6.0")
# Loop through each trait
for trait in "${traits[@]}"; do
    # Nested loop through each value
    for value in ${values[@]}; do
        python $run_script --trait $trait --value $value $training_args
    done
done