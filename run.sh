#!/bin/bash
sleep 18000

run_script="train_piaa_mir_lapis.py"
training_args="--trainset GIAA --dropout 0.5"

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

traits=("VAIAK1" "VAIAK2" "VAIAK3" "VAIAK4" "VAIAK5" "VAIAK6" "VAIAK7" "2VAIAK1" "2VAIAK2" "2VAIAK3" "2VAIAK4")
values=("0.0" "1.0")
for trait in "${traits[@]}"; do
    # Nested loop through each value
    for value in ${values[@]}; do
        python $run_script --trait $trait --value $value $training_args
    done
done


training_args="--trainset PIAA --dropout 0.5"

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

traits=("VAIAK1" "VAIAK2" "VAIAK3" "VAIAK4" "VAIAK5" "VAIAK6" "VAIAK7" "2VAIAK1" "2VAIAK2" "2VAIAK3" "2VAIAK4")
values=("0.0" "1.0")
for trait in "${traits[@]}"; do
    # Nested loop through each value
    for value in ${values[@]}; do
        python $run_script --trait $trait --value $value $training_args
    done
done



training_args="--trainset PIAA --dropout 0.5 --disable_onehot --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth"

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

traits=("VAIAK1" "VAIAK2" "VAIAK3" "VAIAK4" "VAIAK5" "VAIAK6" "VAIAK7" "2VAIAK1" "2VAIAK2" "2VAIAK3" "2VAIAK4")
values=("0.0" "1.0")
for trait in "${traits[@]}"; do
    # Nested loop through each value
    for value in ${values[@]}; do
        python $run_script --trait $trait --value $value $training_args
    done
done


