#!/bin/bash
# run_script="train_nima.py"
run_script="train_nima_attr.py"
# run_script="train_histonet_traitsample.py"
# run_script="compare_traitsample.py"

# resume="models_pth/best_model_resnet50_nima_lr5e-05_decay_20epoch_woven-fire-109.pth"
training_args="--num_epochs 1 --no_log"


# Gender
for value in male female; do
    python $run_script --trait gender --value $value $training_args
done

# # Age
# for value in "18-21" "22-25" "26-29" "30-34" "35-40"; do
#     python $run_script --trait age --value "$value" $training_args 
# done

# # Educational Level
# for value in junior_college junior_high_school senior_high_school technical_secondary_school university; do
#     python $run_script --trait EducationalLevel --value $value $training_args
# done

# # Art Experience
# for value in beginner competent proficient expert; do
#     python $run_script --trait artExperience --value $value $training_args 
# done

# # Photography Experience
# for value in beginner competent proficient expert; do
#     python $run_script --trait photographyExperience --value $value $training_args
# done
