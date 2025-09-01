#!/bin/bash
# run_script="train_nima.py"
# run_script="train_piaa_mir.py"
run_script="compare_traitsample.py"
training_args="--trainset GIAA"

# python $run_script --trait gender --value male $training_args

# Gender
# for value in male female; do
#     python $run_script --trait gender --value $value $training_args
# done

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

run_script="compute_gini_index.py"
training_args="--trainset GIAA"

# Gender
python $run_script --trait gender --values male female $training_args
# Age
python $run_script --trait age --values "18-21" "22-25" "26-29" "30-34" "35-40" $training_args
# Educational Level
python $run_script --trait EducationalLevel --values junior_college junior_high_school senior_high_school technical_secondary_school university $training_args
# Art Experience
python $run_script --trait artExperience --values beginner competent proficient expert $training_args
# Photography Experience
python $run_script --trait photographyExperience --values beginner competent proficient expert $training_args
