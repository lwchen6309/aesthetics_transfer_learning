#!/bin/bash
# run_script="train_nima.py"
# run_script="train_nima_attr.py"
# run_script="train_histonet_traitsample.py"
run_script="compare_traitsample.py"
training_args="--lr 5e-5 --no_log"

# Gender
for value in male female; do
    python $run_script --trait gender --value $value $training_args
done

# Age
for value in "18-21" "22-25" "26-29" "30-34" "35-40"; do
    python $run_script --trait age --value "$value" $training_args 
done

# Educational Level
for value in junior_college junior_high_school senior_high_school technical_secondary_school university; do
    python $run_script --trait EducationalLevel --value $value $training_args
done

# Art Experience
for value in beginner competent proficient expert; do
    python $run_script --trait artExperience --value $value $training_args 
done

# Photography Experience
for value in beginner competent proficient expert; do
    python $run_script --trait photographyExperience --value $value $training_args
done

# run_script="train_piaa_mir.py"
# resume="models_pth/lapis_best_model_resnet50_piaamir_still-sponge-1070.pth"
# training_args="--lr 5e-5 --dropout 0.5 --no_log"

# # Gender
# for value in male female; do
#     pretrained_model=$(ls models_pth/trait_disjoint_exp/mir_pretrain/*gender_$value*)
#     python $run_script --trait gender --value $value --pretrained_model $pretrained_model $training_args
# done

# # Age
# for value in "18-21" "22-25" "26-29" "30-34" "35-40"; do
#     pretrained_model=$(ls models_pth/trait_disjoint_exp/mir_pretrain/*age_$value*)
#     python $run_script --trait age --value "$value" --pretrained_model $pretrained_model $training_args 
# done

# # Educational Level
# for value in junior_college junior_high_school senior_high_school technical_secondary_school university; do
#     pretrained_model=$(ls models_pth/trait_disjoint_exp/mir_pretrain/*EducationalLevel_$value*)
#     python $run_script --trait EducationalLevel --value $value --pretrained_model $pretrained_model $training_args
# done

# # Art Experience
# for value in beginner competent proficient expert; do
#     pretrained_model=$(ls models_pth/trait_disjoint_exp/mir_pretrain/*artExperience_$value*)
#     python $run_script --trait artExperience --value $value --pretrained_model $pretrained_model $training_args 
# done

# # Photography Experience
# for value in beginner competent proficient expert; do
#     pretrained_model=$(ls models_pth/trait_disjoint_exp/mir_pretrain/*photographyExperience_$value*)
#     python $run_script --trait photographyExperience --value $value --pretrained_model $pretrained_model $training_args
# done
