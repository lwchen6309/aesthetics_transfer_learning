#!/bin/bash
run_script="compare_traitsample.py"
training_args="--no_log --output_dir pairwise_score_dict"

traits=("gender" "age" "EducationalLevel" "artExperience" "photographyExperience")
values_gender=("male" "female")
values_age=("18-21" "22-25" "26-29" "30-34" "35-40")
values_EducationalLevel=("junior_college" "junior_high_school" "senior_high_school" "technical_secondary_school" "university")
values_artExperience=("beginner" "competent" "proficient" "expert")
values_photographyExperience=("beginner" "competent" "proficient" "expert")

# Create an associative array to map traits to their corresponding values
declare -A trait_values
trait_values=(
    ["gender"]="${values_gender[@]}"
    ["age"]="${values_age[@]}"
    ["EducationalLevel"]="${values_EducationalLevel[@]}"
    ["artExperience"]="${values_artExperience[@]}"
    ["photographyExperience"]="${values_photographyExperience[@]}"
)

# Flatten all (trait, value) pairs
pairs=()
for trait in "${traits[@]}"; do
    for value in ${trait_values[$trait]}; do
        pairs+=("$trait:$value")
    done
done

# Perform pairwise comparisons
for ((i=0; i<${#pairs[@]}; i++)); do
    pair1="${pairs[$i]}"
    trait1="${pair1%%:*}"
    value1="${pair1##*:}"
    for ((j=i+1; j<${#pairs[@]}; j++)); do
        pair2="${pairs[$j]}"
        trait2="${pair2%%:*}"
        value2="${pair2##*:}"
        echo "Comparing ($trait1, $value1) with ($trait2, $value2)"
        python $run_script --trait1 $trait1 --value1 "$value1" --trait2 $trait2 --value2 "$value2" $training_args
    done
done


# Pass all pairs to the EMD computation script
# python task_embedding.py --pairs "${pairs[@]}" --output_dir pairwise_score_dict
