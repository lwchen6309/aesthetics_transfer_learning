#!/bin/bash
# run_script="train_nima_lapis.py" # Uncomment if needed
# run_script="train_piaa_mir_lapis.py" # Uncomment if needed
run_script="compare_traitsample_lapis.py"
training_args="--trainset GIAA --no_log --is_eval"

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
# values=("0.0" "1.0" "2.0" "3.0" "4.0" "5.0" "6.0")
# Loop through each trait
for trait in "${traits[@]}"; do
    # Nested loop through each value
    for value in ${values[@]}; do
        python $run_script --trait $trait --value $value $training_args
    done
done


# Gini Index Computation
run_script="compute_gini_index_lapis.py"
training_args="--trainset GIAA"

# 1. Gender-based Gini Index Computation
echo "Computing Gini Index based on gender..."
gender_values=("male" "female")
python $run_script --trait demo_gender --values "${gender_values[@]}" $training_args

# 2. Education-based Gini Index Computation
echo "Computing Gini Index based on education level..."
education_values=("primary education" "secondary education" "Bachelor's or equivalent" "Master's or equivalent" "Doctorate")
python $run_script --trait demo_edu --values "${education_values[@]}" $training_args

# 3. Age-based Gini Index Computation
echo "Computing Gini Index based on age groups..."
age_values=("18-27" "28-38" "39-49" "50-60" "61-71")
python $run_script --trait age --values "${age_values[@]}" $training_args

# 4. VAIAK trait-based Gini Index Computation
echo "Computing Gini Index based on VAIAK traits..."
traits=("VAIAK1" "VAIAK2" "VAIAK3" "VAIAK4" "VAIAK5" "VAIAK6" "VAIAK7" "2VAIAK1" "2VAIAK2" "2VAIAK3" "2VAIAK4")
values=("0.0" "1.0")
# Loop through traits, but pass all values at once for each trait
for trait in "${traits[@]}"; do
    echo "Computing Gini Index for trait: $trait"
    python $run_script --trait $trait --values "${values[@]}" $training_args
done

echo "Gini Index computations completed."
