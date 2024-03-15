#!/bin/bash
#run_script="train_histonet_attr_latefusion_traitsample.py"
run_script="train_nima_traitsample.py"

python $run_script --trait gender --value male
python $run_script --trait gender --value female

python $run_script --trait age --value "18-21"
python $run_script --trait age --value "22-25"
python $run_script --trait age --value "26-29"
python $run_script --trait age --value "30-34"
python $run_script --trait age --value "35-40"

python $run_script --trait EducationalLevel --value junior_college
python $run_script --trait EducationalLevel --value junior_high_school
python $run_script --trait EducationalLevel --value senior_high_school
python $run_script --trait EducationalLevel --value technical_secondary_school
python $run_script --trait EducationalLevel --value university

python $run_script --trait artExperience --value beginner
python $run_script --trait artExperience --value competent
python $run_script --trait artExperience --value proficient
python $run_script --trait artExperience --value expert

python $run_script --trait photographyExperience --value beginner
python $run_script --trait photographyExperience --value competent
python $run_script --trait photographyExperience --value proficient
python $run_script --trait photographyExperience --value expert
