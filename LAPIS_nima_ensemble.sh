#!/bin/bash
run_script="train_nima_lapis_traitsample_ensemble.py"



exp_dir="LAPIS_trait_joint_exp/"
# ls $exp_dir

eduPrim=$exp_dir"lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_crimson-glade-377demo_edu_primary_education.pth"
eduSec=$exp_dir"lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_dainty-sunset-378demo_edu_secondary_education.pth"
eduBach=$exp_dir"lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_dulcet-snowflake-379demo_edu_Bachelors_or_equivalent.pth"
eduMaster=$exp_dir"lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_sunny-jazz-380demo_edu_Masters_or_equivalent.pth"
eduDoc=$exp_dir"lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_solar-sky-381demo_edu_Doctorate.pth"

# ls $eduPrim
# ls $eduSec
# ls $eduBach
# ls $eduMaster
# ls $eduDoc

# "lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_rose-wave-375demo_gender_non-binary.pth"
# "lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_rural-resonance-374demo_gender_female.pth"
# "lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_dainty-universe-373demo_gender_male.pth"

# Educational Level
python $run_script --trait demo_edu --value "primary education" --model_paths "$eduPrim" "$eduSec" "$eduBach" "$eduMaster" "$eduDoc"
python $run_script --trait demo_edu --value "secondary education" --model_paths "$eduPrim" "$eduSec" "$eduBach" "$eduMaster" "$eduDoc"
python $run_script --trait demo_edu --value "Bachelor's or equivalent" --model_paths "$eduPrim" "$eduSec" "$eduBach" "$eduMaster" "$eduDoc"
python $run_script --trait demo_edu --value "Master's or equivalent" --model_paths "$eduPrim" "$eduSec" "$eduBach" "$eduMaster" "$eduDoc"
python $run_script --trait demo_edu --value "Doctorate" --model_paths "$eduPrim" "$eduSec" "$eduBach" "$eduMaster" "$eduDoc"
