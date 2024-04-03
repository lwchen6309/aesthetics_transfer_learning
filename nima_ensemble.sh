#!/bin/bash
run_script="train_nima_traitsample_ensemble.py"

exp_dir=trait_joint_exp/

age18="best_model_resnet50_nima_lr5e-05_decay_20epoch_cosmic-resonance-617_age_18-21.pth"
age22="best_model_resnet50_nima_lr5e-05_decay_20epoch_dazzling-sun-618_age_22-25.pth"
age26="best_model_resnet50_nima_lr5e-05_decay_20epoch_fanciful-wildflower-619_age_26-29.pth"
age30="best_model_resnet50_nima_lr5e-05_decay_20epoch_vibrant-grass-620_age_30-34.pth"
age35="best_model_resnet50_nima_lr5e-05_decay_20epoch_balmy-sun-621_age_35-40.pth"

eduJun="best_model_resnet50_nima_lr5e-05_decay_20epoch_visionary-tree-622_EducationalLevel_junior_college.pth"
eduJunH="best_model_resnet50_nima_lr5e-05_decay_20epoch_magic-pyramid-623_EducationalLevel_junior_high_school.pth"
eduSenH="best_model_resnet50_nima_lr5e-05_decay_20epoch_valiant-music-624_EducationalLevel_senior_high_school.pth"
eduTech="best_model_resnet50_nima_lr5e-05_decay_20epoch_fast-night-625_EducationalLevel_technical_secondary_school.pth"
eduUni="best_model_resnet50_nima_lr5e-05_decay_20epoch_rare-galaxy-626_EducationalLevel_university.pth"

photoBeg="best_model_resnet50_nima_lr5e-05_decay_20epoch_deep-monkey-627_artExperience_beginner.pth"
photoCom="best_model_resnet50_nima_lr5e-05_decay_20epoch_peachy-serenity-628_artExperience_competent.pth"
photoPro="best_model_resnet50_nima_lr5e-05_decay_20epoch_major-glitter-629_artExperience_proficient.pth"
photoExp="best_model_resnet50_nima_lr5e-05_decay_20epoch_scarlet-capybara-630_artExperience_expert.pth"

artBeg="best_model_resnet50_nima_lr5e-05_decay_20epoch_rare-tree-631_photographyExperience_beginner.pth"
artCom="best_model_resnet50_nima_lr5e-05_decay_20epoch_dulcet-waterfall-632_photographyExperience_competent.pth"
artPro="best_model_resnet50_nima_lr5e-05_decay_20epoch_serene-galaxy-633_photographyExperience_proficient.pth"
artExp="best_model_resnet50_nima_lr5e-05_decay_20epoch_volcanic-grass-634_photographyExperience_expert.pth"


# Age
python $run_script --trait age --value "18-21" --model_paths $exp_dir$age22 $exp_dir$age26 $exp_dir$age30 $exp_dir$age35
python $run_script --trait age --value "22-25" --model_paths $exp_dir$age18 $exp_dir$age26 $exp_dir$age30 $exp_dir$age35
python $run_script --trait age --value "26-29" --model_paths $exp_dir$age18 $exp_dir$age22 $exp_dir$age30 $exp_dir$age35
python $run_script --trait age --value "30-34" --model_paths $exp_dir$age18 $exp_dir$age22 $exp_dir$age26 $exp_dir$age35
python $run_script --trait age --value "35-40" --model_paths $exp_dir$age18 $exp_dir$age22 $exp_dir$age26 $exp_dir$age30

# Educational Level
python $run_script --trait EducationalLevel --value junior_college --model_paths $exp_dir$eduJunH $exp_dir$eduSenH $exp_dir$eduTech $exp_dir$eduUni
python $run_script --trait EducationalLevel --value junior_high_school --model_paths $exp_dir$eduJun $exp_dir$eduSenH $exp_dir$eduTech $exp_dir$eduUni
python $run_script --trait EducationalLevel --value senior_high_school --model_paths $exp_dir$eduJun $exp_dir$eduJunH $exp_dir$eduTech $exp_dir$eduUni
python $run_script --trait EducationalLevel --value technical_secondary_school --model_paths $exp_dir$eduJun $exp_dir$eduJunH $exp_dir$eduSenH $exp_dir$eduUni
python $run_script --trait EducationalLevel --value university --model_paths $exp_dir$eduJun $exp_dir$eduJunH $exp_dir$eduSenH $exp_dir$eduTech

# Photography Experience
python $run_script --trait photographyExperience --value beginner --model_paths $exp_dir$photoCom $exp_dir$photoPro $exp_dir$photoExp
python $run_script --trait photographyExperience --value competent --model_paths $exp_dir$photoBeg $exp_dir$photoPro $exp_dir$photoExp
python $run_script --trait photographyExperience --value proficient --model_paths $exp_dir$photoBeg $exp_dir$photoCom $exp_dir$photoExp
python $run_script --trait photographyExperience --value expert --model_paths $exp_dir$photoBeg $exp_dir$photoCom $exp_dir$photoPro

# Art Experience
python $run_script --trait artExperience --value beginner --model_paths $exp_dir$artCom $exp_dir$artPro $exp_dir$artExp
python $run_script --trait artExperience --value competent --model_paths $exp_dir$artBeg $exp_dir$artPro $exp_dir$artExp
python $run_script --trait artExperience --value proficient --model_paths $exp_dir$artBeg $exp_dir$artCom $exp_dir$artExp
python $run_script --trait artExperience --value expert --model_paths $exp_dir$artBeg $exp_dir$artCom $exp_dir$artPro
