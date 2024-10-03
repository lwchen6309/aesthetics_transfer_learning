#!/bin/bash
# run_script="train_histonet_latefusion_lapis_ctloss.py"
# run_script="train_nima_lapis.py"
# run_script="train_piaa_mir_lapis.py"

# python train_nima_lapis.py

# python $run_script --trainset GIAA --resume models_pth/lapis_best_model_resnet50_histo_lr5e-05_decay_20epoch_vocal-glitter-469.pth --no_log --is_eval # GIAA
# python $run_script --trainset GIAA --resume models_pth/lapis_best_model_resnet50_histo_lr5e-05_decay_20epoch_lucky-voice-471.pth --no_log --is_eval # PIAA
run_script="train_piaa_ici_lapis.py"
trainparams="--trainset PIAA --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth"
python $run_script --n_fold 4 --fold_id 1 --use_cv $trainparams
python $run_script --n_fold 4 --fold_id 2 --use_cv $trainparams
python $run_script --n_fold 4 --fold_id 3 --use_cv $trainparams
python $run_script --n_fold 4 --fold_id 4 --use_cv $trainparams

run_script="train_piaa_mir_lapis.py"
python $run_script --n_fold 4 --fold_id 1 --use_cv $trainparams
python $run_script --n_fold 4 --fold_id 2 --use_cv $trainparams
python $run_script --n_fold 4 --fold_id 3 --use_cv $trainparams
python $run_script --n_fold 4 --fold_id 4 --use_cv $trainparams

