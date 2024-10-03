# run_script="train_histonet_latefusion_ctloss.py"
# run_script="train_histonet_latefusion.py"
# run_script="train_crossattn.py"
# run_script="train_nima.py"

# run_script="train_piaa_ici.py"
run_script="train_piaa_mir.py"

# python $run_script --is_eval --no_log --resume "models_pth/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_golden-moon-119.pth"

trainparams="--disable_onehot"
giaa_pretrain=models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
# giaa_pretrain=models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_solar-mountain-252.pth
python $run_script --trainset PIAA --use_cv --fold_id 1 --n_fold 4 --pretrained_model $giaa_pretrain $trainparams
# giaa_pretrain=models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_snowy-oath-253.pth
python $run_script --trainset PIAA --use_cv --fold_id 2 --n_fold 4 --pretrained_model $giaa_pretrain $trainparams
# giaa_pretrain=models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_driven-vortex-254.pth
# python $run_script --trainset PIAA --use_cv --fold_id 3 --n_fold 4 --pretrained_model $giaa_pretrain $trainparams
# giaa_pretrain=models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_faithful-surf-255.pth
# python $run_script --trainset PIAA --use_cv --fold_id 4 --n_fold 4 --pretrained_model $giaa_pretrain $trainparams

