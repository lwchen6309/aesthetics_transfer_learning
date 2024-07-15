# run_script="train_histonet_latefusion_ctloss.py"
# run_script="train_histonet_latefusion.py"
run_script="train_crossattn.py"
# run_script="train_nima.py"

# python $run_script --is_eval --no_log --resume "models_pth/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_golden-moon-119.pth"

# python $run_script --trainset GIAA --dropout 0.5
# python $run_script --trainset sGIAA --dropout 0.5

# python $run_script --use_cv --fold_id 1 --n_fold 4 --trainset sGIAA --dropout 0.5
# python $run_script --use_cv --fold_id 2 --n_fold 4 --trainset sGIAA --dropout 0.5
# python $run_script --use_cv --fold_id 3 --n_fold 4 --trainset sGIAA --dropout 0.5
# python $run_script --use_cv --fold_id 4 --n_fold 4 --trainset sGIAA --dropout 0.5

python $run_script --use_cv --fold_id 1 --n_fold 4 --trainset GIAA  --dropout 0.5
# python $run_script --use_cv --fold_id 2 --n_fold 4 --trainset GIAA  --dropout 0.5
# python $run_script --use_cv --fold_id 3 --n_fold 4 --trainset GIAA  --dropout 0.5
# python $run_script --use_cv --fold_id 4 --n_fold 4 --trainset GIAA  --dropout 0.5

# python $run_script --trainset sGIAA --use_cv --fold_id 1 --n_fold 4
# python $run_script --trainset sGIAA --use_cv --fold_id 2 --n_fold 4
# python $run_script --trainset sGIAA --use_cv --fold_id 3 --n_fold 4
# python $run_script --trainset sGIAA --use_cv --fold_id 4 --n_fold 4

# python $run_script --trainset PIAA --use_cv --fold_id 1 --n_fold 4
# python $run_script --trainset PIAA --use_cv --fold_id 2 --n_fold 4
# python $run_script --trainset PIAA --use_cv --fold_id 3 --n_fold 4
# python $run_script --trainset PIAA --use_cv --fold_id 4 --n_fold 4

