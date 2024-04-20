#!/bin/bash
python train_histonet_latefusion_lapis.py --trainset GIAA
python train_histonet_latefusion_lapis.py --trainset sGIAA
python train_histonet_latefusion_lapis.py --trainset PIAA

# for i in {1..1}
# do
#     python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 1 --is_eval --no_log --resume models_pth/random_cvs/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_zany-yogurt-430.pth
#     python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 2 --is_eval --no_log --resume models_pth/random_cvs/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_dutiful-terrain-431.pth
#     python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 3 --is_eval --no_log --resume models_pth/random_cvs/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_spring-wood-432.pth
#     python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 4 --is_eval --no_log --resume models_pth/random_cvs/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_fancy-morning-433.pth
# done

# for i in {1..1}
# do
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 1 --use_cv --trainset sGIAA
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 2 --use_cv --trainset sGIAA
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 3 --use_cv --trainset sGIAA
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 4 --use_cv --trainset sGIAA
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 1 --use_cv --trainset sGIAA --importance_sampling
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 2 --use_cv --trainset sGIAA --importance_sampling 
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 3 --use_cv --trainset sGIAA --importance_sampling 
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 4 --use_cv --trainset sGIAA --importance_sampling 
# done