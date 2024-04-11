#!/bin/bash
python train_nima_lapis.py --n_fold 4 --fold_id 1

for i in {1..1}
do
    python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 1
    python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 2
    python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 3
    python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 4
    # python train_nima_lapis.py --n_fold 4 --fold_id 1 --no_log --is_eval --resume "model_pth/random_cvs/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_faithful-moon-110.pth"
    # python train_nima_lapis.py --n_fold 4 --fold_id 2 --no_log --is_eval --resume "model_pth/random_cvs/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_misty-jazz-111.pth"
    # python train_nima_lapis.py --n_fold 4 --fold_id 3 --no_log --is_eval --resume "model_pth/random_cvs/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_sunny-sun-112.pth"
    # python train_nima_lapis.py --n_fold 4 --fold_id 4 --no_log --is_eval --resume "model_pth/random_cvs/lapis_best_model_resnet50_nima_lr5e-05_decay_20epoch_glad-dew-113.pth"
done

# for i in {1..1}
# do
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 1
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 2
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 3
#     python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 4
# done