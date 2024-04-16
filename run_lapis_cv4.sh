#!/bin/bash
# python train_nima_lapis.py --n_fold 4 --fold_id 1

# for i in {1..1}
# do
#     python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 1
#     python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 2
#     python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 3
#     python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 4
# done

for i in {1..1}
do
    python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 1 --use_cv --trainset sGIAA --importance_sampling
    python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 2 --use_cv --trainset sGIAA --importance_sampling 
    python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 3 --use_cv --trainset sGIAA --importance_sampling 
    python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 4 --use_cv --trainset sGIAA --importance_sampling 
done