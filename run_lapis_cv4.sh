#!/bin/bash

python train_nima_lapis.py

python train_histonet_latefusion_lapis.py --trainset GIAA
python train_histonet_latefusion_lapis.py --trainset sGIAA
python train_histonet_latefusion_lapis.py --trainset PIAA

# NIMA
python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 1
python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 2
python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 3
python train_nima_lapis.py --use_cv --n_fold 4 --fold_id 4

python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 1 --use_cv --trainset GIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 2 --use_cv --trainset GIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 3 --use_cv --trainset GIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 4 --use_cv --trainset GIAA

python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 1 --use_cv --trainset sGIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 2 --use_cv --trainset sGIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 3 --use_cv --trainset sGIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 4 --use_cv --trainset sGIAA

python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 1 --use_cv --trainset PIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 2 --use_cv --trainset PIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 3 --use_cv --trainset PIAA
python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 4 --use_cv --trainset PIAA

