#!/bin/bash

## PARA
# trainparams="--trainset GIAA --no_log --is_eval"
# python train_histonet_latefusion.py $trainparams
# python train_piaa_mir.py $trainparams
# python train_piaa_ici.py $trainparams

## LAPIS
trainparams="--trainset GIAA --no_log --is_eval"
python train_nima_lapis.py $trainparams
python train_histonet_latefusion_lapis.py $trainparams
python train_piaa_mir_lapis.py $trainparams
python train_piaa_ici_lapis.py $trainparams

# trainparams="--trainset PIAA --no_log --is_eval"
# python train_histonet_latefusion_lapis.py $trainparams
# python train_piaa_mir_lapis.py $trainparams
# python train_piaa_ici_lapis.py $trainparams

# trainparams="--trainset PIAA --no_log --is_eval"
# python train_piaa_mir_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
# python train_piaa_ici_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth

# trainparams="--trainset PIAA --no_log --is_eval --disable_onehot"
# python train_piaa_mir_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
# python train_piaa_ici_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth