#!/bin/bash

bash run_GIAA.sh
bash run_sGIAA.sh


# trainparams="--trainset PIAA --is_eval --no_log"
# python train_histonet_latefusion_lapis.py $trainparams
# python train_piaa_mir_lapis.py $trainparams
# python train_piaa_ici_lapis.py $trainparams

# trainparams="--trainset PIAA"
# python train_piaa_mir_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
# python train_piaa_ici_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth

# trainparams="--trainset PIAA --disable_onehot"
# python train_piaa_mir_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
# python train_piaa_ici_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth