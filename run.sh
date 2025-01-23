#!/bin/bash

## PARA
trainparams="--trainset sGIAA --is_eval"
# trainparams="--trainset sGIAA"

# run_script="train_nima.py"
# # python $run_script $trainparams --backbone resnet50
# python $run_script $trainparams --backbone vit_small_patch16_224
# python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

# run_script="train_histonet_latefusion.py"
# # python $run_script $trainparams --backbone resnet50
# python $run_script $trainparams --backbone vit_small_patch16_224
# python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

# run_script="train_piaa_mir.py"
# # python $run_script $trainparams --backbone resnet50
# python $run_script $trainparams --backbone vit_small_patch16_224
# python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

# run_script="train_piaa_ici.py"
# # python $run_script $trainparams --backbone resnet50
# python $run_script $trainparams --backbone vit_small_patch16_224
# python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

# python train_piaa_mir.py $trainparams
# python train_piaa_ici.py $trainparams

## LAPIS
trainparams="--trainset GIAA --is_eval"
# python train_nima_lapis.py $trainparams
# python train_histonet_latefusion_lapis.py $trainparams
# python train_piaa_mir_lapis.py $trainparams
# python train_piaa_ici_lapis.py $trainparams

run_script="train_nima_lapis.py"
python $run_script $trainparams --backbone vit_small_patch16_224
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

run_script="train_histonet_latefusion_lapis.py"
python $run_script $trainparams --backbone vit_small_patch16_224
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

run_script="train_piaa_mir_lapis.py"
python $run_script $trainparams --backbone vit_small_patch16_224
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

run_script="train_piaa_ici_lapis.py"
python $run_script $trainparams --backbone vit_small_patch16_224
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224


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