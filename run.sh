#!/bin/bash
<<<<<<< HEAD
trainargs=''
python train_nima_lapis.py --backbone resnet18 $trainargs
python train_nima_lapis.py --backbone resnet50 $trainargs
python train_nima_lapis.py --backbone mobilenet_v2 $trainargs
python train_nima_lapis.py --backbone swin_v2_t $trainargs
# python train_nima_lapis.py --backbone swin_v2_s $trainargs

trainargs='--trainset PIAA'
python train_piaa_mir_lapis.py --backbone resnet18 $trainargs
python train_piaa_mir_lapis.py --backbone resnet50 $trainargs
python train_piaa_mir_lapis.py --backbone mobilenet_v2 $trainargs
python train_piaa_mir_lapis.py --backbone swin_v2_t $trainargs
# python train_piaa_mir_lapis.py --backbone swin_v2_s $trainargs
=======

# bash run_GIAA.sh
# bash run_sGIAA.sh

# trainparams="--trainset GIAA --is_eval --disable_onehot"
# python train_histonet_latefusion_lapis.py $trainparams --resume models_pth/lapis_best_model_vit_small_patch16_224_histo_lr5e-05_decay_20epoch_happy-shape-1168.pth
# python train_piaa_mir_lapis.py $trainparams
# python train_piaa_ici_lapis.py $trainparams

# trainparams="--trainset sGIAA --is_eval --disable_onehot"
# python train_histonet_latefusion_lapis.py $trainparams
# python train_piaa_mir_lapis.py $trainparams
# python train_piaa_ici_lapis.py $trainparams

trainparams="--trainset PIAA --is_eval --no_log --disable_onehot"
# python train_histonet_latefusion_lapis.py $trainparams
# python train_piaa_mir.py $trainparams --resume models_pth/best_model_resnet50_piaamir_lr5e-05_decay_20epoch_dainty-bush-256.pth
python train_piaa_ici.py $trainparams --resume models_pth/best_model_resnet50_piaaici_graceful-darkness-640.pth


# trainparams="--trainset PIAA"
# python train_piaa_mir_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
# python train_piaa_ici_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth

# trainparams="--trainset PIAA --disable_onehot"
# python train_piaa_mir_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
# python train_piaa_ici_lapis.py $trainparams --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
>>>>>>> release
