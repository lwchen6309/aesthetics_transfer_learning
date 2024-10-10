#!/bin/bash
trainargs='--is_eval --no_log --batch_size 10'
python train_nima_lapis.py --backbone navit $trainargs --disable_resize
# --disable_resize
# python train_nima_lapis.py --backbone resnet18 $trainargs
# python train_nima_lapis.py --backbone resnet50 $trainargs
# python train_nima_lapis.py --backbone mobilenet_v2 $trainargs
# python train_nima_lapis.py --backbone swin_v2_t $trainargs
# python train_nima_lapis.py --backbone swin_v2_s $trainargs

# trainargs='--is_eval --no_log --trainset PIAA'
# python train_piaa_mir_lapis.py --backbone resnet18 $trainargs
# python train_piaa_mir_lapis.py --backbone resnet50 $trainargs
# python train_piaa_mir_lapis.py --backbone mobilenet_v2 $trainargs
# python train_piaa_mir_lapis.py --backbone swin_v2_t $trainargs
# python train_piaa_mir_lapis.py --backbone swin_v2_s $trainargs
