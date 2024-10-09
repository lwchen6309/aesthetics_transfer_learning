#!/bin/bash

# bash run_piaa.sh
# bash run_lapis_cv4.sh
trainargs=''
python train_nima_lapis.py --backbone resnet18 $trainargs
python train_nima_lapis.py --backbone resnet50 $trainargs
python train_nima_lapis.py --backbone mobilenet_v2 $trainargs
python train_nima_lapis.py --backbone swin_v2_t $trainargs
python train_nima_lapis.py --backbone swin_v2_s $trainargs