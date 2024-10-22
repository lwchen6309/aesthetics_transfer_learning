#!/bin/bash
trainparams="--trainset GIAA --no_log --is_eval"
# python train_histonet_latefusion.py $trainparams
# python train_piaa_mir.py $trainparams
# python train_piaa_ici.py $trainparams

python train_histonet_latefusion_lapis.py $trainparams
# python train_piaa_mir_lapis.py $trainparams
# python train_piaa_ici_lapis.py $trainparams
