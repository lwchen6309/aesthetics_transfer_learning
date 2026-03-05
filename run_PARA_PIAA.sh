#!/usr/bin/env bash
set -euo pipefail

# PARA PIAA eval pipeline (MIR + ICI)
trainparams="--trainset PIAA --is_eval --no_log"

run_script="train_piaa_mir.py"
python "$run_script" $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth
python "$run_script" $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_piaamir_fanciful-blaze-742.pth

run_script="train_piaa_ici.py"
python "$run_script" $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth
python "$run_script" $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_piaaici_laced-bird-742.pth
