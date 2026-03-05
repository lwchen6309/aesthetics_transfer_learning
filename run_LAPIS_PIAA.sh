#!/usr/bin/env bash
set -euo pipefail

# LAPIS PIAA eval pipeline (MIR + ICI)
# Note: keep checkpoints aligned with validated local artifacts.

trainparams="--trainset PIAA --is_eval"

run_script="train_piaa_mir_lapis.py"
python "$run_script" $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_piaamir_woven-wind-1160.pth
python "$run_script" $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_piaamir_electric-wind-1161.pth

run_script="train_piaa_ici_lapis.py"
python "$run_script" $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_piaaici_crimson-armadillo-1151.pth
python "$run_script" $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_piaaici_misunderstood-pond-1151.pth
