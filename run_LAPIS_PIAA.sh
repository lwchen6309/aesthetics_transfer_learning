#!/usr/bin/env bash
set -euo pipefail

# LAPIS PIAA eval pipeline (MIR + ICI)
# Note: keep checkpoints aligned with validated local artifacts.
HF_REPO="https://huggingface.co/stupidog04/Unified_IAA/resolve/main/models"
MODELS_DIR="${MODELS_DIR:-models_pth}"

mkdir -p "$MODELS_DIR"

ensure_model() {
  local fname="$1"
  local dest="$MODELS_DIR/$fname"
  if [[ -f "$dest" ]]; then
    echo "[skip] $fname already exists"
  else
    echo "[download] $fname"
    wget -q --show-progress -O "$dest" "$HF_REPO/$fname"
  fi
}

ensure_model "best_model_vit_small_patch16_224_piaamir_woven-wind-1160.pth"
ensure_model "best_model_swin_tiny_patch4_window7_224_piaamir_electric-wind-1161.pth"
ensure_model "best_model_swin_tiny_patch4_window7_224_piaaici_crimson-armadillo-1151.pth"
ensure_model "best_model_vit_small_patch16_224_piaaici_misunderstood-pond-1151.pth"

trainparams="--trainset PIAA --is_eval"

run_script="train_piaa_mir_lapis.py"
python "$run_script" $trainparams --backbone vit_small_patch16_224 --resume "$MODELS_DIR"/best_model_vit_small_patch16_224_piaamir_woven-wind-1160.pth
python "$run_script" $trainparams --backbone swin_tiny_patch4_window7_224 --resume "$MODELS_DIR"/best_model_swin_tiny_patch4_window7_224_piaamir_electric-wind-1161.pth

run_script="train_piaa_ici_lapis.py"
python "$run_script" $trainparams --backbone swin_tiny_patch4_window7_224 --resume "$MODELS_DIR"/best_model_swin_tiny_patch4_window7_224_piaaici_crimson-armadillo-1151.pth
python "$run_script" $trainparams --backbone vit_small_patch16_224 --resume "$MODELS_DIR"/best_model_vit_small_patch16_224_piaaici_misunderstood-pond-1151.pth
