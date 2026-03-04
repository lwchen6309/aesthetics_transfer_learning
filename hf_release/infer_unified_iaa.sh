#!/usr/bin/env bash
set -euo pipefail

# One-shot inference wrapper for Unified_IAA
# Modes:
#   piaa       -> personalized inference with demographics JSON
#   giaa-prior -> prior-based inference with prior_mean_vector.pt
#
# Examples:
#   bash hf_release/infer_unified_iaa.sh piaa \
#     --task mir --backbone vit_small_patch16_224 \
#     --checkpoint models/best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth \
#     --image /path/to/test.jpg \
#     --demographics_json /path/to/user_demo.json \
#     --para_root /mnt/d/datasets/PARA
#
#   bash hf_release/infer_unified_iaa.sh giaa-prior \
#     --task ici --backbone swin_tiny_patch4_window7_224 \
#     --checkpoint models/best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth \
#     --image /path/to/test.jpg \
#     --prior_vector_path inference/prior_mean_vector.pt \
#     --para_root /mnt/d/datasets/PARA

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <piaa|giaa-prior> [args...]"
  exit 1
fi

MODE="$1"
shift

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

source /home/lwchen/anaconda3/bin/activate torch2

ENCODER_JSON="inference/demographics_encoder.json"
PARA_ROOT="/mnt/d/datasets/PARA"
PRIOR_VEC="inference/prior_mean_vector.pt"

# parse optional shared args so we can bootstrap artifacts
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --para_root)
      PARA_ROOT="${ARGS[$((i+1))]}"
      ;;
    --prior_vector_path)
      PRIOR_VEC="${ARGS[$((i+1))]}"
      ;;
  esac
done

# ensure demographics encoder exists
if [[ ! -f "$ENCODER_JSON" ]]; then
  python inference/demographics_encoder.py \
    --userinfo_csv "$PARA_ROOT/annotation/PARA-UserInfo.csv" \
    --out_json "$ENCODER_JSON"
fi

if [[ "$MODE" == "piaa" ]]; then
  exec python inference/predict_piaa.py \
    --encoder_json "$ENCODER_JSON" \
    "$@"
elif [[ "$MODE" == "giaa-prior" ]]; then
  # if prior vector missing, generate and save automatically
  if [[ ! -f "$PRIOR_VEC" ]]; then
    # Need minimal required args to trigger save path + quick run
    # We'll let user-provided args drive task/backbone/checkpoint/image
    python inference/prior_giaa.py \
      --save_prior_vector \
      --prior_vector_path "$PRIOR_VEC" \
      --para_root "$PARA_ROOT" \
      "$@"
  else
    exec python inference/prior_giaa.py \
      --prior_vector_path "$PRIOR_VEC" \
      --para_root "$PARA_ROOT" \
      "$@"
  fi
else
  echo "Unknown mode: $MODE (expected piaa or giaa-prior)"
  exit 1
fi
