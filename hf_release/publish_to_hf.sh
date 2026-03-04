#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   HF_TOKEN=xxx bash hf_release/publish_to_hf.sh <hf_repo_id>
# Example:
#   HF_TOKEN=... bash hf_release/publish_to_hf.sh yourname/piaa-giaa-pt70-resnet50

if [[ $# -lt 1 ]]; then
  echo "Usage: HF_TOKEN=xxx bash hf_release/publish_to_hf.sh <hf_repo_id>"
  exit 1
fi

REPO_ID="$1"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${ROOT_DIR}/.hf_tmp_repo"
SRC_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

source /home/lwchen/anaconda3/bin/activate torch2

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "HF_TOKEN not set, using existing hf auth session..."
  hf auth whoami >/dev/null
fi

rm -rf "$WORK_DIR"
git clone "https://huggingface.co/${REPO_ID}" "$WORK_DIR"
cd "$WORK_DIR"

git lfs install
git lfs track "*.pth" "*.pt"

mkdir -p models inference configs
cp -f "${ROOT_DIR}/README.md" README.md
cp -f "${ROOT_DIR}/environment.torch2.yml" environment.torch2.yml
cp -f "${ROOT_DIR}/configs/compatibility.json" configs/compatibility.json
cp -f "${ROOT_DIR}/inference/"*.py inference/
cp -f "${ROOT_DIR}/inference/prior_mean_vector.pt" inference/
cp -f "${ROOT_DIR}/inference/demographics_encoder.json" inference/

cp -f "${SRC_DIR}/models_pth/best_model_vit_small_patch16_224_piaamir_super-yogurt-742.pth" models/
cp -f "${SRC_DIR}/models_pth/best_model_swin_tiny_patch4_window7_224_piaamir_fanciful-blaze-742.pth" models/
cp -f "${SRC_DIR}/models_pth/best_model_swin_tiny_patch4_window7_224_piaaici_ethereal-cherry-741.pth" models/
cp -f "${SRC_DIR}/models_pth/best_model_vit_small_patch16_224_piaaici_laced-bird-742.pth" models/

git add .
if git diff --cached --quiet; then
  echo "No changes to push."
  exit 0
fi

git commit -m "Release pt70 ViT/Swin checkpoints + prior vector + inference scripts"
git push

echo "Published to https://huggingface.co/${REPO_ID}"
