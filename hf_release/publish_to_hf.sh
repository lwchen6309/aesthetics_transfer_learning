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

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set"
  exit 1
fi

source /home/lwchen/anaconda3/bin/activate torch2

hf auth login --token "$HF_TOKEN" --add-to-git-credential

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

cp -f "${SRC_DIR}/models_pth/best_model_resnet50_piaamir_desert-dawn-621.pth" models/
cp -f "${SRC_DIR}/models_pth/best_model_resnet50_piaaici_crimson-sound-642.pth" models/

git add .
if git diff --cached --quiet; then
  echo "No changes to push."
  exit 0
fi

git commit -m "Release pt70 resnet50 checkpoints + prior vector + inference scripts"
git push

echo "Published to https://huggingface.co/${REPO_ID}"
