#!/usr/bin/env bash
set -euo pipefail

source /home/lwchen/anaconda3/bin/activate torch2
cd /home/lwchen/active_nngp
exec bash run_PARA_PIAA.sh "$@"
