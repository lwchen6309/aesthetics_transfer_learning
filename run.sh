## PARA
trainparams="--trainset GIAA" # Change to sGIAA or PIAA

models=(resnet50 vit_small_patch16_224 swin_tiny_patch4_window7_224)

for run_script in train_nima.py \
                  train_histonet_latefusion.py \
                  train_piaa_mir.py \
                  train_piaa_ici.py; do
  for backbone in "${models[@]}"; do
    python $run_script $trainparams --backbone $backbone
  done
done


## LAPIS
for run_script in train_nima_lapis.py \
                  train_histonet_latefusion_lapis.py \
                  train_piaa_mir_lapis.py \
                  train_piaa_ici_lapis.py; do
  for backbone in "${models[@]}"; do
    python $run_script $trainparams --backbone $backbone
  done
done
