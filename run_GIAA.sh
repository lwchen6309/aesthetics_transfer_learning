## PARA
trainparams="--trainset GIAA"

run_script="train_nima.py"
python $run_script $trainparams --backbone resnet50
python $run_script $trainparams --backbone vit_small_patch16_224
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

run_script="train_histonet_latefusion.py"
python $run_script $trainparams --backbone resnet50
python $run_script $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_histo_latefusion_sparkling-fire-747.pth
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_histo_latefusion_giddy-star-748.pth

run_script="train_piaa_mir.py"
python $run_script $trainparams --backbone resnet50
python $run_script $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_piaamir_grateful-snowball-749.pth
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_piaamir_sage-sponge-750.pth

run_script="train_piaa_ici.py"
python $run_script $trainparams --backbone resnet50
python $run_script $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_piaaici_vague-sun-751.pth
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_piaaici_sandy-hill-752.pth


# LAPIS
run_script="train_nima_lapis.py"
python $run_script $trainparams --backbone vit_small_patch16_224
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224

run_script="train_histonet_latefusion_lapis.py"
python $run_script $trainparams --backbone vit_small_patch16_224 --resume models_pth/lapis_best_model_vit_small_patch16_224_histo_lr5e-05_decay_20epoch_fluent-bee-1157.pth
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/lapis_best_model_swin_tiny_patch4_window7_224_histo_lr5e-05_decay_20epoch_young-yogurt-1158.pth

run_script="train_piaa_mir_lapis.py"
python $run_script $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_piaamir_scarlet-universe-1159.pth
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_piaamir_honest-frog-1164.pth

run_script="train_piaa_ici_lapis.py"
python $run_script $trainparams --backbone vit_small_patch16_224 --resume models_pth/best_model_vit_small_patch16_224_piaaici_apricot-rain-1165.pth
python $run_script $trainparams --backbone swin_tiny_patch4_window7_224 --resume models_pth/best_model_swin_tiny_patch4_window7_224_piaaici_still-dragon-1166.pth
