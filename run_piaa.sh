## PA-IAA
# python train_paiaa_giaa.py --num_epochs 20
# python train_paiaa.py --num_epochs 2 --no_log --pretrained_model models_pth/best_model_resnet50_paiaa_giaa_lr5e-05_decay_2epoch_smooth-terrain-227.pth


## PIAA-MIR
# python train_nima_attr.py
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 1
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 2
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 3
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 4

python train_piaa_mir.py --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
python train_piaa_mir.py --use_cv --n_fold 4 --fold_id 1 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_solar-mountain-252.pth
python train_piaa_mir.py --use_cv --n_fold 4 --fold_id 2 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_snowy-oath-253.pth
python train_piaa_mir.py --use_cv --n_fold 4 --fold_id 3 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_driven-vortex-254.pth
python train_piaa_mir.py --use_cv --n_fold 4 --fold_id 4 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_faithful-surf-255.pth

