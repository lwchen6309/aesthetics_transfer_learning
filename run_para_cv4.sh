# python train_nima_attr.py --n_fold 4 --fold_id 1

# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 1
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 2
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 3
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 4

python train_piaa_mir.py --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_floral-snow-57.pth

python train_piaa_mir.py --use_cv --n_fold 4 --fold_id 1 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_honest-capybara-58.pth
python train_piaa_mir.py --use_cv --n_fold 4 --fold_id 2 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_misunderstood-leaf-59.pth
python train_piaa_mir.py --use_cv --n_fold 4 --fold_id 3 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_sunny-night-60.pth
python train_piaa_mir.py --use_cv --n_fold 4 --fold_id 4 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_stilted-meadow-61.pth

