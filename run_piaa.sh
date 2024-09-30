## PA-IAA
# python train_paiaa_giaa.py --num_epochs 20 --big5_amp 100

# python train_paiaa.py --num_epochs 2 --no_log --pretrained_model models_pth/best_model_resnet50_paiaa_giaa_lr5e-05_decay_2epoch_smooth-terrain-227.pth

## PIAA-MIR
# python train_nima_attr.py --resume models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth --is_eval --no_log
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 1
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 2
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 3
# python train_nima_attr.py --use_cv --n_fold 4 --fold_id 4

giaa_pretrain=models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth
# python train_piaa_usersample.py --no_log --max_annotations_per_user 100 --batch_size 100 --model PIAA_MIR --disable_onehot --num_users 500 --pretrained_model $giaa_pretrain
# resume=models_pth/best_model_resnet50_piaamir_comic-morning-623.pth

# python train_piaa_usersample.py --no_log --max_annotations_per_user 100 --batch_size 50 --model PIAA_ICI --disable_onehot --num_users 500 --pretrained_model $giaa_pretrain
# python train_piaa_usersample.py --no_log --max_annotations_per_user 10 --batch_size 50 --model PIAA_ICI --disable_onehot --num_users 500 --pretrained_model $giaa_pretrain


# trainparams="--trainset PIAA --lr 5e-5 --num_epochs 20 --disable_onehot --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth"
# trainparams="--trainset PIAA --lr 5e-5 --disable_onehot"
# python train_piaa_mir.py $trainparams --model MIR
# python train_piaa_mir.py --trainset PIAA --use_cv --n_fold 4 --fold_id 1 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_solar-mountain-252.pth
# python train_piaa_mir.py --trainset PIAA --use_cv --n_fold 4 --fold_id 2 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_snowy-oath-253.pth
# python train_piaa_mir.py --trainset PIAA --use_cv --n_fold 4 --fold_id 3 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_driven-vortex-254.pth
# python train_piaa_mir.py --trainset PIAA --use_cv --n_fold 4 --fold_id 4 --pretrained_model models_pth/random_cvs/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_faithful-surf-255.pth
# python train_piaa_ici.py $trainparams

# trainparams="--trainset PIAA --pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth"
# trainparams="--trainset PIAA --num_epochs 20"
# python train_piaa_mir_lapis.py $trainparams --model PIAA_MIR
# python train_piaa_mir.py $trainparams --model PIAA_MIR_CF
# python train_piaa_mir_lapis.py $trainparams --model PIAA_MIR_Rank


# pretrained="--pretrained_model models_pth/best_model_resnet50_nima_attr_lr5e-05_decay_20epoch_swept-energy-251.pth"
# python train_piaa_mir_lapis.py $trainparams $pretrained