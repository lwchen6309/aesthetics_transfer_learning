# python train_nima.py --use_cv --n_fold 4 --fold_id 1
# python train_nima.py --use_cv --n_fold 4 --fold_id 2
# python train_nima.py --use_cv --n_fold 4 --fold_id 3
# python train_nima.py --use_cv --n_fold 4 --fold_id 4

for i in {1..1}
do
    # python train_histonet_latefusion.py --n_fold 4 --fold_id 1 --use_cv --is_eval --no_log --trainset sGIAA --resume models_pth/random_cvs/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_gallant-jazz-50.pth
    # python train_histonet_latefusion.py --n_fold 4 --fold_id 2 --use_cv --is_eval --no_log --trainset sGIAA --resume models_pth/random_cvs/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_volcanic-thunder-51.pth
    # python train_histonet_latefusion.py --n_fold 4 --fold_id 3 --use_cv --is_eval --no_log --trainset sGIAA --resume models_pth/random_cvs/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_frosty-mountain-52.pth
    # python train_histonet_latefusion.py --n_fold 4 --fold_id 4 --use_cv --is_eval --no_log --trainset sGIAA --resume models_pth/random_cvs/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_eternal-firefly-53.pth
    python train_histonet_latefusion.py --n_fold 4 --fold_id 1 --use_cv --is_eval --no_log --trainset sGIAA --resume models_pth/random_cvs/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_iconic-wind-38.pth
    python train_histonet_latefusion.py --n_fold 4 --fold_id 2 --use_cv --is_eval --no_log --trainset sGIAA --resume models_pth/random_cvs/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_frosty-spaceship-39.pth
    python train_histonet_latefusion.py --n_fold 4 --fold_id 3 --use_cv --is_eval --no_log --trainset sGIAA --resume models_pth/random_cvs/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_vital-plant-40.pth
    python train_histonet_latefusion.py --n_fold 4 --fold_id 4 --use_cv --is_eval --no_log --trainset sGIAA --resume models_pth/random_cvs/best_model_resnet50_histo_latefusion_lr5e-05_decay_20epoch_polished-grass-41.pth
done

