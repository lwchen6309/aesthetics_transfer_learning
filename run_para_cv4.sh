# python train_nima.py --n_fold 2 --fold_id 1
# python train_nima.py --n_fold 2 --fold_id 2

# python train_nima.py --n_fold 4 --fold_id 1
# python train_nima.py --n_fold 4 --fold_id 2
# python train_nima.py --n_fold 4 --fold_id 3
# python train_nima.py --n_fold 4 --fold_id 4

for i in {1..1}
do
    # python train_nima.py --n_fold 4 --fold_id 1 --use_cv --no_log --is_eval --resume "models_pth/random_cvs/best_model_resnet50_nima_lr5e-05_decay_20epoch_lemon-flan-557.pth"
    # python train_nima.py --n_fold 4 --fold_id 2 --use_cv --no_log --is_eval --resume "models_pth/random_cvs/best_model_resnet50_nima_lr5e-05_decay_20epoch_rhubarb-pastry-558.pth"
    # python train_nima.py --n_fold 4 --fold_id 3 --use_cv --no_log --is_eval --resume "models_pth/random_cvs/best_model_resnet50_nima_lr5e-05_decay_20epoch_custard-cobbler-559.pth"
    # python train_nima.py --n_fold 4 --fold_id 4 --use_cv --no_log --is_eval --resume "models_pth/random_cvs/best_model_resnet50_nima_lr5e-05_decay_20epoch_butterscotch-strudel-560.pth"
    python train_histonet_latefusion.py --n_fold 4 --fold_id 1 --use_cv --importance_sampling --trainset sGIAA 
    python train_histonet_latefusion.py --n_fold 4 --fold_id 2 --use_cv --importance_sampling --trainset sGIAA 
    python train_histonet_latefusion.py --n_fold 4 --fold_id 3 --use_cv --importance_sampling --trainset sGIAA 
    python train_histonet_latefusion.py --n_fold 4 --fold_id 4 --use_cv --importance_sampling --trainset sGIAA 
done