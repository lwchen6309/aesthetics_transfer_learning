# python train_nima.py --n_fold 2 --fold_id 1
# python train_nima.py --n_fold 2 --fold_id 2

# python train_nima.py --n_fold 4 --fold_id 1
# python train_nima.py --n_fold 4 --fold_id 2
# python train_nima.py --n_fold 4 --fold_id 3
# python train_nima.py --n_fold 4 --fold_id 4

for i in {1..1}
do
    python train_histonet_latefusion.py --n_fold 4 --fold_id 1
    python train_histonet_latefusion.py --n_fold 4 --fold_id 2
    python train_histonet_latefusion.py --n_fold 4 --fold_id 3
    python train_histonet_latefusion.py --n_fold 4 --fold_id 4
done