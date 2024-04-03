# for i in {1..5}
# do
#     python train_nima_lapis.py --n_fold 4 --fold_id 1
#     python train_nima_lapis.py --n_fold 4 --fold_id 2
#     python train_nima_lapis.py --n_fold 4 --fold_id 3
#     python train_nima_lapis.py --n_fold 4 --fold_id 4
# done
# python train_histonet_latefusion_lapis.py


for i in {1..1}
do
    python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 1
    python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 2
    python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 3
    python train_histonet_latefusion_lapis.py --n_fold 4 --fold_id 4
done