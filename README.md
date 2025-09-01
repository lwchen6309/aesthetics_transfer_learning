# Transfer learning in IAA

## Installation

Run
```
git clone git@github.com:lwchen6309/aesthetics_transfer_learning.git
cd aesthetics_transfer_learning
conda env create -f enviroment.yaml
conda activate iaa_transfer

## Setup models and compiled data path
mkdir models_pth -p
cd models_pth
mkdir random_cvs -p
cd ..

mkdir LAPIS_dataset_pkl -p
cd LAPIS_dataset_pkl
mkdir user_cv -p
cd ..

mkdir dataset_pkl -p
cd dataset_pkl
mkdir user_cv -p
cd ..
```


### Dataset
## PARA
Please download PARA dataset from [here](https://cv-datasets.institutecv.com/#/data-sets).

create the dataset folder ```PARA``` and unzip ```para_image_user_split.tar.gz``` to it. With the following structure:

```
PARA
|-- annotation
|------ARA-GiaaTest.csv
|------PARA-GiaaTrain.csv
|------PARA-Images.csv
|------PARA-UserInfo.csv  
|-- imgs  
|------ *.jpg
|-- validation_images.txt
|-- userset
|------TrainUserIDs_Fold[1-4].txt
|------TestUserIDs_Fold[1-4].txt
```

## LAPIS
Please download LAPIS dataset from [here](git@github.com:Anne-SofieMaerten/LAPIS.git), 

create the dataset folder ```PARA``` and unzip ```lapis_image_user_split.tar.gz``` to it. With the following structure:

```
LAPIS
|-- annotation
|------LAPIS_individualratings_metaANDdemodata.csv
|-- images
|------ *.jpg
|-- imageset
|------TrainImageSet.txt
|------ValImageSet.txt
|------TestImageSet.txt
|-- userset
|------TrainUserIDs_Fold[1-4].txt
|------TestUserIDs_Fold[1-4].txt
```
and set the dataset path in ```data_config.yaml```. 

### Model Paths
```
models_pth
|-- random_cvs
|------*.pth (models trained with the users split by 4-fold cross validation)
|--- *.pth (models trained with native setup)
```

## Run GIAA/sGIAA/PIAA Models with overlapped user
### Onehot encoded models
Run
```
bash run.sh
```
to train all models [NIMA, NIMA-Trait, PIAA-MIR (onehot-enc.) and PIAA-ICI (onehot-enc.)] with all backbones (resnet50, vit_small_patch16_224, swin_tiny_patch4_window7_224) on both PARA and LAPIS datasets.

### PIAA baselines

#### Train GIAA models 
To access the pretrained_model, run 
```
python train_nima_attr.py --trainset GIAA
python train_nima_attr_lapis.py --trainset GIAA
```
for PARA and LAPIS datasets.
#### Fine-tuning PIAA-MIR and PIAA-ICI from GIAA pretrained_model
```
trainargs='--trainset PIAA'
run_script="train_piaa_mir.py"
run_script="train_piaa_ici.py"
run_script="train_piaa_mir_lapis.py"
run_script="train_piaa_ici_lapis.py"
python $run_script --pretrained_model pth_to_pretrained_giaa
```

## Run PIAA Models with disjoint user
### Split by 4 fold cross validation 
Set arguments
```
--n_fold 4 --fold_id 1 --use_cv
```
Here we take ```--fold_id 1``` as an example, ```fold_id``` can be 1, 2, 3 and 4.


Run 
```
bash run_4foldcv.sh
```
to train both PIAA-MIR and PIAA-ICI on the unseen users split by 4-fold cv.


## Licence
Our model and code are released under MIT licence.


## Citation
Please cite
```
@article{chen2025role,
  title={On the Role of Individual Differences in Current Approaches to Computational Image Aesthetics},
  author={Chen, Li-Wei and Strafforello, Ombretta and Maerten, Anne-Sofie and Tuytelaars, Tinne and Wagemans, Johan},
  journal={arXiv preprint arXiv:2502.20518},
  year={2025}
}
```
if you use this dataset.
