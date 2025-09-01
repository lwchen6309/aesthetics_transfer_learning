# LAPIS_IAA

## Installation

Run
```
git clone git@gitlab.kuleuven.be:u0162014/lapis_iaa.git
cd lapis_iaa
conda env create -f enviroment.yaml
conda activate lapis_iaa

## Setup models and compiled data path
mkdir models_pth -p
cd models_pth
mkdir random_cvs -p
cd ..

mkdir LAPIS_dataset_pkl -p
cd LAPIS_dataset_pkl
mkdir user_cv -p
cd ..
```


### Data Path
Please download the annotation file from [here](https://kuleuven-my.sharepoint.com/:u:/g/personal/li-wei_chen_kuleuven_be/EaS3jktc5xhAqAjIRWB5V6ABkfysoe6IP6wEEcErm4IXLg?e=ekOMW9), 
create the dataset folder ```LAPIS``` and 
unzip ```LAPIS_annotation_collection.tar.gz``` to it.
It should be as structured as follows, 
```
LAPIS
|-- annotation
|------LAPIS_individualratings_metaANDdemodata.csv
|------QIP_LAPIS.csv
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
and copy ```data_config_tmp.yaml``` as ```data_config.yaml``` and set the dataset path in the it. For example,
```
LAPIS_datapath: /data/leuven/XXX/vscXXXXX/datasets/LAPIS
```

### Model Paths
```
models_pth
|-- random_cvs
|------*.pth (models trained with the users split by 4-fold cross validation)
|--- *.pth (models trained with native setup)
```

## Run GIAA Models

Run NIMA the GIAA baseline, supporting resnet18, resnet50, mobilenet_v2, swin_v2_t, swin_v2_s as backbone
```
python train_nima_lapis.py --backbone resnet18 --trainset GIAA
```

Run
```
bash run_giaa.sh
```
to train NIMA with all backbones


## Run PIAA Models
Please download the ```pretrained_model``` from [here](https://kuleuven-my.sharepoint.com/:u:/g/personal/li-wei_chen_kuleuven_be/EdrTgzk7Zn9Ak6aX7Vd9PtIB_jjQNQeC46-yOvqmyYDTEA?e=Werd8o) to ```models_pth```,  and run 
```
trainargs='--trainset PIAA --pretrained_model models_pth/lapis_resnet50_nima_stilted-jazz-101.pth'

run_script="train_piaa_ici_qip_lapis.py"
python $run_script $trainargs 

run_script="train_piaa_mir_qip_lapis.py"
python $run_script $trainargs 
```

or run 
```
bash run_piaa.sh
```
to train both PIAA-MIR and PIAA-ICI


## Run PIAA Models with Unseen Users
Here we use 4-fold cross validation (cv) to split the train and test users. It is specified by arguments
```
--n_fold 4 --fold_id 1 --use_cv
```
Here we take ```--fold_id 1``` as an example, ```fold_id``` can be 1, 2, 3 and 4.

### Access pre-trained GIAA models 
To access the pretrained_model, run 
```
python train_nima_attr_lapis.py --n_fold 4 --fold_id 1 --use_cv --trainset GIAA
```
or download it from [Url] to ```models_pth/random_cvs```

### Fine-tuning PIAA-MIR and PIAA-ICI from GIAA pretrained_model
```
trainargs='--trainset PIAA'

run_script="train_piaa_mir_qip_lapis.py"
python $run_script --n_fold 4 --fold_id 1 --use_cv $trainargs --pretrained_model "models_pth/random_cvs/lapis_resnet50_nima_restful-dawn-114.pth"

run_script="train_piaa_ici_qip_lapis.py"
python $run_script --n_fold 4 --fold_id 1 --use_cv $trainargs --pretrained_model "models_pth/random_cvs/lapis_resnet50_nima_restful-dawn-114.pth"
```
where 
```
models_pth/random_cvs/lapis_resnet50_nima_restful-dawn-114.pth
```
is the models trained from
```
python train_nima_attr_lapis.py --n_fold 4 --fold_id 1 --use_cv --trainset GIAA
```
a mentioned above. Replace the file acocrding to your own pre-trained models.

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
@misc{maerten2024lapis,
  author       = {Anne-Sofie Maerten and Li-Wei Chen and Stefanie De Winter and Christophe Bossens and Johan Wagemans},
  title        = {{LAPIS: A novel dataset for personalized image aesthetic assessment}},
  year         = {2024},
  howpublished = {\url{https://github.com/lwchen6309/LAPIS_IAA}},
  note         = {}
}
```
if you use this dataset.
