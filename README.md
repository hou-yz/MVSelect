# Learning to Select Camera Views: Efficient Multiview Understanding at Few Glances

## Overview
We release code for **MVSelect**, a view selection module for efficient multiview understanding. Parallel to reducing the image resolution or using lighter network backbones, the proposed approach reduces the computational cost for multiview understanding by limiting the number of views considered. 


 
## Content
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Training](#training)


## Dependencies
Please install dependencies with 
```
pip install -r requirements.txt
```

## Data Preparation

For multiview classification, we use ModelNet40 dataset with the circular 12-view setup [[link](https://github.com/jongchyisu/mvcnn_pytorch)][[download](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)] and the dodecahedral 20-view setup [[link](https://github.com/kanezaki/pytorch-rotationnet)][[download](https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar)]. 

For multiview detection, we use MultiviewX [[link](https://github.com/hou-yz/MultiviewX)][[download](https://1drv.ms/u/s!AtzsQybTubHfgP9BJt2g7R_Ku4X3Pg?e=GFGeVn)] and Wildtrack [[link](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)][[download](http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip)] in this project. 

Your `~/Data/` folder should look like this
```
Data/
├── modelnet/
│   ├── modelnet40_images_new_12x/
│   │   └── ...
│   └── modelnet40v2png_ori4/
|       └── ...
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```


## Training
In order to train the task networks, please run the following
```shell script
# task network
python main.py
``` 
This should automatically return the full N-view results, as well as the oracle performances. 

To train the MVSelect, please run
```shell script
# MVSelect only
python main.py --step 2 --base_lr 0 --other_lr 0
# joint training
python main.py --step 2
``` 
The default dataset is Wildtrack. For other datasets, please specify with the `-d` argument.


## Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRCxKzkaOiLCKkIIA?e=fQxfhI).
