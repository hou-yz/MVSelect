# Learning to Select Camera Views: Efficient Multiview Understanding at Few Glances [[arXiv](link)]

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
By default, all datasets are in `~/Data/`. 

For multiview classification, we use ModelNet40 dataset with the circular [12-view](https://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1png.tar) setup and the dodecahedral [20-view](https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar) setup. 

For multiview detection, we use [MultiviewX](https://github.com/hou-yz/MultiviewX) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) in this project. 

Your `~/Data/` folder should look like this
```
Data
├── modelnet/
│   ├── modelnet40v1png/
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

