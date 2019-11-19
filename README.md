# Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection (DexiNed)

<!-- ```diff
- Sorry for any inconvenience, we are updating the repo
``` -->

This work presents a new Convolutional Neural Network (CNN) arquitecture for edge detection. Unlike of the state-of-the-art CNN based edge detectors, this models has a single training stage, but it is still able to overcome those models in the edge detection datasets. Moreover, Dexined does not need pre-trained weights, and it is trained from the scratch with fewer parameters tunning.

<div style="text-align:center"><img src='figs/DexiNed_banner.png' width=800>

# Introduction

* [Geting Started](#geting-started)
* [Datasets](#datasets)
* [Performance](#performance)

# Geting Started

 Before starting to use this model,  there are some requerements to fullfile.
 
## Requirements

* [TensorFlow>=1.8 <=1.13.1](https://www.tensorflow.org) (tested on such versions)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Matplotlib](https://matplotlib.org/3.1.1/users/installing.html)
* Other package like Numpy, h5py, PIL. 

Once the packages are installed,  clone this repo as follow: 

    git clone https://github.com/xavysp/DexiNed.git
    cd DexiNed

## Project Architecture

```
├── data                          # dataset generator 
|   ├── pre_data_generator.py   # data genertor for pre-train phase
|   └── meta_data_generator.py  # data genertor for meta-train phase
├── figs                      # tensorflow model files 
|   ├── resnet12.py             # resnet12 class
|   └── meta_model.py           # meta-train model class
├── models                     # tensorflow trianer files  
|   └── dexined.py                 # meta-train trainer class
├── utls                       # a series of tools used in this repo
|   └── dataset_manager.py      # miscellaneous tool functions
|   └── losses.py               # miscellaneous tool functions
|   └── utls.py                 # miscellaneous tool functions
├── run_model.py                # the python file with main function and parameter settings
└── test.py               # the script to run the whole experiment
└── train.py              # the script to run the whole experiment
```

* [Checkpoints](https://drive.google.com/open?id=1fLBpOrSXC2VOWUvDtNGyrHcuB2IB-4_D)

## Test
    python run_model.py 


# Datasets

## Trained dataset

BIPED (Barcelona Images for Perceptual Edge Detection): This dataset is collected and annotated in the edge level for this work. See more details and download [here](https://xavysp.github.io/MBIPED/)

## Tested datasets

Edge detection dataset
* [BIPED](https://xavysp.github.io/MBIPED/) and [MDBD](http://serre-lab.clps.brown.edu/resource/multicue/)

Non-edge detection datasets

* [CID](http://www.cs.rug.nl/~imaging/databases/contour_database/contour_database.html) <!-- * [DCD](http://www.cs.cmu.edu/~mengtial/proj/sketch/)-->, [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [NYUD](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and [PASCAL-context](https://cs.stanford.edu/~roozbeh/pascal-context/)

# Performance
<center>

|     Methods    |    ODS   |    ODS   |    AP    |
| -------------- | ---------| -------- | -------- |
| `[SED]()`      | `60.2 ±` | `74.3 ±` | `43.6 ±` |
| `[HED]()`      | `60.8 ±` | `74.3 ±` | `44.3 ±` |
| `[RCF]()`      | `60.8 ±` | `74.3 ±` | `44.3  ` |
| `[BDCN]()`     | `60.8 ±` | `74.3  ` | `44.3 ±` |
| `DexiNed(Ours)`| `60.8  ` | `74.3 ±` | `44.3 ±` |
</center>

# Citation
Please cite our paper if you find helpful,
```
@InProceedings{soria2020dexined,
    title={Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection},
    author={Xavier Soria and Edgar Riba and Angel Sappa},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision (WACV '20)},
    year={2020}
}
```

```
@misc{soria2020dexined_ext,
    title={Towards a Robust Deep Learning Model for Edge Detection},
    author={Xavier Soria and Edgar Riba and Angel Sappa},
    year={2020},
    eprint={000000000},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
```
# Acnowledgement
