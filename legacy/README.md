[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dense-extreme-inception-network-towards-a/edge-detection-on-biped)](https://paperswithcode.com/sota/edge-detection-on-biped?p=dense-extreme-inception-network-towards-a)

# Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection (DexiNed)

<!-- ```diff
- Sorry for any inconvenience, we are updating the repo
``` -->

This work presents a new Convolutional Neural Network (CNN) arquitecture for edge detection. Unlike of the state-of-the-art CNN based edge detectors, this models has a single training stage, but it is still able to overcome those models in the edge detection datasets. Moreover, Dexined does not need pre-trained weights, and it is trained from the scratch with fewer parameters tunning. To know more about DexiNed, read our first version of Dexined in [arxiv](https://arxiv.org/abs/1909.01955), the last version will be uploaded after the camera-ready deadline of WACV2020.

<div style="text-align:center"><img src='figs/DexiNed_banner.png' width=800>
<div/>

## Table of Contents
* [Datasets](#datasets)
* [Performance](#performance)
* [Citation](#citation)

This is the first version of DexiNed presented in WACV2020, so it can run on TensorFlow 1.15.4. The last version of DexiNed has been implemented on Pytorch and you can see in the main path of this repo.
## Requirements

* [Python 3.7](https://www.python.org/downloads/release/python-370/g)
* [TensorFlow>=1.8 <=1.15.4](https://www.tensorflow.org) (tested on such versions)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Matplotlib](https://matplotlib.org/3.1.1/users/installing.html)
* Other package like Numpy, h5py, PIL. 

Once the packages are installed,  clone this repo as follow: 

    git clone https://github.com/xavysp/DexiNed.git
    cd DexiNed

## Project Architecture

```
├── data                        # sample images for testing
|   ├── lena_std.tif            # sample 1
|   └── stonehengeuk.jpg        # sample 2
├── figs                        # Images used in README.md
|   └── DexiNed_banner.png      # DexiNed banner
├── models                      # tensorflow model file  
|   └── dexined.py              # DexiNed class
├── utls                        # a series of tools used in this repo
|   └── dataset_manager.py      # tools for dataset managing
|   └── losses.py               # Loss function used to train DexiNed 
|   └── utls.py                 # miscellaneous tool functions
├── run_model.py                # the main python file with main functions and parameter settings
└── test.py                     # the script to run the test experiment
└── train.py                    # the script to run the train experiment
```

As described above, run_model.py has the parameters settings, whether DexiNed is used for training or testing, before those processes the parameters need to be set. As highlighted, DexiNed is trained just one time with our proposed dataset BIPED, so in "--train_dataset" as the default setting is BIDEP; however, in the testing stage (--test_dataset), any dataset can be used, even CLASSIC, which is an arbitrary image downloaded from the internet. However, to evaluate with single images or CLASSIC "--use_dataset" has to be in FALSE mode. Whenever a dataset is used to test or train on DexiNed the arguments have to have the list of training or testing files (--train_list, --test_list). Pay attention in the parameters' settings, and change whatever you want, like ''--image_width'' or ''--image_height''. To test the Lena image I set 512x51 (see "test" section).
```
parser.add_argument('--train_dataset', default='BIPED', choices=['BIPED','BSDS'])
parser.add_argument('--test_dataset', default='CLASSIC', choices=['BIPED', 'BSDS','MULTICUE','NYUD','PASCAL','CID'])
parser.add_argument('--dataset_dir',default=None,type=str)
parser.add_argument('--dataset_augmented', default=True,type=bool)
parser.add_argument('--train_list',default='train_rgb.lst', type=str)
parser.add_argument('--test_list', default='test_pair.lst',type=str)  
```

## Test
Before test the DexiNed model, it is necesarry to download the checkpoint here [Checkpoint from Drive](https://drive.google.com/open?id=1fLBpOrSXC2VOWUvDtNGyrHcuB2IB-4_D) and save those files into the DexiNed folder like: checkpoints/DXN_BIPED/train/(here the checkpoints from Drive), then run as follow:

    python run_model.py --image_width=512 --image_height=512
Make sure that in run_model.py the test setting be as:
```parser.add_argument('--model_state', default='test', choices=['train','test','None'])```
DexiNed downsample the input image till 16 scales, please make sure that the image width and height be multiple of 16, like 512, 960, and etc. **In the Checkpoint from Drive you will get data_list.zip, train_1.zip, and train_2.zip. The train_2  contains our last checkpoint trained with the updated BIPED; train_1 has checkpoints with the results presented in WACV'20, and data_list has a list of MDBD dataset images used for testing, if you choose another random list of images, you probably get a better or worst result, I think is not fair.**

## Train

    python run_model.py 
Make sure that in run_model.py the train setting be as:
```parser.add_argument('--model_state', default='train', choices=['train','test','None'])```

# Datasets

## Dataset used for Training

BIPED (Barcelona Images for Perceptual Edge Detection): This dataset is collected and annotated in the edge level for this work. See more details and download in: [Option1](https://xavysp.github.io/MBIPED/), [Option2 kaggle](https://www.kaggle.com/xavysp/biped). The BIPED dataset has been updated, adding more annotations and correcting few mistakes, so those links have the renewed version of BIPED, if you want the older version you may ask us by email. The last performance (table below) will be updated soon. 

## Datasets used for Testing

Edge detection datasets
* [BIPED](https://xavysp.github.io/MBIPED/) and [MDBD](http://serre-lab.clps.brown.edu/resource/multicue/)

Non-edge detection datasets

* [CID](http://www.cs.rug.nl/~imaging/databases/contour_database/contour_database.html) <!-- * [DCD](http://www.cs.cmu.edu/~mengtial/proj/sketch/)-->, [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [NYUD](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and [PASCAL-context](https://cs.stanford.edu/~roozbeh/pascal-context/)

# Performance

The results below are from the last version of BIPEP. After WACV20, the BIPED images have been again checked and added annotations. All of those models have been trained again. 

<center>

|     Methods    |    ODS   |    ODS   |    AP    |
| -------------- | ---------| -------- | -------- |
| [SED](https://github.com/ArashAkbarinia/BoundaryDetection)     | `.717` | `.731` | `.756` |
| [HED](https://github.com/s9xie/hed)     | `.823` | `.847` | `.869` |
| [RCF](https://github.com/yun-liu/rcf)     | `.843` | `.859` | `.882` |
| [BDCN](https://github.com/pkuCactus/BDCN)    | `.839` | `.854` | `.887` |
| DexiNed(WACV'20)| `.859` | `.867` | `.905` |
</center>
Evaluation performed to BIPED dataset. We will update the result soon.

# Citation

If you like DexiNed, why not starring the project on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/xavysp/DexiNed.svg?style=social&label=Star&maxAge=3600)](https://GitHub.com/xavysp/DexiNed/stargazers/)

Please cite our paper if you find helpful in your academic/scientific publication,
```
@InProceedings{soria2020dexined,
    title={Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection},
    author={Xavier Soria and Edgar Riba and Angel Sappa},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision (WACV '20)},
    year={2020}
}
```

<!--```
@misc{soria2020dexined_ext,
    title={Towards a Robust Deep Learning Model for Edge Detection},
    author={Xavier Soria and Edgar Riba and Angel Sappa},
    year={2020},
    eprint={000000000},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
```-->

