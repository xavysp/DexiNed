[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dense-extreme-inception-network-for-edge/edge-detection-on-biped-1)](https://paperswithcode.com/sota/edge-detection-on-biped-1?p=dense-extreme-inception-network-for-edge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dense-extreme-inception-network-for-edge/edge-detection-on-mdbd)](https://paperswithcode.com/sota/edge-detection-on-mdbd?p=dense-extreme-inception-network-for-edge)
# DexiNed: Dense Extreme Inception Network for Edge Detection (Extended version)

```diff
- We have just updated the last version of DexiNed in Pytorch.
- The implementation with TF1 is fransfered to the "legacy" dir.
- These days we are going to updated minor changes but the model, 
  loss function, and the testing and training procedures does not need changes,
  hopefully :), if you have time please check errors and let me know
  in Issues.

```

This is the extended work from DexiNed presented in WACV2020. 
We have improved the architecture and the dataset. Unlike of the state-of-the-art CNN
based edge detectors, this models has a single training stage, 
but it is still able to overcome those models in edge detection datasets. Moreover, 
Dexined does not need pre-trained weights, and it is trained from the scratch 
with fewer parameters tunning. To know more about DexiNed, read our first 
version of Dexined in [ieeexplore](https://ieeexplore.ieee.org/document/9093290), 
and its extended version in [arXiv](https://arxiv.org/abs/2112.02250).
The last version of DexiNed is implemented in Pytorch, in TF2 it 
will be available soon. If to want our preliminary version, please check legacy DIR.

We have prepared a basic intro to DexiNed in a video, [check it in YouTube](https://youtu.be/Hz0uU04B3U8)

<div style="text-align:center"><img src='figs/DexiNed_banner.png' width=800>
</div>

## Table of Contents
* [DexiNed-v1](#dexiNed-v1)
* [TensorFlow](#tensorflow2)
* [Datasets](#datasets)
* [Performance](#performance)
* [Citation](#citation)

# DexiNed-v1

This version was presented in WACV2020, this is the base DexiNed model [implemented in TensorFlow 1](https://github.com/xavysp/DexiNed/tree/master/legacy)


# TensorFlow2

Dexined version on TF 2 is not ready 
## Requirements

* [Python 3.7](https://www.python.org/downloads/release/python-370/g)
* [Pytorch >=1.4](https://pytorch.org/) (Last test 1.9)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Matplotlib](https://matplotlib.org/3.1.1/users/installing.html)
* [Kornia](https://kornia.github.io/)
* Other package like Numpy, h5py, PIL, json. 

Once the packages are installed,  clone this repo as follow: 

    git clone https://github.com/xavysp/DexiNed.git
    cd DexiNed

## Project Architecture

```
├── data                        # Sample images for testing (paste your image here)
|   ├── lena_std.tif            # Sample 1
├── DexiNed-TF2                 # DexiNed in TensorFlow2 (in construction)   
├── figs                        # Images used in README.md
|   └── DexiNed_banner.png      # DexiNed banner
├── legacy                      # DexiNed in TensorFlow1 (presented in WACV2020)
├── utls                        # A series of tools used in this repo
|   └── image.py                # Miscellaneous tool functions
├── datasets.py                 # Tools for dataset managing 
├── dexi_utils.py               # New functions still not used in the currecnt version
├── losses.py                   # Loss function used to train DexiNed (BDCNloss2)
├── main.py                     # The main python file with main functions and parameter settings
                                # here you can test and train
├── model.py                    # DexiNed class in pythorch
```

Before to start please check dataset.py, from the first line of code you can see the datasets used for training/testing. The main.py, line 194, call the data  for the training or testing, see the example of the code below:
```
    parser = argparse.ArgumentParser(description='DexiNed trainer.')
    parser.add_argument('--choose_test_data',
                        type=int,
                        default=1,
                        help='Already set the dataset for testing choice: 0 - 8')
    # ----------- test -------0--

    TEST_DATA = DATASET_NAMES[parser.parse_args().choose_test_data] # max 8
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    test_dir = test_inf['data_dir']
    is_testing = True# current test -352-SM-NewGT-2AugmenPublish

    # Training settings
    TRAIN_DATA = DATASET_NAMES[0] # BIPED=0
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    train_dir = train_inf['data_dir']
```

## Test
As previously mentioned, the datasets.py has, among other things, the whole datasets configurations used in DexiNed for testing and training:
```
DATASET_NAMES = [
    'BIPED',
    'BSDS',
    'BSDS300',
    'CID',
    'DCD',
    'MULTICUE', #5
    'PASCAL',
    'NYUD',
    'CLASSIC'
] 
```
For example, if want to test your own dataset or image choose "CLASSIC" and save your test data in "data" dir.
Before test the DexiNed model, it is necesarry to download the checkpoint here [Checkpoint Pytorch](https://drive.google.com/file/d/1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu/view?usp=sharing) and save this file into the DexiNed folder like: checkpoints/BIPED/10/(here the checkpoints from Drive), then run as follow:

```python main.py --choose_test_data=-1 ```
Make sure that in main.py the test setting be as:
```parser.add_argument('--is_testing', default=True, help='Script in testing mode.')```
DexiNed downsample the input image till 16 scales, please make sure that, in dataset_info fucn (datasets.py), the image width and height be multiple of 16, like 512, 960, and etc. **In the Checkpoint from Drive you will find the last trained checkpoint, which has been trained in the last version of BIPED dataset that will be updated soon in Kaggle **

## Train

    python main.py 
Make sure that in main.py the train setting be as:
```parser.add_argument('--is_testing', default=False, help='Script in testing mode.')```

# Datasets

## Dataset used for Training
```diff
- The BIPED dataset has been updated to the last version and it can be download from Kaggle

```

BIPED (Barcelona Images for Perceptual Edge Detection): This dataset is collected and annotated in the edge level for this work. **See more details, augmentation script, and download** in: [Option1](https://xavysp.github.io/MBIPED/), [Option2 kaggle](https://www.kaggle.com/xavysp/biped). The BIPED dataset has been updated, adding more annotations and correcting few mistakes, so those links have the renewed version of BIPED, if you want the older version you may ask us by email. The last performance (table below) will be updated soon. 

## Datasets used for Testing

Edge detection datasets
* [BIPED](https://xavysp.github.io/MBIPED/) and [MDBD](http://serre-lab.clps.brown.edu/resource/multicue/)

Non-edge detection datasets

* [CID](http://www.cs.rug.nl/~imaging/databases/contour_database/contour_database.html) <!-- * [DCD](http://www.cs.cmu.edu/~mengtial/proj/sketch/)-->, [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [NYUD](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and [PASCAL-context](https://cs.stanford.edu/~roozbeh/pascal-context/)

# Performance

The results below are from the last version of BIPEP (BIPEDv2, we leave as BIPED because this is the default version).
After WACV20, the BIPED images have been checked again and added more annotations. All of those models have been trained again. 

<center>

|     Methods    |    ODS   |    ODS   |    AP    |
| -------------- | ---------| -------- | -------- |
| [RCF](https://github.com/yun-liu/rcf)      | `.849` | `.861` | `.906` |
| [BDCN](https://github.com/pkuCactus/BDCN)     | `.890` | `.899` | `.934` |
| [CATS](https://github.com/WHUHLX/CATS)     | `.887` | `.892` | `.817` |
| DexiNed-f(Ours)| `.895` | `.900` | `.927` |
| DexiNed-a(Ours)| `.893` | `.897` | `.940` |

</center>


# Citation

If you like DexiNed, why not starring the project on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/xavysp/DexiNed.svg?style=social&label=Star&maxAge=3600)](https://GitHub.com/xavysp/DexiNed/stargazers/)

Please cite our paper if you find helpful in your academic/scientific publication,
```
@INPROCEEDINGS {xsoria2020dexined,
author = {X. Soria and E. Riba and A. Sappa},
booktitle = {2020 IEEE Winter Conference on Applications of Computer Vision (WACV)},
title = {Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection},
year = {2020},
volume = {},
issn = {},
pages = {1912-1921},
keywords = {image edge detection;convolution;training;feeds;machine learning;task analysis;kernel},
doi = {10.1109/WACV45572.2020.9093290},
url = {https://doi.ieeecomputersociety.org/10.1109/WACV45572.2020.9093290},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {mar}
}
```
Extended version.
```
@article{soria2023dexined_ext,
title = {Dense extreme inception network for edge detection},
journal = {Pattern Recognition},
volume = {139},
pages = {109461},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109461},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323001619},
author = {Xavier Soria and Angel Sappa and Patricio Humanante and Arash Akbarinia},
keywords = {Edge detection, Deep learning, CNN, Contour detection, Boundary detection, Segmentation}
}
```

