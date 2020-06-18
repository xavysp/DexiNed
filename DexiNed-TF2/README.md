In cosntruction

# DexiNed-TF2


This repo is an official unofficial version of DexiNed in TensorFlow Keras. The first official version of [DexiNed is in Tensorflow1.13](https://github.com/xavysp/DexiNed).

## Requerements

* Python 3.7
* Tensorflow 2.2.
* OpenCV

## Training and Testing  Settings

Either for training or testing you should set dataset_manager.py, there you could find details of datasets like size, source, and so on, you should set before running DexiNed-TF2.
Then you can set the following code in main.py:
```
DATASET_NAME= ['BIPED','BSDS','BSDS300','CID','DCD','MULTICUE',
                'PASCAL','NYUD','CLASSIC'] # 8
TEST_DATA = DATASET_NAME[1]
TRAIN_DATA = DATASET_NAME[0]
...
parser.add_argument("--model_state",default='test', choices=["train", "test", "export"])

```
Model_state should be "train" for training :)
To train DexiNed-TF2 is similar to training in Tensorflow. For more details see [DexiNed](https://github.com/xavysp/DexiNed/blob/master/README.md).

To summarize: firstly you should download and unzip the BIPED dataset hosted in [Kaggle](https://www.kaggle.com/xavysp/biped). Secondly, augment the dataset with [this ripo](https://github.com/xavysp/MBIPED). Once the BIPED is augmented run.

If you want to use just for testing with single images please choice "Classic" dataset, make a dir "data" into DexiNed-TF2, and leave the images for testing into "data" dir and run in "test" mode. You will find the model's weights [Here](https://drive.google.com/file/d/19Gwa6egqzNolvX4eUoXn-SjRKzxB68AA/view?usp=sharing)
 
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
```diff
+ If you find some typos or you think we can improve the code, we will appreciate your contribution

```
