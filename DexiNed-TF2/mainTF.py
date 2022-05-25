#!/usr/bin/python3.7 python
"""

"""
__author__ = "Xavier Soria Poma"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"
__homepage__="www.cvc.uab.cat/people/xsoria"
__credits__=["tensorflow_tutorial"]
__copyright__   = "Copyright 2020, CIMI"

import argparse
import platform
import tensorflow as tf

from run_model import run_DexiNed
from dataset_manager import dataset_info


# Testing settings

in_linux=True if platform.system()=="Linux" else False

DATASET_NAME= ['BIPED','MBIPED','BSDS','BSDS300','CID','DCD','MULTICUE',
                'PASCAL','NYUD','CLASSIC'] # 8
TEST_DATA = DATASET_NAME[-1] # MULTICUE=6
TRAIN_DATA = DATASET_NAME[0]
test_data_inf = dataset_info(TEST_DATA, is_linux=in_linux)
train_data_inf = dataset_info(TRAIN_DATA, is_linux=in_linux)
test_model=False
is_testing ="test" if test_model else "train"
# training settings

parser = argparse.ArgumentParser(description='Edge detection parameters for feeding the model')
parser.add_argument("--train_dir",default=train_data_inf['data_dir'], help="path to folder containing images")
parser.add_argument("--test_dir",default=test_data_inf['data_dir'], help="path to folder containing images")
parser.add_argument("--data4train",default=TRAIN_DATA, type=str)
parser.add_argument("--data4test",default=TEST_DATA, type=str)
parser.add_argument('--train_list', default=train_data_inf['train_list'], type=str)  # SSMIHD: train_rgb_pair.lst, others train_pair.lst
parser.add_argument('--test_list', default=test_data_inf['test_list'], type=str)  # SSMIHD: train_rgb_pair.lst, others train_pair.lst
parser.add_argument("--model_state",default=is_testing , choices=["train", "test", "export"])
parser.add_argument("--output_dir", default='results', help="where to put output files")
parser.add_argument("--checkpoint_dir", default='checkpoints', help="directory with checkpoint to resume training from or use for testing")

parser.add_argument('--model_name', default='DexiNed', choices=['DexiNed'])
parser.add_argument('--continue_training', default=False, type=bool)
parser.add_argument("--max_epochs", type=int,default=24, help="number of training epochs")#24
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--display_freq", type=int, default=10, help="write current training images every display_freq steps")
parser.add_argument("--scale", type=float, default=None, help="scale image before fed DexiNed.0.5, 1.5 ")

parser.add_argument('--adjust_lr', default=[10, 15],type=int,
                    help='Learning rate step size.')  # [5,10]BIRND [10,15]BIPED/BRIND
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--test_bs", type=int, default=1, help="number of images in test batch")
parser.add_argument("--batch_normalization", type=bool, default=True, help=" use batch norm")
parser.add_argument("--image_height", type=int, default=352, help="scale images to this size before cropping to 256x256")
parser.add_argument("--image_width", type=int, default=352, help="scale images to this size before cropping to 256x256")
parser.add_argument("--crop_img", type=bool, default=False,
                    help="4Training: True crop image, False resize image")
parser.add_argument("--test_img_height", type=int, default=test_data_inf["img_height"],
                    help="network input height size")
parser.add_argument("--test_img_width", type=int, default=test_data_inf["img_width"],
                    help="network input height size")

parser.add_argument("--lr", type=float, default=0.0001, help=" learning rate for adam 1e-4")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--rgbn_mean", type=float, default=[103.939,116.779,123.68, 137.86], help="pixels mean")
parser.add_argument("--checkpoint", type=str, default='DexiNed19_model.h5', help="checkpoint Name")

arg = parser.parse_args()
def main(args):
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    model = run_DexiNed(args=args)
    if args.model_state=='train':
        model.train()
    elif args.model_state =='test':
        model.test()
    else:
        raise NotImplementedError('Sorry you just can test or train the model, please set in '
                                  'args.model_state=')

if __name__=='__main__':
    main(args=arg)