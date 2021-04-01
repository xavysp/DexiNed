""" DexiNed main script

This code is based on DexiNed (Dense Extreme Inception Network for Edge Detection),
Please pay attention in the function config_model() to set any parameter before training or
testing the model.
"""
__author__ = "Xavier Soria Poma, CVC-UAB"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"
__homepage__="www.cvc.uab.cat/people/xsoria"
__credits__=['DexiNed']
__copyright__   = "MIT License [see LICENSE for details]"#"Copyright 2019, CIMI"

import sys
import argparse
import tensorflow as tf

import utls.dataset_manager as dm
from train import m_trainer
from test import m_tester
import platform

def config_model():
    in_linux = True if platform.system() == "Linux" else False
    base_dir = "/opt/dataset/" if in_linux else "../../dataset/"
    parser = argparse.ArgumentParser(description='Basic details to run HED')
    # dataset config
    parser.add_argument('--train_dataset', default='BIPED', choices=['BIPED','BSDS'])
    parser.add_argument('--test_dataset', default='CLASSIC', choices=['BIPED', 'BSDS','MULTICUE','NYUD','PASCAL','CID','DCD'])
    parser.add_argument('--dataset_dir',default=base_dir,type=str) # default:'/opt/dataset/'
    parser.add_argument('--dataset_augmented', default=True,type=bool)
    parser.add_argument('--train_list',default='train_rgb.lst', type=str) # BSDS train_pair.lst, SSMIHD train_rgb_pair.lst/train_rgbn_pair.lst
    parser.add_argument('--test_list', default='test_rgb.lst',type=str) # for NYUD&BSDS:test_pair.lst, biped msi_test.lst/test_rgb.lst
    parser.add_argument('--trained_model_dir', default='train',type=str) # 'trainV2_RN'
    # SSMIHD_RGBN msi_valid_list.txt and msi_test_list.txt is for unified test
    parser.add_argument('--use_nir', default=False, type=bool)
    parser.add_argument('--use_dataset', default=False, type=bool) # test: dataset=True single image=FALSE
    # model config
    parser.add_argument('--model_state', default='train', choices=['train','test','None']) # always in None
    parser.add_argument('--model_name', default='DXN',choices=['DXN','XCP','None'])
    parser.add_argument('--use_v1', default=False,type=bool)
    parser.add_argument('--model_purpose', default='edges',choices=['edges','restoration','None'])
    parser.add_argument('--batch_size_train',default=8,type=int)
    parser.add_argument('--batch_size_val',default=8, type=int)
    parser.add_argument('--batch_size_test',default=1,type=int)
    parser.add_argument('--checkpoint_dir', default='checkpoints',type=str)
    parser.add_argument('--logs_dir', default='logs',type=str)
    parser.add_argument('--learning_rate',default=1e-4, type=float) # 1e-4=0.0001
    parser.add_argument('--lr_scheduler',default=None,choices=[None,'asce','desc']) # check here
    parser.add_argument('--learning_rate_decay', default=0.1,type=float)
    parser.add_argument('--weight_decay', default=0.0002, type=float)
    parser.add_argument('--model_weights_path', default='vgg16_.npy')
    parser.add_argument('--train_split', default=0.9, type=float) # default 0.8
    parser.add_argument('--max_iterations', default=180000, type=int) # 100000
    parser.add_argument('--learning_decay_interval',default=25000, type=int) # 25000
    parser.add_argument('--loss_weights', default=1.0, type=float)
    parser.add_argument('--save_interval', default=20000, type=int)  # 50000
    parser.add_argument('--val_interval', default=30, type=int)
    parser.add_argument('--use_subpixel', default=None, type=bool)  # None=upsampling with transp conv
    parser.add_argument('--deep_supervision', default=True, type= bool)
    parser.add_argument('--target_regression',default=True, type=bool) # true
    parser.add_argument('--mean_pixel_values', default=[103.939,116.779,123.68, 137.86], type=float)# [103.939,116.779,123.68]
    # for Nir pixels mean [103.939,116.779,123.68, 137.86]
    parser.add_argument('--channel_swap', default=[2,1,0], type=int)
    parser.add_argument('--gpu-limit',default=1.0, type= float, )
    parser.add_argument('--use_trained_model', default=True, type=bool)  # for vvg16
    parser.add_argument('--use_previous_trained', default=False, type=bool) # for training
    # image configuration
    parser.add_argument('--image_width', default=512, type=int) # 480 NYUD=560 BIPED=1280 default 400 other 448
    parser.add_argument('--image_height', default=512, type=int) # 480 for NYUD 425 BIPED=720 default 400
    parser.add_argument('--n_channels', default=3, type=int) # last ssmihd_xcp trained in 512
    # test config
    parser.add_argument('--test_snapshot', default=149999, type=int) #  BIPED: 149736 BSDS:101179
    #DexiNedv1=149736,DexiNedv2=149999
    parser.add_argument('--testing_threshold', default=0.0, type=float)
    parser.add_argument('--base_dir_results',default='results/edges',type=str) # default: '/opt/results/edges'
    # single image default=None
    args = parser.parse_args()
    return args

def get_session(gpu_fraction):

    num_threads = False
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto())


def main(args):

    if not args.dataset_augmented:
        # Only for BIPED dataset
        # dm.augment_data(args)
        print("Please visit the webpage of BIPED in:")
        print("https://xavysp.github.io/MBIPED/")
        print("and run the code")
        sys.exit()

    if args.model_state =='train' or args.model_state=='test':
        sess = get_session(args.gpu_limit)
        # sess =tf.Session()
    else:
        print("The model state is None, so it will exit...")
        sys.exit()

    if args.model_state=='train':
        trainer = m_trainer(args)
        trainer.setup()
        trainer.run(sess)
        sess.close()

    if args.model_state=='test':

        if args.test_dataset=="BIPED":
            if args.image_width >700:
                pass
            else:
                print(' image size is not set in non augmented data')
                sys.exit()
        tester = m_tester(args)
        tester.setup(sess)
        tester.run(sess)
        sess.close()

    if args.model_state=="None":
        print("Sorry the model state is {}".format(args.model_state))
        sys.exit()

if __name__=='__main__':

    args = config_model()
    main(args=args)
