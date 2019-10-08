import os, sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

from models.vgg16 import Vgg16
from models.xception import xceptionet
from utls.utls import *
from utls.dataset_manager import (data_parser,get_single_image,
                                  get_testing_batch, open_images)

class m_tester():

    def __init__(self, args):

        self.args = args
        self.init = True

    def setup(self, session):

        try:

            if self.args.model_state=='train':
                is_training=True
            else:
                is_training=False
            if self.args.model_name=='HED'or self.args.model_name=='RCF':
                self.model = Vgg16(self.args,is_training)
            elif self.args.model_name=='XCP':
                self.model = xceptionet(self.args, is_training)
            else:
                print_error("Error setting model, {}".format(self.args.model_name))

            # meta_model_file = os.path.join(self.cfgs['save_dir'], 'models/hed-model-{}'.format(self.cfgs['test_snapshot']))
            if self.args.use_nir:
                meta_model_file = os.path.join(self.args.checkpoint_dir,
                                           os.path.join('SSMIHD_RGBN',
                                                        os.path.join('train','model-{}'.format(
                                                            self.args.test_snapshot))))
            else:
                meta_model_file = os.path.join(self.args.checkpoint_dir,
                                               os.path.join(self.args.model_name+'_'+self.args.train_dataset,
                                                            os.path.join('train', '{}-{}'.format(
                                                                self.args.model_name,self.args.test_snapshot))))
                # test_snapshot the last checkpoint name

            saver = tf.train.Saver()

            saver.restore(session, meta_model_file)

            print_info('Done restoring VGG-16 model from {}'.format(meta_model_file))

        except Exception as err:

            print_error('Error setting up VGG-16 traied model, {}'.format(err))
            self.init = False

    def run(self, session):

        if not self.init:
            return

        self.model.setup_testing(session)

        if self.args.use_dataset:
            test_data= data_parser(self.args)
            n_data = len(test_data[1])
        else:

            test_data=get_single_image(self.args)
            n_data = len(test_data)


        print_info('Writing PNGs at {}'.format(self.args.base_dir_results))

        if self.args.batch_size_test==1 and (self.args.test_dataset=='SSMIHD' and self.args.use_nir):

            for i in range(n_data):
                im, em, file_name = get_testing_batch(self.args,
                                    [test_data[0][test_data[1][i]], test_data[1][i]], use_batch=False)
                self.out_name = file_name[-11:]
                self.gt_maps, self.gt_names= open_images(test_data[0][test_data[1][i]])

                edgemap = session.run(self.model.predictions, feed_dict={self.model.images: [im]})

                self.save_egdemaps(edgemap,index=i, is_image=True)

                print_info('Done testing {}, {}'.format(file_name, im.shape))

        elif self.args.batch_size_test==1 and self.args.use_dataset:
            for i in range(n_data):
                im, em, file_name = get_testing_batch(self.args,
                                    [test_data[0][test_data[1][i]], test_data[1][i]], use_batch=False)
                self.out_name = file_name  #.replace('/opt/dataset/BSDS/test_edge','/opt/results/bsds_hed/pred')
                self.gt_names=None  # for data other than SSMIHD
                # self.gt_maps, self.gt_names= open_images(test_data[0][test_data[1][i]]) # comment for NYUD

                edgemap = session.run(self.model.predictions, feed_dict={self.model.images: [im]})

                self.save_egdemaps(edgemap,index=i, is_image=True)

                print_info('Done testing {}, {}'.format(file_name, im.shape))

        # for individual images
        elif self.args.batch_size_test==1 and not self.args.use_dataset:
            for i in range(n_data):
                im, file_name = get_single_image(self.args,file_path=test_data[i])
                self.out_name = file_name
                edgemap = session.run(self.model.predictions, feed_dict={self.model.images: [im]})
                self.save_egdemaps(edgemap,index=i, is_image=True)

                print_info('Done testing {}, {}'.format(self.out_name, im.shape))

    def save_egdemaps(self, em_maps, index, is_image=False):

        # Take the edge map from the network from side layers and fuse layer
        # for saving results
        result_dir = self.args.test_dataset.lower()+'_'+self.args.model_name.lower()
        res_dir = os.path.join(self.args.base_dir_results,result_dir)
        res_dir = os.path.join(res_dir,'msi_'+self.args.model_name) if self.args.use_nir else os.path.join(res_dir,'vis_'+self.args.model_name)
        gt_dir = os.path.join(res_dir,'gt')
        all_dir = os.path.join(res_dir,'all_res')
        resf_dir = os.path.join(res_dir,'imgs')
        resa_dir = os.path.join(res_dir,'pred-a')

        if not os.path.exists(resf_dir):
            os.makedirs(resf_dir)
        if not os.path.exists(resa_dir):
            os.makedirs(resa_dir)
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)
        if not os.path.exists(all_dir):
            os.makedirs(all_dir)

        if is_image:
            em_maps = [e[0] for e in em_maps]
            em_a = np.mean(np.array(em_maps), axis=0)
            em_maps = em_maps + [em_a ]
            # save gt image

            em = em_maps[len(em_maps)-2]


            em[em < self.args.testing_threshold] = 0.0
            em_a[em_a < self.args.testing_threshold] = 0.0

            em = 255.0 * (1.0 - em)
            em_a = 255.0 * (1.0 - em_a)

            em = np.tile(em, [1, 1, 3])
            em_a = np.tile(em_a, [1, 1, 3])

            em = Image.fromarray(np.uint8(em))
            em_a = Image.fromarray(np.uint8(em_a))
            if self.args.test_dataset=='SSMIHD' and self.args.test_augmented:

                if index < 10:
                    tmp_name = self.out_name[:-4] + '_000' + str(index + 1)
                elif index >= 10 and index < 100:
                    tmp_name = self.out_name[:-4] + '_00' + str(index + 1)
                elif index >= 100 and index < 1000:
                    tmp_name = self.out_name[:-4] + '_0' + str(index + 1)
                else:
                    tmp_name = self.out_name[:-4] + '_' + str(index + 1)
            else:
                tmp_name = os.path.basename(self.out_name)
                tmp_name = tmp_name[:-4]
            if self.args.test_dataset.lower()=='ssmihd' and self.args.use_nir:
                em.save(os.path.join(resf_dir, tmp_name + '.png'))
                self.gt_maps.save(os.path.join(gt_dir, tmp_name + '.png'))  # just for ssmihd dataset
                em_maps = tensor_norm_01(em_maps)
                save_variable_h5(os.path.join(all_dir, tmp_name + '.h5'), np.float16(em_maps))
            else:
                em.save(os.path.join(resf_dir, tmp_name + '.png'))
                em_a.save(os.path.join(resa_dir, tmp_name + '.png'))
                em_maps =tensor_norm_01(em_maps)
                save_variable_h5(os.path.join(all_dir, tmp_name + '.h5'), np.float16(em_maps))

        else:
            pass