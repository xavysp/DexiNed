# X model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os.path as path
# import inspect

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from utilities.utls import make_dirs, read_pretrained_data
from utilities.losses import sigmoid_cross_entropy_balanced

slim = tf.contrib.slim


class dexined():

    def __init__(self, args):

        self.args = args
        self.utw = args.use_trained_model
        self.img_height = args.image_height
        self.img_width = args.image_width
        if args.vgg16_param and args.use_trained_weights:

            base_path = path.abspath(path.dirname(__file__))
            base_path = path.join(base_path, 'data')
            if not make_dirs(base_path):
                self.data_dict = read_pretrained_data(base_path, args.vgg16_param)
                print('VGG16 pretrained data read')
            else:
                print('There is not any pretrained data')
        else:
            print("====== The model will be setted to train from the scratch ====")

        if self.args.use_nir:
            self.images = tf.placeholder(tf.float32, [None, self.args.image_height, \
                                                      self.args.image_width, self.args.n_channels + 1])
        else:
            self.images = tf.placeholder(tf.float32, [None, self.args.image_height, \
                                                      self.args.image_width, self.args.n_channels])
        self.edgemaps = tf.placeholder(tf.float32, [None, self.args.image_height,
                                                    self.args.image_width, 1])
        self.define_model()

    def get_var_on_cpu(self, name, shape, initializer=None):
        """Get variables un cpu:0"""
        if initializer is None:
            initializer = tf.constant_initializer(0.0)
        with tf.device('/cpu:0'):
            var = tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32)
        return var

    def get_var_weight_decay(self, name, shape, w_init=None, wd=None):
        """Get variables with weight decay"""
        if w_init is None:
            w_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
        var = self.get_var_on_cpu(name, shape, initializer=w_init)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name=name + '_w_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def conv2d(self, input, n_output, k_size, stride, use_bias=True, use_trained=False,
               weight_decay=None, name='', is_sconv=False, w_initializer=None, use_bn=False):

        n_input = input.get_shape().as_list()[-1]
        stride = [1, stride, stride, 1]
        if not is_sconv:  # for convolution
            if not use_trained:  # generate random  weights
                ks = k_size
                k_size = [k_size, k_size, n_input, n_output]
                w = self.get_var_weight_decay(name=name + '_W', shape=k_size, w_init=w_initializer,
                                              wd=weight_decay)
                p = int((ks - 1) / 2)
                in_padded = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT")
                conv = tf.nn.conv2d(input=in_padded, filter=w, stride=stride, padding='VALID', name=name)
                if use_bias:
                    b = self.get_var_on_cpu(name=name + '_b', shape=[n_output])
                    conv = tf.nn.bias_add(conv, b)
            else:  # get trained weights
                w_shape = self.data_dict[name + '_W'].shape
                w_init = tf.constant_initializer(self.data_dict[name + '_W'], dtype=tf.float32)
                w = tf.get_variable(name=name + '_W', initializer=w_init, shape=w_shape)
                conv = tf.nn.conv2d(input=input, filter=w, strides=[1, stride, stride, 1], padding="VALID",
                                    name=name)
                if use_bias:
                    b_shape = self.data_dict[name + '_b'].shape
                    b_init = tf.constant_initializer(self.data_dict[name + '_b'], dtype=tf.float32)
                    b = tf.get_variable(name=name + '_b', shape=b_shape, initializer=b_init)
                    conv = tf.nn.bias_add(conv, b)
            if use_bn:
                conv = tf.nn.batch_normalization(conv)


        else:  # for separable convolution *Not implemented for get trained weights*
            k_size = [k_size, k_size, n_input, 1]  # 1= channel_multiplier
            dw = self.get_var_weight_decay(name=name + '_dW', shape=k_size, w_init=w_initializer,
                                           wd=weight_decay)  # deptwise filter
            pw = self.get_var_weight_decay(name=name + '_pW', shape=[1, 1, 1 * n_input, n_output],
                                           w_init=w_initializer, wd=weight_decay)  # deptwise filter
            conv = tf.nn.separable_conv2d(input=input, depthwise_filter=dw, pointwise_filter=pw,
                                          strides=stride, padding="SAME", name=name)
            if use_bias:
                b = self.get_var_on_cpu(name=name + '_b', shape=[n_output])
                conv = tf.nn.bias_add(conv, b)
            if use_bn:  # batch normalization
                conv = tf.nn.batch_normalization(conv)
        return conv

    def max_pool(self, input, name):
        # the ksize is [1, 2, 2, 1]
        ks = 3
        p = int((ks - 1) / 2)
        in_padded = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT")
        return tf.nn.max_pool(in_padded, ksize=[1, ks, ks, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def deconv2d(self, input, n_output, name, k_size=None, weight_decay=None, use_bn=False, stride=2):
        # k_size=[fsize,fsize,nInput,nOutput] return a double size of input,
        # the upsampling process is by transpose convolution
        Winit = None  # please set initializer default, is xavier it should be
        # tf.truncated_normal_initializer(stddev=0.1)
        n_input = input.get_shape().as_list()[-1]
        k_size = [k_size, k_size, n_input, n_output]
        w = self.get_var_weight_decay(name=name + '_W', shape=k_size,
                                      w_init=Winit, wd=weight_decay)
        b = self.get_var_on_cpu(name=name + '_b', shape=[n_output])
        dconv = tf.nn.conv2d_transpose(input=input, filter=w, strides=[1, stride, stride, 1],
                                       padding='SAME', name=name)
        dconv = tf.nn.bias_add(dconv, b)
        if use_bn:  # batch normalization
            dconv = tf.nn.batch_normalization(dconv)
        return dconv

    def up_block(self, input, n_outs=1, name='', upscale=None, use_subpixel=False, wd=False):
        """:param use_subpixel: None = use tf resize image, False= deconvolution,
        True= deconvolution and at the end subpixel convolution"""
        i = 1
        scale = 2
        out_conv = input
        gen_outputs = 16
        while scale <= upscale:
            if scale == upscale:
                if use_subpixel is False:
                    # for deconvolution
                    out_conv = self.conv2d(input=out_conv, n_output=n_outs, k_size=1, stride=1,
                                           weight_decay=wd, name=name + '_conv' + str(i))
                    out_conv = tf.nn.relu(out_conv)
                    out_conv = self.deconv2d(input=out_conv, n_output=n_outs, name=name + '_dconv' + str(i),
                                             k_size=upscale, weight_decay=wd)
                elif use_subpixel:
                    # for subpixel
                    out_conv = self.conv2d(input=out_conv, n_output=4, k_size=1, stride=1,
                                           weight_decay=wd, name=name + '_conv' + str(i))
                    _err_log = "SubpixelConv2d: The number of input channels == (scale x scale)" \
                               " x The number of output channels"
                    r = 2
                    if n_outs >= 1:
                        if int(out_conv.get_shape()[-1]) != int(r ** 2 * n_outs):
                            raise Exception(_err_log)
                        out_conv = tf.depth_to_space(out_conv, r)
                    else:
                        raise Exception(' the output channel is not setted')
                elif use_subpixel is None:  # for bilinear upsampling
                    out_conv = self.conv2d(input=out_conv, n_output=1, k_size=1, stride=1,
                                           weight_decay=wd, name=name + '_conv' + str(i))
                    out_conv = tf.nn.relu(out_conv)
                    out_conv = tf.image.resize_bilinear(images=out_conv, size=[self.img_height, self.img_width])
                else:
                    raise NotImplementedError

            else:
                if use_subpixel is False:
                    # for deconvolution
                    out_conv = self.conv2d(input=out_conv, n_output=gen_outputs, k_size=1, stride=1,
                                           weight_decay=wd, name=name + '_conv' + str(i))
                    out_conv = tf.nn.relu(out_conv)
                    out_conv = self.deconv2d(input=out_conv, n_output=gen_outputs, name=name + '_dconv' + str(i),
                                             k_size=upscale, weight_decay=wd)
                elif use_subpixel:
                    # for subpixel
                    out_conv = self.conv2d(input=out_conv, n_output=32, k_size=3, stride=1,
                                           weight_decay=wd, name=name + '_conv' + str(i))  # recommend 2
                    out_conv = tf.nn.relu(out_conv)

                    _err_log = "SubpixelConv2d: The number of input channels == (scale x scale)" \
                               " x The number of output channels"
                    r = 2
                    sp_filter = 8
                    if sp_filter >= 1:
                        if int(out_conv.get_shape()[-1]) != int(r ** 2 * sp_filter):
                            raise Exception(_err_log)
                        out_conv = tf.nn.depth_to_space(out_conv, r)
                    else:
                        raise Exception(' the output channel is not setted')
                elif use_subpixel is None:
                    # for bilinear interpolation
                    out_conv = self.conv2d(input=out_conv, n_output=1, k_size=2, stride=1,
                                           weight_decay=wd, name=name + '_conv' + str(i))
                    out_conv = tf.nn.relu(out_conv)
                    im_h = out_conv.get_shape().as_list()[1]
                    im_w = out_conv.get_shape().as_list()[2]
                    out_conv = tf.image.resize_bilinear(images=out_conv, size=[im_h * 2, im_w * 2])

                else:
                    raise NotImplementedError
            i += 1
            scale = 2 ** i
        return out_conv

    def des_block(self, input, res_input, n_outputs, k_size, is_sconv=False, use_bnorm=False, wd=False, name=''):

        conv_block = tf.nn.relu(input)
        conv_block = self.conv2d(input=conv_block, n_output=n_outputs, k_size=k_size,
                                 stride=1, weight_decay=wd, name=name + '_conv1', is_sconv=is_sconv)
        if use_bnorm:
            conv_block = tf.nn.batch_normalization(conv_block)
        conv_block = tf.nn.relu(conv_block)

        conv_block = self.conv2d(input=conv_block, n_output=n_outputs, k_size=k_size,
                                 stride=1, weight_decay=wd, name=name + '_conv2', is_sconv=is_sconv)
        if use_bnorm:
            conv_block = tf.nn.batch_normalization(conv_block)
        conv_block = tf.add(conv_block, res_input) / 2
        return conv_block

    # here start all :) til here everything ok
    def define_model(self, is_training=True):
        """ defining rxnt"""
        start_time = time.time()
        use_sconv = self.args.use_separable_conv  # use separable convolution
        use_spixel = self.args.use_spixel  # use sub pixel convolution
        wdecay = self.args.weight_decay

        with tf.variable_scope('dexinw') as sc:
            # if the input size is [BSx400x400x3]
            # ------------------------ Block 1 ----------------------
            self.conv1_1 = self.conv2d(input=self.images, n_output=32, k_size=3, stride=2, weight_decay=wdecay,
                                       name='conv1_1', use_bn=True, is_sconv=False)  # [BSx200x200x32]
            self.conv1_1 = tf.nn.relu(self.conv1_1)

            self.conv1_2 = self.conv2d(input=self.conv1_1, n_output=64, k_size=3, stride=1, weight_decay=wdecay,
                                       name='conv1_2', use_bn=True, is_sconv=False)  # [BSx200x200x64]
            self.conv1_2 = tf.nn.relu(self.conv1_2)
            self.output1 = self.up_block(input=self.conv1_2, n_outs=1, name='output1', upscale=int(2 ** 1),
                                         use_subpixel=use_spixel, wd=wdecay)  # [BSx400x400x1]
            self.sk4b2output = self.conv2d(input=self.conv1_2, n_output=128, k_size=1, stride=2, weight_decay=wdecay,
                                           name='sk4out_b2', use_bn=True, is_sconv=False)  # [BSx100x100x128]

            # ------------------------ Block 2 -----------------------------
            self.conv2_1 = self.conv2d(input=self.conv1_2, n_output=128, k_size=3, stride=1, weight_decay=wdecay,
                                       name='conv2_1', use_bn=True, is_sconv=False)  # [BSx200x200x128]
            self.conv2_1 = tf.nn.relu(self.conv2_1)

            self.conv2_2 = self.conv2d(input=self.conv2_1, n_output=128, k_size=3, stride=1, weight_decay=wdecay,
                                       name='conv2_2', use_bn=True, is_sconv=False)  # [BSx200x200x128]
            self.conv2_2 = tf.nn.relu(self.conv2_2)
            self.output2 = self.up_block(input=self.conv2_2, n_outs=1, name='output2', upscale=int(2 ** 1),
                                         use_subpixel=use_spixel, wd=wdecay)  # [BSx400x400x1]
            self.maxpool2 = self.max_pool(input=self.conv2_2, name='maxpool2')  # [BSx100x100x128]
            self.add2 = tf.add(self.maxpool2, self.sk4b2output)

            self.sk4b3output = self.conv2d(input=self.add2, n_output=256, k_size=1, stride=2, weight_decay=wdecay,
                                           name='sk4out_b3', use_bn=True, is_sconv=False)  # [BSx50x50x256]

            # ------------------------ Block 3 -----------------------------
            self.block3 = self.add2
            self.add4b3 = self.conv2d(input=self.maxpool2, n_output=256, k_size=1, stride=1, weight_decay=wdecay,
                                      name='sk4out_b3', use_bn=True, is_sconv=False)  # [BSx100x100x256]
            for i in range(2):
                self.block3 = tf.nn.relu(self.block3)
                self.block3 = self.conv2d(input=self.block3, n_output=256, k_size=3, stride=1, weight_decay=wdecay,
                                          name='block3_conv1_{}'.format(i + 1), use_bn=True,
                                          is_sconv=False)  # [BSx100x100x256]
                self.block3 = tf.nn.relu(self.block3)

                self.block3 = self.conv2d(input=self.block3, n_output=256, k_size=3, stride=1, weight_decay=wdecay,
                                          name='block3_conv2_{}'.format(i + 1), use_bn=True,
                                          is_sconv=False)  # [BSx100x100x256]
                self.block3 = tf.add(self.block3, self.add4b3) / 2

            self.output3 = self.up_block(input=self.block3, n_outs=1, name='output3', upscale=int(2 ** 2),
                                         use_subpixel=use_spixel, wd=wdecay)  # [BSx400x400x1]
            self.maxpool3 = self.max_pool(self.block3, name='maxpool3')  # [BSx50x50x256

            self.add3 = tf.add(self.maxpool3, self.sk4b3output)
            self.sk4b4output = self.conv2d(input=self.add3, n_output=512, k_size=1, stride=2, weight_decay=wdecay,
                                           name='sk4b4_out', use_bn=True, is_sconv=False)  # [BSx25x25x512]

            # ----------------------------------- Block4 ---------------------------------------
            self.block4 = self.add3
            self.add4b4_fb2 = self.conv2d(input=self.maxpool2, n_output=256, k_size=1, stride=2, weight_decay=wdecay,
                                          name='add4b4_conv1', use_bn=True, is_sconv=False)  # [BSx50x50x256]
            self.add4b4 = tf.add(self.add4b4_fb2, self.maxpool3)

            self.add4b4 = self.conv2d(input=self.add4b4, n_output=512, k_size=1, stride=1, weight_decay=wdecay,
                                      name='add4b4_conv2', use_bn=True, is_sconv=False)  # [BSx25x25x512]
            for i in range(3):
                self.block4 = tf.nn.relu(self.block4)
                self.block4 = self.conv2d(input=self.block4, n_output=512, k_size=3, stride=1, weight_decay=wdecay,
                                          name='block4_conv1_{}'.format(i + 1), use_bn=True,
                                          is_sconv=False)  # [BSx50x50x512]
                self.block4 = tf.nn.relu(self.block4)

                self.block4 = self.conv2d(input=self.block3, n_output=512, k_size=3, stride=1, weight_decay=wdecay,
                                          name='block4_conv2_{}'.format(i + 1), use_bn=True,
                                          is_sconv=False)  # [BSx50x50x512]
                self.block4 = tf.add(self.block4, self.add4b4) / 2

            self.output4 = self.up_block(input=self.block4, n_outs=1, name='output4', upscale=int(2 ** 3),
                                         use_subpixel=use_spixel, wd=wdecay)  # [BSx400x400x1]
            self.maxpool4 = self.max_pool(self.block3, name='maxpool4')  # [BSx25x25x512
            self.add4 = tf.add(self.maxpool4, self.sk4b4output)
            self.sk4b5output = self.conv2d(input=self.add4, n_output=512, k_size=1, stride=1, weight_decay=wdecay,
                                           name='sk4b5_out', use_bn=True, is_sconv=False)  # [BSx25x25x512]
            # -------------------------- Block5 --------------------------------
            self.block5 = self.add4
            self.add4b5 = self.conv2d(input=self.add4b4_fb2, n_output=512, k_size=1, stride=2, weight_decay=wdecay,
                                      name='add4b5_conv1', use_bn=True, is_sconv=False)  # [BSx25x25x512]
            self.add4b5 = tf.add(self.add4b5, self.maxpool4)
            self.add4b5 = self.conv2d(input=self.add4b5, n_output=512, k_size=1, stride=1, weight_decay=wdecay,
                                      name='add4b5_conv2', use_bn=True, is_sconv=False)  # [BSx25x25x512]
            for k in range(3):
                self.block5 = tf.nn.relu(self.block5)
                self.block5 = self.conv2d(input=self.block5, n_output=512, k_size=3, stride=1, weight_decay=wdecay,
                                          name='block5_conv1_{}'.format(i + 1), use_bn=True,
                                          is_sconv=False)  # [BSx25x25x512]
                self.block5 = tf.nn.relu(self.block5)

                self.block5 = self.conv2d(input=self.block5, n_output=512, k_size=3, stride=1, weight_decay=wdecay,
                                          name='block5_conv2_{}'.format(i + 1), use_bn=True,
                                          is_sconv=False)  # [BSx25x25x512]
                self.block5 = tf.add(self.block5, self.add4b5) / 2

            self.output5 = self.up_block(input=self.block5, n_outs=1, name='output5', upscale=int(2 ** 4),
                                         use_subpixel=use_spixel, wd=wdecay)  # [BSx400x400x1]
            self.add5 = tf.add(self.block5, self.sk4b5output)
            # ------------------------------------ block 6 -----------------------------------

            self.block6 = self.conv2d(input=self.add5, n_output=256, k_size=1, stride=1, weight_decay=wdecay,
                                      name='block6_conv0', use_bn=True, is_sconv=False)  # [BSx25x25x256]
            self.add4b6 = self.conv2d(input=self.block5, n_output=256, k_size=1, stride=1, weight_decay=wdecay,
                                      name='add4b6_conv', use_bn=True, is_sconv=False)  # [BSx25x25x256]
            for i in range(3):
                self.block6 = tf.nn.relu(self.block6)
                self.block6 = self.conv2d(input=self.block6, n_output=256, k_size=3, stride=1, weight_decay=wdecay,
                                          name='block6_conv1_{}'.format(i + 1), use_bn=True,
                                          is_sconv=False)  # [BSx25x25x256]
                self.block6 = tf.nn.relu(self.block6)

                self.block6 = self.conv2d(input=self.block6, n_output=256, k_size=3, stride=1, weight_decay=wdecay,
                                          name='block6_conv2_{}'.format(i + 1), use_bn=True,
                                          is_sconv=False)  # [BSx25x25x256]
                self.block6 = tf.add(self.block6, self.add4b6) / 2

            self.output6 = self.up_block(input=self.block6, n_outs=1, name='output6', upscale=int(2 ** 4),
                                         use_subpixel=use_spixel, wd=wdecay)  # [BSx400x400x1]
            # ******************** Fusion block *******************************

            self.all_outputs = [self.output1, self.output2, self.output3,
                                self.output4, self.output5, self.output6]
            n_outputs = len(self.all_outputs)

            self.fuse = self.conv2d(input=tf.concat(self.all_outputs, axis=3), n_output=1, k_size=1,
                                    stride=1, weight_decay=None, name='fuse_conv', use_bn=False, is_sconv=False,
                                    w_initializer=tf.constant_initializer(1 / n_outputs))  # [BSx400x400x1]
            self.outputs = self.all_outputs + [self.fuse]

        print("Build model finished: {:.4f}s".format(time.time() - start_time))

    def setup_testing(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs for predictions
        """

        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)

    def setup_training(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs
            Compute total loss := side_layer_loss + fuse_layer_loss
            Compute predicted edge maps from fuse layer as pseudo performance metric to track
        """
        self.predictions = []
        self.loss = 0

        print('Deep supervision application set to {}'.format(self.args.deep_supervision))

        for idx, b in enumerate(self.all_outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='cross_entropy{}'.format(idx))
            # before sigmoid_cross_entropy_balanced

            self.predictions.append(output)
            if self.args.deep_supervision:
                s_cost = (self.args.loss_weights * cost)
                tf.add_to_collection('losses', s_cost)
                self.loss += cost
                # deep_supervision
        self.fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')

        self.predictions.append(self.fuse_output)
        f_cost = (self.args.loss_weights * fuse_cost)
        tf.add_to_collection('losses', f_cost)
        self.loss += f_cost

        # *************evaluation code
        tf.add_to_collection('losses', self.loss)
        self.all_losses = tf.add_n(tf.get_collection('losses'), name='all_losses')
        mean_loss = tf.train.ExponentialMovingAverage(0.9, name='avg')
        all_loss = tf.get_collection('losses')
        self.mean_average_op = mean_loss.apply(all_loss + [self.all_losses])
        # end ***************************

        pred = tf.cast(tf.greater(self.fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('Train', self.all_losses)  # previously self.loss
        tf.summary.scalar('Validation', self.error)

        train_log_dir = path.join(self.args.logs_dir, self.args.model_name.lower() +
                                  '_' + self.args.dataset4training.lower(), self.args.model_state)
        val_log_dir = path.join(self.args.logs_dir, self.args.model_name.lower() +
                                '_' + self.args.dataset4training.lower(), 'val')
        _ = make_dirs(train_log_dir)
        _ = make_dirs(val_log_dir)

        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(train_log_dir, session.graph)
        self.val_writer = tf.summary.FileWriter(val_log_dir)