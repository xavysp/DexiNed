

# import os, sys
# import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import functools
# import numpy as np

from models.vgg16 import Vgg16
from models.xception import xceptionet
from utls.utls import *
from utls.dataset_manager import (data_parser,save_result,
                                  get_training_batch,get_validation_batch, visualize_result)

class m_trainer():

    def __init__(self,args ):
        self.init = True
        self.args = args

    def setup(self):
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

            print_info("Done initializing VGG-16")
        except Exception as err:
            print_error("Error setting up VGG-16, {}".format(err))
            self.init=False

    def expon_decay(self,learning_rate, global_step, decay_steps,
                                 decay_rate, staircase, name):
        if global_step is None:
            raise ValueError("global_step is required for exponential_decay.")
        def decayed_lr(learning_rate,global_step, decay_steps, decay_rate,
                      staircase, name):
            with tf.name_scope(name, "ExponentialDecay",
                               [learning_rate, global_step, decay_steps, decay_rate]) as name:
                learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
                dtype = learning_rate.dtype
                decay_steps = tf.cast(decay_steps, dtype)
                decay_rate = tf.cast(decay_rate, dtype)
                global_step_recomp = tf.cast(global_step, dtype)
                p = global_step_recomp / decay_steps
                if staircase:
                    p = tf.math.floor(p)
                return tf.math.divide(
                        learning_rate, tf.math.pow(decay_rate, p), name=name)

        return functools.partial(decayed_lr, learning_rate, global_step, decay_steps,
                                 decay_rate, staircase, name)

    def lr_scheduler_desc(self,learning_rate,global_step, decay_steps, decay_rate,
                      staircase=False, name=None):
        decayed_lr=self.expon_decay(learning_rate,global_step,decay_steps,\
                              decay_rate,staircase=staircase,name=name)
        if not tf.executing_eagerly():
            decayed_lr =decayed_lr()
        return  decayed_lr


    def run(self, sess):
        if not self.init:
            return
        if self.args.dataset_name.upper()=="BSDS" and not self.args.use_trained_model:
            print("While you are using {} dataset  it is necessary use a trained data".format(
                self.args.dataset_name))
            # sys.exit()
        train_data = data_parser(self.args)  # train_data = "files_path": train_samples,"n_files": n_train,
                # "train_indices": train_ids,"validation_indices": valid_ids

        self.model.setup_training(sess)

        if self.args.lr_scheduler is not None:
            global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        if self.args.lr_scheduler is None:
            # learning_rate=tf.convert_to_tensor(self.args.learning_rate,dtype=tf.float32)
            learning_rate = tf.constant(self.args.learning_rate, dtype=tf.float16)
        elif self.args.lr_scheduler=='asce':
            learning_rate = tf.train.exponential_decay(learning_rate=self.args.learning_rate, \
                                                   global_step=global_step,
                                                   decay_steps=self.args.learning_decay_interval,
                                                   decay_rate=self.args.learning_rate_decay, staircase=True)
        elif self.args.lr_scheduler=='desc':
            learning_rate = self.lr_scheduler_desc(learning_rate=self.args.learning_rate,
                                              decay_rate=self.args.learning_rate_decay,
                                              global_step=global_step,
                            decay_steps=self.args.learning_decay_interval)
        else:
            raise NotImplementedError('Learning rate scheduler type [%s] is not implemented',
                                      self.args.lr_scheduler)

        if self.args.optimizer=="adamw" or self.args.optimizer=="adamW":
            # opt = tf.train.AdamOptimizer(learning_rate)
            opt = tf.contrib.opt.AdamWOptimizer(weight_decay = self.args.weight_decay, learning_rate=learning_rate)
        elif self.args.optimizer=="momentum" or self.args.optimizer=="MOMENTUM":
            opt =tf.train.MomentumOptimizer(learning_rate=self.args.learning_rate,
                                                  momentum=0.9)
        elif self.args.optimizer=="adam" or self.args.optimizer=="ADAM":
            # just for adam optimazer
            opt = tf.train.AdamOptimizer(learning_rate)

        # coded by me
        trainG = opt.minimize(self.model.loss, global_step=global_step) if self.args.lr_scheduler is not None \
            else opt.minimize(self.model.loss)# like hed
        saver = tf.train.Saver(max_to_keep=7)
        sess.run(tf.global_variables_initializer())
        # here to recovery previous training
        if self.args.use_previous_trained:
            if self.args.dataset_name.lower()!='ssmihd': # using ssmihd pretrained to use in other dataset
                model_path = 'checkpoints/XCP_SSMIHD/train'
            else:
                model_path = os.path.join(self.args.checkpoint_dir, self.args.model_name + '_' + self.args.which_dataset)
                model_path = os.path.join(model_path, 'train')
            if not os.path.exists(model_path) or len(os.listdir(model_path))==0: # :
                ini = 0
                maxi = self.args.max_iterations+1
                print_warning('There is not previous trained data for the current model... and')
                print_warning('*** The training process is starting from scratch ***')
            else:
                # restoring using the last checkpoint
                assert (len(os.listdir(model_path)) != 0),'There is not previous trained data for the current model...'
                last_ckpt = tf.train.latest_checkpoint(model_path)
                saver.restore(sess,last_ckpt)
                ini=self.args.max_iterations
                maxi=ini+self.args.max_iterations+1 # check
                print_info('--> Previous model restored successfully: {}'.format(last_ckpt))
        else:
            print_warning('*** The training process is starting from scratch ***')
            ini = 0

        prev_loss=0.0
        prev_val = None
        # directories for checkpoints
        if self.args.use_nir:
            checkpoint_dir = os.path.join(self.args.checkpoint_dir,
                                          os.path.join(
                                              self.args.model_name + '_' + self.args.dataset_name + '_RGBN',
                                              self.args.model_state))
        else:
            checkpoint_dir = os.path.join(self.args.checkpoint_dir,
                                          os.path.join(self.args.model_name + '_' + self.args.dataset_name,
                                                       self.args.model_state))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        fig = plt.figure()
        for idx in range(ini, maxi):

            x_batch, y_batch,_ = get_training_batch(self.args, train_data)

            run_metadata = tf.RunMetadata()

            # _,_,_,_,_,_, summary,losses, loss, pred_map,pred_map4 = sess.run([trainG,trainF,train1,train2,train3,
            #                             train4,self.model.merged_summary,self.model.losses,
            #                              self.model.loss, self.model.fuse_output, self.model.output_4],
            #                             feed_dict={self.model.images: x_batch,
            #                                        self.model.edgemaps: y_batch})
            _, summary, loss,pred_maps= sess.run([trainG,
                                        self.model.merged_summary,
                                         self.model.loss, self.model.predictions],
                                        feed_dict={self.model.images: x_batch,
                                                   self.model.edgemaps: y_batch,})

            self.model.train_writer.add_run_metadata(run_metadata,
                                                     'step{:06}'.format(idx))
            self.model.train_writer.add_summary(summary, idx)
            print(time.ctime(), '[{}/{}]'.format(idx, maxi), ' TRAINING loss: %.5f' % loss,
                  'prev_loss: %.5f' % prev_loss)

            # saving trained parameters
            save_inter = ini+self.args.save_interval
            if prev_loss>0.001 and idx>save_inter:
                if loss<=(prev_loss-prev_loss*0.05) and loss>=(prev_loss-prev_loss*0.3):
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("parameters saver because of 10% of previous loss ", idx)
                elif idx%(self.args.max_iterations*0.48)==0: # 0.48
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("parameters saved when the iteration get 48% of its purpose", idx, "max_iter= ",maxi)

            else:
                if prev_loss<0.001:
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("parameters saved for the first time ", idx)
                elif(idx % (self.args.save_interval//4) == 0) and \
                            (loss<=(prev_loss-prev_loss*0.05) and loss>=(prev_loss-prev_loss*0.5)):
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("parameters saver because of 10% of previous loss (save_interval//4)", idx)

            # ********* for validation **********
            if (idx+1) % self.args.val_interval== 0:
                pause_show=0.01
                # plt.close()
                # *** recode with restore_rgb fuinction **********
                imgs_list = []
                img = x_batch[2][:,:,0:3]
                gt_mp= y_batch[2]
                imgs_list.append(img)
                imgs_list.append(gt_mp)
                for i in range(len(pred_maps)):
                    tmp=pred_maps[i][2,...]
                    imgs_list.append(tmp)
                vis_imgs = visualize_result(imgs_list, self.args)
                # plt.title("Epoch:" + str(idx + 1) + " Loss:" + '%.5f' % loss + " training")

                fig.suptitle("Iterac:" + str(idx + 1) + " Loss:" + '%.5f' % loss + " training")
                # plt.imshow(np.uint8(img))
                fig.add_subplot(1,1,1)
                plt.imshow(np.uint8(vis_imgs))

                print("Evaluation in progress...")
                plt.draw()
                plt.pause(pause_show)

                im, em, _ = get_validation_batch(self.args, train_data)

                summary, error, pred_val = sess.run([self.model.merged_summary, self.model.error,
                                           self.model.fuse_output],
                                          feed_dict={self.model.images: im, self.model.edgemaps: em})
                if error<=0.08: # all annotation concideration: 0.13, when is greather that 50 <=0.09
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("Parameters saved in the validation stage when its error is <=0.08::", error)
                # save valid result
                # if idx % self.args.save_interval == 0:
                #     imi = restore_rgb([self.args.channel_swap,self.args.mean_pixel_value[:3]],
                #                  im)
                #     save_result(self.args,[imi,em,pred_val])
                self.model.val_writer.add_summary(summary, idx)
                print_info(('[{}/{}]'.format(idx, self.args.max_iterations),'VALIDATION error: %0.5f'%error,
                           'pError: %.5f'%prev_loss))
                if (idx+1) % (self.args.val_interval*170)== 0:
                    print('updating visualisation')
                    plt.close()
                    fig = plt.figure()

        # plt.show()
        self.model.train_writer.close()
