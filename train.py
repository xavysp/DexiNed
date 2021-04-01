

import tensorflow as tf
import matplotlib.pyplot as plt

from models.dexined import dexined
# from models.dexinedBs import dexined
from utls.utls import *
from utls.dataset_manager import (data_parser,
                                  get_training_batch,get_validation_batch, visualize_result)

class m_trainer():

    def __init__(self,args ):
        self.init = True
        self.args = args

    def setup(self):
        try:
            if self.args.model_name=='DXN':
                self.model = dexined(self.args)
            else:
                print_error("Error setting model, {}".format(self.args.model_name))

            print_info("DL model Set")
        except Exception as err:
            print_error("Error setting up DL model, {}".format(err))
            self.init=False

    def run(self, sess):
        if not self.init:
            return
        train_data = data_parser(self.args)

        self.model.setup_training(sess)
        if self.args.lr_scheduler is not None:
            global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        if self.args.lr_scheduler is None:
            learning_rate = tf.constant(self.args.learning_rate, dtype=tf.float16)
        else:
            raise NotImplementedError('Learning rate scheduler type [%s] is not implemented',
                                      self.args.lr_scheduler)
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
        trainG = opt.minimize(self.model.loss)# like hed
        saver = tf.compat.v1.train.Saver(max_to_keep=7)

        sess.run(tf.compat.v1.global_variables_initializer())
        # here to recovery previous training
        if self.args.use_previous_trained:
            if self.args.dataset_name.lower()!='biped': # using biped pretrained to use in other dataset
                model_path = os.path.join(self.args.checkpoint_dir,self.args.model_name+
                                          '_'+self.args.train_dataset,'train')
            else:
                model_path = os.path.join(self.args.checkpoint_dir, self.args.model_name + '_' + self.args.train_dataset)
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
            maxi = ini + self.args.max_iterations
        prev_loss=1000.
        prev_val = None
        # directories for checkpoints
        checkpoint_dir = os.path.join(
            self.args.checkpoint_dir, self.args.model_name + '_' + self.args.train_dataset,
            self.args.model_state)
        os.makedirs(checkpoint_dir,exist_ok=True)

        fig = plt.figure()
        for idx in range(ini, maxi):

            x_batch, y_batch,_ = get_training_batch(self.args, train_data)
            run_metadata = tf.compat.v1.RunMetadata()

            _, summary, loss,pred_maps= sess.run(
                [trainG, self.model.merged_summary, self.model.loss, self.model.predictions],
                feed_dict={self.model.images: x_batch, self.model.edgemaps: y_batch})
            if idx%5==0:
                self.model.train_writer.add_run_metadata(run_metadata,
                                                         'step{:06}'.format(idx))
                self.model.train_writer.add_summary(summary, idx)
                print(time.ctime(), '[{}/{}]'.format(idx, maxi), ' TRAINING loss: %.5f' % loss,
                  'prev_loss: %.5f' % prev_loss)

            # saving trained parameters
            save_inter = ini+self.args.save_interval
            if prev_loss>loss:
                saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                prev_loss = loss
                print("Weights saved in the lowest loss",idx, " Current Loss",prev_loss)

            if idx % self.args.save_interval == 0:
                saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                prev_loss = loss
                print("Weights saved in the interval", idx, " Current Loss",prev_loss)

            # ********* for validation **********
            if (idx+1) % self.args.val_interval== 0:
                pause_show=0.01
                imgs_list = []
                img = x_batch[2][:,:,0:3]
                gt_mp= y_batch[2]
                imgs_list.append(img)
                imgs_list.append(gt_mp)
                for i in range(len(pred_maps)):
                    tmp=pred_maps[i][2,...]
                    imgs_list.append(tmp)
                vis_imgs = visualize_result(imgs_list, self.args)
                fig.suptitle("Iterac:" + str(idx + 1) + " Loss:" + '%.5f' % loss + " training")
                fig.add_subplot(1,1,1)
                plt.imshow(np.uint8(vis_imgs))

                print("Evaluation in progress...")
                plt.draw()
                plt.pause(pause_show)

                im, em, _ = get_validation_batch(self.args, train_data)
                summary, error, pred_val = sess.run(
                    [self.model.merged_summary, self.model.error, self.model.fuse_output],
                    feed_dict={self.model.images: im, self.model.edgemaps: em})
                if error<=0.08:
                    saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
                    prev_loss = loss
                    print("Parameters saved in the validation stage when its error is <=0.08::", error)

                self.model.val_writer.add_summary(summary, idx)
                print_info(('[{}/{}]'.format(idx, self.args.max_iterations),'VALIDATION error: %0.5f'%error,
                           'pError: %.5f'%prev_loss))
                if (idx+1) % (self.args.val_interval*150)== 0:
                    print('updating visualisation')
                    plt.close()
                    fig = plt.figure()

        saver.save(sess, os.path.join(checkpoint_dir, self.args.model_name), global_step=idx)
        print("Final Weights saved", idx, " Current Loss", loss)
        self.model.train_writer.close()
        sess.close()

