

import tensorflow as tf
from PIL import Image

from models.dexined import dexined
# from models.dexinedBs import dexined
from utls.utls import *
from utls.dataset_manager import data_parser,get_single_image,\
    get_testing_batch

class m_tester():

    def __init__(self, args):
        self.args = args

    def setup(self, session):

        try:
            if self.args.model_name=='DXN':
                self.model = dexined(self.args)
            else:
                print_error("Error setting model, {}".format(self.args.model_name))
            if self.args.trained_model_dir is None:
                meta_model_file = os.path.join(
                    self.args.checkpoint_dir, os.path.join(
                        self.args.model_name + '_' + self.args.train_dataset,
                        os.path.join('train',
                                     '{}-{}'.format(self.args.model_name, self.args.test_snapshot))))
            else:
                meta_model_file = os.path.join(
                    self.args.checkpoint_dir, os.path.join(
                        self.args.model_name + '_' + self.args.train_dataset,
                        os.path.join(self.args.trained_model_dir,
                                     '{}-{}'.format(self.args.model_name, self.args.test_snapshot))))


            saver = tf.train.Saver()
            saver.restore(session, meta_model_file)
            print_info('Done restoring DexiNed model from {}'.format(meta_model_file))

        except Exception as err:

            print_error('Error setting up DexiNed traied model, {}'.format(err))

    def run(self, session):

        self.model.setup_testing(session)
        if self.args.use_dataset:
            test_data= data_parser(self.args)
            n_data = len(test_data[1])
        else:
            test_data=get_single_image(self.args)
            n_data = len(test_data)
        print_info('Writing PNGs at {}'.format(self.args.base_dir_results))

        if self.args.batch_size_test==1 and self.args.use_dataset:
            for i in range(n_data):
                im, em, file_name = get_testing_batch(self.args,
                                    [test_data[0][test_data[1][i]], test_data[1][i]], use_batch=False)
                self.img_info = file_name
                edgemap = session.run(self.model.predictions, feed_dict={self.model.images: [im]})

                self.save_egdemaps(edgemap, single_image=True)
                print_info('Done testing {}, {}'.format(self.img_info[0], self.img_info[1]))

        # for individual images
        elif self.args.batch_size_test==1 and not self.args.use_dataset:
            for i in range(n_data):
                im, file_name = get_single_image(self.args,file_path=test_data[i])
                self.img_info  = file_name
                edgemap = session.run(self.model.predictions, feed_dict={self.model.images: [im]})
                self.save_egdemaps(edgemap, single_image=True)
                print_info('Done testing {}, {}'.format(self.img_info[0], self.img_info[1]))

    def save_egdemaps(self, em_maps, single_image=False):
        """ save_edgemaps descriptios

        :param em_maps:
        :param single_image:
        save predicted edge maps
        """
        result_dir = 'DexiNed_'+self.args.train_dataset+'2'+self.args.test_dataset
        if self.args.base_dir_results is None:
            res_dir = os.path.join('../results', result_dir)
        else:
            res_dir = os.path.join(self.args.base_dir_results,result_dir)
        gt_dir = os.path.join(res_dir,'gt')
        all_dir = os.path.join(res_dir,'pred-h5')
        resf_dir = os.path.join(res_dir,'pred-f')
        resa_dir = os.path.join(res_dir,'pred-a')

        os.makedirs(resf_dir, exist_ok=True)
        os.makedirs(resa_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(all_dir, exist_ok=True)

        if single_image:
            em_maps = [e[0] for e in em_maps]
            em_a = np.mean(np.array(em_maps), axis=0)
            em_maps = em_maps + [em_a ]

            em = em_maps[len(em_maps)-2]
            em[em < self.args.testing_threshold] = 0.0
            em_a[em_a < self.args.testing_threshold] = 0.0
            em = 255.0 * (1.0 - em)
            em_a = 255.0 * (1.0 - em_a)
            em = np.tile(em, [1, 1, 3])
            em_a = np.tile(em_a, [1, 1, 3])

            em = Image.fromarray(np.uint8(em))
            em_a = Image.fromarray(np.uint8(em_a))

            tmp_name = os.path.basename(self.img_info[0])
            tmp_name = tmp_name[:-4]
            tmp_size = self.img_info[-1][:2]
            tmp_size = (tmp_size[1],tmp_size[0])

            em_f = em.resize(tmp_size)
            em_a = em_a.resize(tmp_size)

            em_f.save(os.path.join(resf_dir, tmp_name + '.png'))
            em_a.save(os.path.join(resa_dir, tmp_name + '.png'))
            em_maps =tensor_norm_01(em_maps)
            save_variable_h5(os.path.join(all_dir, tmp_name + '.h5'), np.float16(em_maps))


        else:
            pass