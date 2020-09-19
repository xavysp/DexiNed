import tensorflow as tf
import numpy as np
import h5py, os
import random
import cv2 as cv


AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1024

img_shape =None

class DataLoader(tf.keras.utils.Sequence):

    def __init__(self,data_name,arg=None, is_val=False):

        self.is_training = True if arg.model_state.lower() == 'train' else False
        self.dim_w = arg.image_width if self.is_training else arg.test_img_width
        self.dim_h = arg.image_height if self.is_training else arg.test_img_height
        self.args = arg
        self.base_dir = arg.train_dir if arg.model_state.lower()=='train' else arg.test_dir
        self.is_val = is_val
        self.data_name =data_name
        self.bs = arg.batch_size if self.is_training else arg.test_bs
        self.shuffle=self.is_training
        if not self.is_training and arg.model_state=="test":
            i_width =  self.dim_w if  self.dim_w%16==0 else (self.dim_w//16+1)*16
            i_height= self.dim_h if self.dim_h%16==0 else (self.dim_h//16+1)*16
            self.input_shape = (None,i_height, i_width,3)
            self.dim_w = i_width
            self.dim_h = i_height
            self.imgs_shape = []
            # OMSIV real size= 320,580,3
        self.data_list = self._build_index()
        self.on_epoch_end()


    def _build_index(self):

        # base_dir = os.path.join(self.base_dir, self.args.model_state.lower())
        list_name= self.args.train_list if self.is_training else self.args.test_list

        if not self.data_name.lower()=='classic':
            file_path = os.path.join(self.base_dir, list_name)
            with open(file_path,'r') as f:
                file_list = f.readlines()
            file_list = [line.strip() for line in file_list] # to clean the '\n'
            file_list = [line.split(' ') for line in file_list] # separate paths
        if self.data_name.lower() in ['biped','mbiped']:
            m_mode = 'train' if self.is_training else 'test'
            input_path = [os.path.join(
                self.base_dir,'imgs',m_mode,line[0]) for line in file_list]
            gt_path = [os.path.join(
                self.base_dir,'edge_maps',m_mode,line[1]) for line in file_list]
        elif self.data_name.lower()=='classic':
            file_list = os.listdir(self.base_dir)
            input_path = [os.path.join(self.base_dir,line) for line in file_list]
            gt_path = None
        else:
            input_path = [os.path.join(self.base_dir, line[0]) for line in file_list]
            gt_path = [os.path.join(self.base_dir, line[1]) for line in file_list]

        # split training and validation, val=10%
        if self.is_training and self.is_val:
            input_path = input_path[int(0.9 * len(input_path)):]
            gt_path = gt_path[int(0.9 * len(gt_path)):]
        elif self.is_training:
            input_path = input_path[:int(0.9 * len(input_path))]
            gt_path = gt_path[:int(0.9 * len(gt_path))]

        if not self.is_training:
            self.imgs_name = [os.path.basename(k) for k in input_path]
            for tmp_path in input_path:
                tmp_i = cv.imread(tmp_path)
                tmp_shape = tmp_i.shape[:2]
                self.imgs_shape.append(tmp_shape)
        sample_indeces= [input_path, gt_path]
        return sample_indeces

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)//self.bs


    def __getitem__(self, index):

        indices = self.indices[index*self.bs:(index+1)*self.bs]
        if not self.data_name.lower()=='classic':

            x_list,y_list = self.data_list
            tmp_x_path = [x_list[k] for k in indices]
            tmp_y_path = [y_list[k] for k in indices]

            x,y = self.__data_generation(tmp_x_path,tmp_y_path)
        else:
            x_list, _ = self.data_list
            tmp_x_path = [x_list[k] for k in indices]
            x, y = self.__data_generation(tmp_x_path, None)
        return x,y

    def __data_generation(self,x_path,y_path):
        if self.args.scale is not None and self.args.model_state.lower()!='train':
            scl= self.args.scale
            scl_h = int(self.dim_h*scl) if (self.dim_h*scl)%16==0 else \
                int(((self.dim_h*scl) // 16 + 1) * 16)
            scl_w = int(self.dim_w * scl) if (self.dim_w * scl) % 16 == 0 else \
                int(((self.dim_h * scl) // 16 + 1) * 16)

            x = np.empty((self.bs, scl_h, scl_w, 3), dtype="float32")
        else:
            x = np.empty((self.bs, self.dim_h, self.dim_w, 3), dtype="float32")
        y = np.empty((self.bs, self.dim_h, self.dim_w, 1), dtype="float32")

        for i,tmp_data in enumerate(x_path):
            tmp_x_path = tmp_data
            tmp_y_path = y_path[i] if not self.data_name.lower()=='classic' else None
            tmp_x,tmp_y = self.transformer(tmp_x_path,tmp_y_path)
            x[i,]=tmp_x
            y[i,]=tmp_y

        return x,y

    def transformer(self, x_path, y_path):
        tmp_x = cv.imread(x_path)
        if y_path is not None:
            tmp_y = cv.imread(y_path,cv.IMREAD_GRAYSCALE)
        else:
            tmp_y=None
        h,w,_ = tmp_x.shape
        if self.args.model_state == "train":
            if self.args.crop_img:
                i_h = random.randint(0,h-self.dim_h)
                i_w = random.randint(0,w-self.dim_w)
                tmp_x = tmp_x[i_h:i_h+self.dim_h,i_w:i_w+self.dim_w,]
                tmp_y = tmp_y[i_h:i_h+self.dim_h,i_w:i_w+self.dim_w,]
            else:
                tmp_x = cv.resize(tmp_x,(self.dim_w,self.dim_h))
                tmp_y = cv.resize(tmp_y,(self.dim_w,self.dim_h))
        else:
            if self.dim_w!=w and self.dim_h!=h:
                tmp_x = cv.resize(tmp_x, (self.dim_w, self.dim_h))
            if self.args.scale is not None:
                scl = self.args.scale
                scl_h = int(self.dim_h * scl) if (self.dim_h * scl) % 16 == 0 else \
                    int(((self.dim_h * scl) // 16 + 1) * 16)
                scl_w = int(self.dim_w * scl) if (self.dim_w * scl) % 16 == 0 else \
                    int(((self.dim_h * scl) // 16 + 1) * 16)
                tmp_x = cv.resize(tmp_x,dsize=(scl_w,scl_h))
            if tmp_y is not None:
                tmp_y = cv.resize(tmp_y, (self.dim_w, self.dim_h))

        if tmp_y is not None:
            tmp_y = np.expand_dims(np.float32(tmp_y)/255.,axis=-1)
        tmp_x = np.float32(tmp_x)
        return tmp_x, tmp_y

    # def __read_h5(self,file_path):
    #
    #     with h5py.File(file_path,'r') as h5f:
    #         # n_var = len(list(h5f.keys()))
    #         data = np.array(h5f.get('data'))
    #     return data

def dataset_info(dataset_name, is_linux=False):

    if is_linux:

        config = {
            'BSDS': {'img_height':400,# 321
                     'img_width':400,#481
                     'test_list': 'test_pair.lst',
                     'data_dir': '/opt/dataset/BSDS',  # mean_rgb
                        'yita': 0.5},
            'BSDS300': {'img_height': 321,
                     'img_width': 481,
                    'test_list': 'test_pair.lst',
                     'data_dir': '/opt/dataset/BSDS300',  # NIR
                     'yita': 0.5},
            'PASCAL': {'img_height':375,
                        'img_width':500,
                       'test_list': 'test_pair.lst',
                       'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                               'yita': 0.3},
            'CID': {'img_height':512,
                       'img_width':512,
                    'test_list': 'test_pair.lst',
                       'data_dir': '/opt/dataset/CID',  # mean_rgb
                       'yita': 0.3},
            'NYUD': {'img_height':425,
                        'img_width':560,
                     'test_list': 'test_pair.lst',
                        'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                           'yita': 0.5},
            'MULTICUE': {'img_height':720,
                        'img_width':1280,
                         'test_list': 'test_pair.lst',
                         'data_dir': '/opt/dataset/MULTICUE',  # mean_rgb
                              'yita': 0.3},
            'BIPED': {'img_height': 720,
                       'img_width': 1280,
                      'test_list': 'test_rgb.lst',
                      'train_list': 'train_rgb.lst',
                       'data_dir': '/opt/dataset/BIPED/edges',  # WIN: '../.../dataset/BIPED/edges'
                       'yita': 0.5},
            'MBIPED': {'img_height': 720,
                      'img_width': 1280,
                      'test_list': 'test_rgbn.lst',
                      'train_list': 'train_rgbn.lst',
                      'data_dir': '/opt/dataset/BIPED/edges',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5},
            'CLASSIC': {'img_height':512,  # 4032
                      'img_width': 512, # 3024
                      'test_list': None,
                      'data_dir': 'data',  # mean_rgb
                      'yita': 0.5},
            'DCD': {'img_height': 336,# 240
                      'img_width': 448,#360
                    
                      'test_list':'test_pair.lst',
                      'data_dir': '/opt/dataset/DCD',  # mean_rgb
                      'yita': 0.2}
        }
        data_info = config[dataset_name]
        return data_info
    else:
        config = {
            'BSDS': {'img_height': 512,#321
                     'img_width': 512,#481
                     'test_list': 'test_pair.lst',
                     'data_dir': '../../dataset/BSDS',  # mean_rgb
                     'yita': 0.5},
            'BSDS300': {'img_height': 512,#321
                        'img_width': 512,#481
                        'test_list': 'test_pair.lst',
                        'data_dir': '../../dataset/BSDS300',  # NIR
                        'yita': 0.5},
            'PASCAL': {'img_height': 375,
                       'img_width': 500,
                       'test_list': 'test_pair.lst',
                       'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                       'yita': 0.3},
            'CID': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_pair.lst',
                    'data_dir': '../../dataset/CID',  # mean_rgb
                    'yita': 0.3},
            'NYUD': {'img_height': 425,
                     'img_width': 560,
                     'test_list': 'test_pair.lst',
                     'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                     'yita': 0.5},
            'MULTICUE': {'img_height': 720,
                         'img_width': 1280,
                         'test_list': 'test_pair.lst',
                         'data_dir': '../../dataset/MULTICUE',  # mean_rgb
                         'yita': 0.3},
            'BIPED': {'img_height': 720,#720
                      'img_width': 1280,#1280
                      'test_list': 'test_rgb.lst',
                      'train_list': 'train_rgb.lst',
                      'data_dir': '../../dataset/BIPED/edges',  # WIN: '../.../dataset/BIPED/edges'
                      'yita': 0.5},
            'CLASSIC': {'img_height': 512,
                        'img_width': 512,
                        'test_list': None,
                        'train_list': None,
                        'data_dir': 'data',  # mean_rgb
                        'yita': 0.5},
            'DCD': {'img_height': 240,
                    'img_width': 360,
                    'test_list': 'test_pair.lst',
                    'data_dir': '/opt/dataset/DCD',  # mean_rgb
                    'yita': 0.2}
        }
        data_info = config[dataset_name]
        return data_info