import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json

DATASET_NAMES = [
    'BIPED',
    'BSDS',
    'BRIND',
    'BSDS300',
    'CID',
    'DCD',
    'MDBD', #5
    'PASCAL',
    'NYUD',
    'CLASSIC'
]  # 8


def dataset_info(dataset_name, is_linux=True):
    if is_linux:

        config = {
            'BSDS': {
                'img_height': 512, #321
                'img_width': 512, #481
                'train_list': 'train_pair.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BSDS',  # mean_rgb
                'yita': 0.5
            },
            'BRIND': {
                'img_height': 512,  # 321
                'img_width': 512,  # 481
                'train_list': 'train_pair2.lst',
                'test_list': 'test_pair.lst',
                'data_dir': '/opt/dataset/BRIND',  # mean_rgb
                'yita': 0.5
            },
            'BSDS300': {
                'img_height': 512, #321
                'img_width': 512, #481
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/BSDS300',  # NIR
                'yita': 0.5
            },
            'PASCAL': {
                'img_height': 416, # 375
                'img_width': 512, #500
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/PASCAL',  # mean_rgb
                'yita': 0.3
            },
            'CID': {
                'img_height': 512,
                'img_width': 512,
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/CID',  # mean_rgb
                'yita': 0.3
            },
            'NYUD': {
                'img_height': 448,#425
                'img_width': 560,#560
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/NYUD',  # mean_rgb
                'yita': 0.5
            },
            'MDBD': {
                'img_height': 720,
                'img_width': 1280,
                'test_list': 'test_pair.lst',
                'train_list': 'train_pair.lst',
                'data_dir': '/opt/dataset/MDBD',  # mean_rgb
                'yita': 0.3
            },
            'BIPED': {
                'img_height': 720, #720 # 1088
                'img_width': 1280, # 1280 5 1920
                'test_list': 'test_pair.lst',
                'train_list': 'train_rgb.lst',
                'data_dir': '/opt/dataset/BIPED',  # mean_rgb
                'yita': 0.5
            },
            'CLASSIC': {
                'img_height': 512,
                'img_width': 512,
                'test_list': None,
                'train_list': None,
                'data_dir': 'data',  # mean_rgb
                'yita': 0.5
            },
            'DCD': {
                'img_height': 352, #240
                'img_width': 480,# 360
                'test_list': 'test_pair.lst',
                'train_list': None,
                'data_dir': '/opt/dataset/DCD',  # mean_rgb
                'yita': 0.2
            }
        }
    else:
        config = {
            'BSDS': {'img_height': 512,  # 321
                     'img_width': 512,  # 481
                     'test_list': 'test_pair.lst',
                     'train_list': 'train_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/BSDS',  # mean_rgb
                     'yita': 0.5},
            'BSDS300': {'img_height': 512,  # 321
                        'img_width': 512,  # 481
                        'test_list': 'test_pair.lst',
                        'data_dir': 'C:/Users/xavysp/dataset/BSDS300',  # NIR
                        'yita': 0.5},
            'PASCAL': {'img_height': 375,
                       'img_width': 500,
                       'test_list': 'test_pair.lst',
                       'data_dir': 'C:/Users/xavysp/dataset/PASCAL',  # mean_rgb
                       'yita': 0.3},
            'CID': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_pair.lst',
                    'data_dir': 'C:/Users/xavysp/dataset/CID',  # mean_rgb
                    'yita': 0.3},
            'NYUD': {'img_height': 425,
                     'img_width': 560,
                     'test_list': 'test_pair.lst',
                     'data_dir': 'C:/Users/xavysp/dataset/NYUD',  # mean_rgb
                     'yita': 0.5},
            'MDBD': {'img_height': 720,
                         'img_width': 1280,
                         'test_list': 'test_pair.lst',
                         'train_list': 'train_pair.lst',
                         'data_dir': 'C:/Users/xavysp/dataset/MDBD',  # mean_rgb
                         'yita': 0.3},
            'BIPED': {'img_height': 720,  # 720
                      'img_width': 1280,  # 1280
                      'test_list': 'test_pair.lst',
                      'train_list': 'train_rgb.lst',
                      'data_dir': 'C:/Users/xavysp/dataset/BIPED',  # WIN: '../.../dataset/BIPED/edges'
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
                    'data_dir': 'C:/Users/xavysp/dataset/DCD',  # mean_rgb
                    'yita': 0.2}
        }
    return config[dataset_name]


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None
                 ):
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args=arg
        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        sample_indices = []
        if self.test_data == "CLASSIC":
            # for single image testing
            images_path = os.listdir(self.data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        else:
            # image and label paths are located in a list file

            if not self.test_list:
                raise ValueError(
                    f"Test list not provided for dataset: {self.test_data}")

            list_name = os.path.join(self.data_root, self.test_list)
            if self.test_data.upper()=='BIPED':

                with open(list_name) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
            else:
                with open(list_name, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]
                pairs = [line.split() for line in files]

                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))
        return sample_indices

    def __len__(self):
        return len(self.data_index[0]) if self.test_data.upper()=='CLASSIC' else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx]
        else:
            image_path = self.data_index[idx][0]
        label_path = None if self.test_data == "CLASSIC" else self.data_index[idx][1]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        # base dir
        if self.test_data.upper() == 'BIPED':
            img_dir = os.path.join(self.data_root, 'imgs', 'test')
            gt_dir = os.path.join(self.data_root, 'edge_maps', 'test')
        elif self.test_data.upper() == 'CLASSIC':
            img_dir = self.data_root
            gt_dir = None
        else:
            img_dir = self.data_root
            gt_dir = self.data_root

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        if not self.test_data == "CLASSIC":
            label = cv2.imread(os.path.join(
                gt_dir, label_path), cv2.IMREAD_COLOR)
        else:
            label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {( img_height,img_width,)}")
            # img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.resize(img, (img_width,img_height))
            gt = None

        # Make images and labels at least 512 by 512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height)) # 512
            gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height)) # 512

        # Make sure images and labels are divisible by 2^4=16
        elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))
        else:
            img_width =self.args.test_img_width
            img_height =self.args.test_img_height
            img = cv2.resize(img, (img_width, img_height))
            gt = cv2.resize(gt, (img_width, img_height))

        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        # img=cv2.resize(img, (400, 464))
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


class BipedDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 arg=None
                 ):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = arg

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        if self.arg.train_data.lower()=='biped':

            images_path = os.path.join(data_root,
                                       'edges/imgs',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)
            labels_path = os.path.join(data_root,
                                       'edges/edge_maps',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)

            for directory_name in os.listdir(images_path):
                image_directories = os.path.join(images_path, directory_name)
                for file_name_ext in os.listdir(image_directories):
                    file_name = os.path.splitext(file_name_ext)[0]
                    sample_indices.append(
                        (os.path.join(images_path, directory_name, file_name + '.jpg'),
                         os.path.join(labels_path, directory_name, file_name + '.png'),)
                    )
        else:
            file_path = os.path.join(data_root, self.arg.train_list)
            if self.arg.train_data.lower()=='bsds':

                with open(file_path, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root,tmp_img),
                         os.path.join(data_root,tmp_gt),))
            else:
                with open(file_path) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        i_h, i_w,_ = img.shape
        # data = []
        # if self.scale is not None:
        #     for scl in self.scale:
        #         img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
        #         data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
        #     return data, gt
        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else None#448# MDBD=480 BIPED=480/400 BSDS=352

        # for BSDS 352/BRIND
        if i_w> crop_size and i_h>crop_size:
            i = random.randint(0, i_h - crop_size)
            j = random.randint(0, i_w - crop_size)
            img = img[i:i + crop_size , j:j + crop_size ]
            gt = gt[i:i + crop_size , j:j + crop_size ]

        # # for BIPED/MDBD
        # if np.random.random() > 0.4: #l
        #     h,w = gt.shape
        #     if i_w> 500 and i_h>500:
        #
        #         LR_img_size = crop_size #l BIPED=256, 240 200 # MDBD= 352 BSDS= 176
        #         i = random.randint(0, h - LR_img_size)
        #         j = random.randint(0, w - LR_img_size)
        #         # if img.
        #         img = img[i:i + LR_img_size , j:j + LR_img_size ]
        #         gt = gt[i:i + LR_img_size , j:j + LR_img_size ]
        #     else:
        #         LR_img_size = 352#256  # l BIPED=208-352, # MDBD= 352-480- BSDS= 176-320
        #         i = random.randint(0, h - LR_img_size)
        #         j = random.randint(0, w - LR_img_size)
        #         # if img.
        #         img = img[i:i + LR_img_size, j:j + LR_img_size]
        #         gt = gt[i:i + LR_img_size, j:j + LR_img_size]
        #         img = cv2.resize(img, dsize=(crop_size, crop_size), )
        #         gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        else:
            # New addidings
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # BRIND
        gt[gt > 0.1] +=0.2#0.4
        gt = np.clip(gt, 0., 1.)
        # gt[gt > 0.1] =1#0.4
        # gt = np.clip(gt, 0., 1.)
        # # for BIPED
        # gt[gt > 0.2] += 0.6# 0.5 for BIPED
        # gt = np.clip(gt, 0., 1.) # BIPED
        # # for MDBD
        # gt[gt > 0.1] +=0.7
        # gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt
