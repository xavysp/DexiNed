
from __future__ import print_function
import os
import argparse
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import transforms
import torchgeometry as tgm

from model import DexiNet
from losses import weighted_cross_entropy_loss
# from dexi_utils import visualize_result


class CidDataset(Dataset):
    def __init__(self, data_root, arg = None):
        self.data_root = data_root
        self.transforms = transforms
        self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values)==4\
            else arg.mean_pixel_values

        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        data_root = os.path.abspath(self.data_root)
        images_path = os.path.join(data_root, 'imgs')
        labels_path = os.path.join(data_root, 'gt')
        for file_name_ext in os.listdir(images_path):
            file_name = file_name_ext[:-4]
            sample_indices.append(
                (os.path.join(images_path, file_name + '.png'),
                 os.path.join(labels_path, file_name + '.png'),)
            )
            assert os.path.isfile(sample_indices[-1][0]), sample_indices[-1][0]
            assert os.path.isfile(sample_indices[-1][1]), sample_indices[-1][1]
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        file_name = os.path.basename(image_path)
        
        # load data
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        label = cv.imread(label_path, cv.IMREAD_COLOR)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label, file_names=file_name)

    def transform(self, img, gt):

        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        # gt[gt< 51] = 0 # test without gt discrimination
        gt /= 255.
        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        gt = torch.from_numpy(np.array([gt])).float()
        img = torch.from_numpy(img.copy()).float()
        return img, gt



class BipedMyDataset(Dataset):
    train_modes = ['train', 'test',]
    dataset_types = ['rgbr',]
    data_types = ['aug',]
    def __init__(self, data_root, train_mode='train', dataset_type='rgbr',
                 is_scaling=None, arg=None):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug' # be aware that this might change in the future
        self.scale = is_scaling
        self.arg =arg
        self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
            else arg.mean_pixel_values

        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type
        sample_indices = []
        data_root = os.path.abspath(self.data_root)
        images_path = os.path.join(data_root, 'imgs', self.train_mode,
                                   self.dataset_type, self.data_type)
        labels_path = os.path.join(data_root, 'edge_maps', self.train_mode,
                                   self.dataset_type, self.data_type)
        for directory_name in os.listdir(images_path):
             image_directories = os.path.join(images_path, directory_name)
             for file_name_ext in os.listdir(image_directories):
                 file_name = file_name_ext[:-4]
                 sample_indices.append(
                     (os.path.join(images_path, directory_name, file_name + '.jpg'),
                      os.path.join(labels_path, directory_name, file_name + '.png'),)
                 )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        
        # load data
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        # if self.transforms is not None:
        #     image = self.transforms(image)
        #     label = self.transforms(label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):

        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        # gt[gt< 51] = 0 # test without gt discrimination
        gt /= 255.
        # if self.yita is not None:
        #     gt[gt >= self.yita] = 1
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img -= self.mean_bgr
        # data = []
        # if self.scale is not None:
        #     for scl in self.scale:
        #         img_scale = cv.resize(img, None, fx=scl, fy=scl, interpolation=cv.INTER_LINEAR)
        #         data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
        #     return data, gt


        crop_size = self.arg.img_height if self.arg.img_height == self.arg.img_width else 400

        if self.arg.crop_img:
            _, h, w = gt.size()
            assert (crop_size < h and crop_size < w)
            i = random.randint(0, h - crop_size)
            j = random.randint(0, w - crop_size)
            img = img[:, i:i + crop_size, j:j + crop_size]
            gt = gt[:, i:i + crop_size, j:j + crop_size]
        else:
            img = cv.resize(img, dsize=(self.arg.img_width, self.arg.img_height ))
            gt = cv.resize(gt, dsize=(self.arg.img_width, self.arg.img_height ))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt

def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img

def restore_rgb(config,I, restore_rgb=False):
    """
    :param config: [args.channel_swap, args.mean_pixel_value]
    :param I: and image or a set of images
    :return: an image or a set of images restored
    """

    if  len(I)>3 and not type(I)==np.ndarray:
        I =np.array(I)
        I = I[:,:,:,0:3]
        n = I.shape[0]
        for i in range(n):
            x = I[i,...]
            x = np.array(x, dtype=np.float32)
            x += config[1]
            if restore_rgb:
                x = x[:, :, config[0]]
            x = image_normalization(x)
            I[i,:,:,:]=x
    elif len(I.shape)==3 and I.shape[-1]==3:
        I = np.array(I, dtype=np.float32)
        I += config[1]
        if restore_rgb:
            I = I[:, :, config[0]]
        I = image_normalization(I)
    else:
        print("Sorry the input data size is out of our configuration")
    # print("The enterely I data {} restored".format(I.shape))
    return I

def visualize_result(imgs_list, arg):
    """
    function for tensorflow results
    :param imgs_list: a list of prediction, gt and input data
    :param arg:
    :return: one image with the whole of imgs_list data
    """
    n_imgs = len(imgs_list)
    data_list =[]
    for i in range(n_imgs):
        tmp = imgs_list[i]
        if tmp.shape[1]==3:
            tmp = np.transpose(np.squeeze(tmp[1]),[1,2,0])
            tmp=restore_rgb([arg.channel_swap,arg.mean_pixel_values[:3]],tmp)
            tmp = np.uint8(image_normalization(tmp))
        else:
            tmp= np.squeeze(tmp[1])
            if len(tmp.shape) == 2:
                tmp = np.uint8(image_normalization(tmp))
                tmp = cv.bitwise_not(tmp)
                tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
            else:
                tmp = np.uint8(image_normalization(tmp))
        data_list.append(tmp)
    img = data_list[0]
    if n_imgs % 2 == 0:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * (n_imgs // 2) + ((n_imgs // 2 - 1) * 5), 3))
    else:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * ((1 + n_imgs) // 2) + ((n_imgs // 2) * 5), 3))
        n_imgs += 1

    k=0
    imgs = np.uint8(imgs)
    i_step = img.shape[0]+10
    j_step = img.shape[1]+5
    for i in range(2):
        for j in range(n_imgs//2):
            if k<len(data_list):
                imgs[i*i_step:i*i_step+img.shape[0],j*j_step:j*j_step+img.shape[1],:]=data_list[k]
                k+=1
            else:
                pass
    return imgs

def create_directory(dir_path):
    """Creates an empty directory.
    Args:
        dir_path (str): the absolute path to the directory to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train(epoch, dataloader, model, criterion, optimizer, device,
          log_interval_vis, tb_writer, args=None):
    imgs_res_folder =os.path.join(args.output_dir,'current_res')
    create_directory(imgs_res_folder)
    model.train()
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxHxW
        # labels = labels[:, None]  # Bx1xHxW

        preds_list = model(images)
        loss = sum([criterion(preds, labels) for preds in preds_list])
        loss /= images.shape[0]  # the batch size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id%5==0:
            print('Epoch: {0} Sample {1}/{2} Loss: {3}' \
              .format(epoch, batch_id, len(dataloader), loss.item()))

        if tb_writer is not None:
            tb_writer.add_scalar('data/loss', loss.detach(), batch_id)

        if batch_id % log_interval_vis == 0:
            #import ipdb;ipdb.set_trace()
            # log images
            # images_vis = torchvision.utils.make_grid(images[:16], 4, 4)
            # images_vis = tgm.utils.tensor_to_image(images_vis)
            # cv.namedWindow('images', cv.WINDOW_NORMAL)
            # cv.imshow('images', images_vis)
            # # log ground truth
            # labels_vis = torchvision.utils.make_grid(labels[:16], 4, 4)
            # labels_vis = tgm.utils.tensor_to_image(labels_vis)
            # cv.namedWindow('edges', cv.WINDOW_NORMAL)
            # cv.imshow('edges', labels_vis)
            # # log prediction
            # edges_vis = torchvision.utils.make_grid(preds_list[-1][:16], 4, 4)
            # edges_vis = tgm.utils.tensor_to_image(edges_vis * 255.).astype(np.uint8)
            # cv.namedWindow('edges_pred', cv.WINDOW_NORMAL)
            # cv.imshow('edges_pred', edges_vis)
            # cv.waitKey(3)
            res_data = []
            img = images.cpu().numpy()
            res_data.append(img)
            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt)
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = torch.sigmoid(tmp)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)
            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data
            vis_imgs = cv.resize(vis_imgs,(int(vis_imgs.shape[1]*0.8),int(vis_imgs.shape[0]*0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
              .format(epoch, batch_id, len(dataloader), loss.item())
            BLACK = (0, 0, 255)
            font = cv.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv.putText(vis_imgs, img_test, (x, y), font, font_size, font_color, font_thickness, cv.LINE_AA)
            cv.imwrite(os.path.join(imgs_res_folder,'results.png'),vis_imgs)


def save_image_batch_to_disk(tensor, output_dir, file_names, ext='.png'):
    assert len(tensor.shape) == 4, tensor.shape

    for tensor_image, file_name in zip(tensor, file_names):
        image_vis = tgm.utils.tensor_to_image(tensor_image)[..., 0]
        image_vis = (image_vis * 255.).astype(np.uint8)
        output_file_name = os.path.join(output_dir, file_name)
        print(output_file_name)
        assert cv.imwrite(output_file_name, image_vis)
                

def validation(epoch, dataloader, model, device, output_dir):
    model.eval()
    total_losses = []

    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)
        labels = sample_batched['labels'].to(device)
        file_names = sample_batched['file_names']
        
        output = model(images)
        save_image_batch_to_disk(output[-1], output_dir, file_names)


def weight_init(m):
    # torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, (nn.Conv2d, )):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.weight.data.shape[1]==torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)
#            print(m.weight)
        if m.weight.data.shape==torch.Size([1,6,1,1]):
            torch.nn.init.constant_(m.weight,0.2)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    # for fusion layer
            
    # if isinstance(m, (nn.ConvTranspose2d,)):
    #     if m.weight.data.shape[:2] == torch.Size([1,1]):
    #         torch.nn.init.normal_(m.weight,std=0.1)
    #
    #     if m.bias is not None:
    #         torch.nn.init.zeros_(m.bias)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Training application.')
    # Data parameters
    parser.add_argument('--input-dir', type=str,default='/opt/dataset/BIPED/edges',
                        help='the path to the directory with the input data.')
    parser.add_argument('--input-val-dir', type=str,default='/opt/dataset/CID',
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='the path to output the results.')
    parser.add_argument('--test_data', type=str, default='CID',
                        help='Name of the dataset.')
    parser.add_argument('--log-interval-vis', type=int, default=50,
                        help='how many batches to wait before logging training status')
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                        help='the optimization solver to use (default: adam)')
    parser.add_argument('--num-epochs', type=int, default=20, metavar='N',
                        help='number of training epochs (default: 100)')
    # parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
    #                     help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-5, metavar='WD',
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_stepsize', default=1e4, type=int,
                        help='Learning rate step size.')
    parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                        help='the mini-batch size (default: 2)')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='the number of workers for the dataloader.')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='use tensorboard for logging purposes'),
    parser.add_argument('--gpu', type=str, default='1',
                        help='select GPU'),
    parser.add_argument('--img_width', type = int, default = 400, help='image size for training')
    parser.add_argument('--img_height', type = int, default = 400, help='image size for training')
    parser.add_argument('--channel_swap', default=[2, 1, 0], type=int)
    parser.add_argument('--crop_img', default=False, type=bool,
                        help='If true crop training images, other ways resizing')
    parser.add_argument('--mean_pixel_values', default=[104.00699, 116.66877, 122.67892, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    args = parser.parse_args()

    tb_writer = None
    if args.tensorboard:
        from tensorboardX import SummaryWriter # previous torch version
        # from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir=args.output_dir)
    print(" **** You have available ", torch.cuda.device_count(), "GPUs!")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
    model = DexiNet().to(device)
    # model = nn.DataParallel(model)
    model.apply(weight_init)

    # height, width = 400, 400
    # transformations_train = transforms.Compose([
    #     transforms.Lambda(lambda img: cv.resize(img, (width, height))),
    #     tgm.utils.image_to_tensor,
    #     transforms.Lambda(lambda tensor: (tensor.float()-arg.mean_pixel_values[0:3]) / 255.),
    # ])
    # transformations_val = transforms.Compose([
    #     tgm.utils.image_to_tensor,
    #     transforms.Lambda(lambda tensor: (tensor.float()-arg.mean_pixel_values[0:3]) / 255.),
    # ])

    dataset_train = BipedMyDataset(args.input_dir, train_mode='train',
                                  arg=args)
    dataset_val = CidDataset(args.input_val_dir, arg=args)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    criterion = weighted_cross_entropy_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler.
    # lr_schd = lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize,
    #                               gamma=args.lr_gamma)

    for epoch in range(args.num_epochs):
        # Create output directory
        output_dir_epoch = os.path.join(args.output_dir, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch,args.test_data+'_res')
        create_directory(output_dir_epoch)
        create_directory(img_test_dir)

        train(epoch, dataloader_train, model, criterion, optimizer, device,
              args.log_interval_vis, tb_writer, args=args)

        # lr_schd.step() # decay lr at the end of the epoch.
    
        with torch.no_grad():
            validation(epoch, dataloader_val, model, device, img_test_dir)

        try:
            net_state_dict = model.module.state_dict()
        except:
            net_state_dict = model.state_dict()

        torch.save(net_state_dict, os.path.join(
                   output_dir_epoch, '{0}_model.pth'.format(epoch)))

        
if __name__ == '__main__':
    main()
