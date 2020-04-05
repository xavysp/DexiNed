
from __future__ import print_function
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import transforms
import torchgeometry as tgm

from model import DexiNet
from losses import weighted_cross_entropy_loss


class CidDataset(Dataset):
    def __init__(self, data_root, transforms=None, arg = None):
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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)

        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)
        return dict(images=image, labels=label, file_names=file_name)


class BipedMyDataset(Dataset):
    train_modes = ['train', 'test',]
    dataset_types = ['rgbr',]
    data_types = ['aug',]
    def __init__(self, data_root, train_mode='train', dataset_type='rgbr',
                 transforms=None, arg=None):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug' # be aware that this might change in the future
        self.transforms = transforms
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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)
        return dict(images=image, labels=label)

    # def transform(self, img, gt):
    #
    #     gt = np.array(gt, dtype=np.float32)
    #     if len(gt.shape) == 3:
    #         gt = gt[:, :, 0]
    #     gt /= 255.
    #     if self.yita is not None:
    #         gt[gt >= self.yita] = 1
    #     gt = torch.from_numpy(np.array([gt])).float()
    #     img = np.array(img, dtype=np.float32)
    #     if self.rgb:
    #         img = img[:, :, ::-1]  # RGB->BGR
    #     img -= self.mean_bgr
    #     data = []
    #     if self.scale is not None and self.is_train:
    #         for scl in self.scale:
    #             img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
    #             data.append(torch.from_numpy(img_scale.transpose((2, 0, 1))).float())
    #         return data, gt
    #     img = img.transpose((2, 0, 1))
    #     img = torch.from_numpy(img.copy()).float()
    #     if self.crop_size:
    #         _, h, w = gt.size()
    #         assert (self.crop_size < h and self.crop_size < w)
    #         i = random.randint(0, h - self.crop_size)
    #         j = random.randint(0, w - self.crop_size)
    #         img = img[:, i:i + self.crop_size, j:j + self.crop_size]
    #         gt = gt[:, i:i + self.crop_size, j:j + self.crop_size]
    #         return img, gt

def train(epoch, dataloader, model, criterion, optimizer, device,
          log_interval_vis, tb_writer):
    model.train()
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxHxW
        labels = labels[:, None]  # Bx1xHxW

        preds_list = model(images)
        loss = sum([criterion(preds, labels) for preds in preds_list])
        loss /= images.shape[0]  # the batch size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: {0} Sample {1}/{2} Loss: {3}' \
              .format(epoch, batch_id, len(dataloader), loss.item()))

        if tb_writer is not None:
            tb_writer.add_scalar('data/loss', loss.detach(), batch_id)

        if batch_id % log_interval_vis == 0:
            #import ipdb;ipdb.set_trace()
            # log images
            # images_vis = torchvision.utils.make_grid(images[:16], 4, 4)
            # images_vis = tgm.utils.tensor_to_image(images_vis)
            # cv2.namedWindow('images', cv2.WINDOW_NORMAL)
            # cv2.imshow('images', images_vis)
            # # log ground truth
            # labels_vis = torchvision.utils.make_grid(labels[:16], 4, 4)
            # labels_vis = tgm.utils.tensor_to_image(labels_vis)
            # cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
            # cv2.imshow('edges', labels_vis)
            # # log prediction
            # edges_vis = torchvision.utils.make_grid(preds_list[-1][:16], 4, 4)
            # edges_vis = tgm.utils.tensor_to_image(edges_vis * 255.).astype(np.uint8)
            # cv2.namedWindow('edges_pred', cv2.WINDOW_NORMAL)
            # cv2.imshow('edges_pred', edges_vis)
            # cv2.waitKey(3)
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





def create_directory(dir_path):
    """Creates an empty directory.
    Args:
        dir_path (str): the absolute path to the directory to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_image_batch_to_disk(tensor, output_dir, file_names, ext='.png'):
    assert len(tensor.shape) == 4, tensor.shape

    for tensor_image, file_name in zip(tensor, file_names):
        image_vis = tgm.utils.tensor_to_image(tensor_image)[..., 0]
        image_vis = (image_vis * 255.).astype(np.uint8)
        output_file_name = os.path.join(output_dir, file_name + ext)
        assert cv2.imwrite(output_file_name, image_vis)
                

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
    parser.add_argument('--log-interval-vis', type=int, default=100,
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
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='use tensorboard for logging purposes'),
    parser.add_argument('--gpu', type=str, default='1',
                        help='select GPU'),
    parser.add_argument('--img_width', type = int, default = 400, help='image size for training')
    parser.add_argument('--img_height', type = int, default = 400, help='image size for training')
    parser.add_argument('--channel_swap', default=[2, 1, 0], type=int)
    parser.add_argument('--mean_pixel_values', default=[104.00699, 116.66877, 122.67892, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    args = parser.parse_args()

    tb_writer = None
    if args.tensorboard:
#        from tensorboardX import SummaryWriter # previous torch version
        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        
        tb_writer = SummaryWriter(log_dir=args.output_dir)
    print(" **** You have available ", torch.cuda.device_count(), "GPUs!")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
    model = DexiNet().to(device)
    # model = nn.DataParallel(model)
    model.apply(weight_init)

    height, width = 400, 400
    transformations_train = transforms.Compose([
        transforms.Lambda(lambda img: cv2.resize(img, (width, height))),
        tgm.utils.image_to_tensor,
        transforms.Lambda(lambda tensor: (tensor.float()-arg.mean_pixel_values[0:3]) / 255.),
    ])
    transformations_val = transforms.Compose([
        tgm.utils.image_to_tensor,
        transforms.Lambda(lambda tensor: (tensor.float()-arg.mean_pixel_values[0:3]) / 255.),
    ])

    dataset_train = BipedMyDataset(args.input_dir, train_mode='train',
                                  transforms=transformations_train)
    dataset_val = CidDataset(args.input_val_dir, transforms=transformations_val)

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
        create_directory(output_dir_epoch)

        train(epoch, dataloader_train, model, criterion, optimizer, device,
              args.log_interval_vis, tb_writer)

        # lr_schd.step() # decay lr at the end of the epoch.
    
        with torch.no_grad():
            validation(epoch, dataloader_val, model, device, output_dir_epoch)

        try:
            net_state_dict = model.module.state_dict()
        except:
            net_state_dict = model.state_dict()

        torch.save(net_state_dict, os.path.join(
                   output_dir_epoch, '{0}_model.pth'.format(epoch)))

        
if __name__ == '__main__':
    main()
