
from __future__ import print_function

import argparse
import os
import sys
import time

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import DATASET_NAMES, BipedDataset, TestDataset, dataset_info
from losses import weighted_cross_entropy_loss
from model import DexiNet
from utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result)


def create_directory(dir_path):
    """Creates an empty directory.

    Args:
        dir_path (str): the absolute path to the directory to create.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device,
                    log_interval_vis, tb_writer, args=None):
    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    create_directory(imgs_res_folder)

    # Put model in training mode
    model.train()

    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW
        labels = sample_batched['labels'].to(device)  # BxHxW
        # labels = labels[:, None]  # Bx1xHxW

        preds_list = model(images)

        loss = sum([criterion(preds, labels) for preds in preds_list])
        loss /= images.shape[0]  #batch size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tb_writer is not None:
            tb_writer.add_scalar('loss',
                                 loss.detach(),
                                 (len(dataloader) * epoch + batch_id))

        if batch_id % 5 == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), loss.item()))
        if batch_id % log_interval_vis == 0:
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

            vis_imgs = cv2.resize(vis_imgs,
                                  (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
                .format(epoch, batch_id, len(dataloader), loss.item())

            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)


def validate_one_epoch(epoch, dataloader, model, device, output_dir, arg=None):
    # XXX This is not really validation, but testing

    # Put model in eval mode
    model.eval()

    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            # labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            preds = model(images)
            save_image_batch_to_disk(preds[-1],
                                     output_dir,
                                     file_names,
                                     arg=arg)


def test(checkpoint_path, dataloader, model, device, output_dir, args):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            if not args.test_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")

            preds = model(images)
            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args)

    print(f"Testing ended for {args.test_data} dataset.")


def parse_args():
    """Parse command line arguments."""

    # Testing settings
    TEST_DATA = DATASET_NAMES[3] # max 8
    data_inf = dataset_info(TEST_DATA)

    parser = argparse.ArgumentParser(description='DexiNed trainer.')
    # Data parameters
    parser.add_argument('--input_dir',
                        type=str,
                        default='/opt/dataset/BIPED/edges',
                        help='the path to the directory with the input data.')
    parser.add_argument('--input_val_dir',
                        type=str,
                        default=data_inf['data_dir'],
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='checkpoints',
                        help='the path to output the results.')
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_list',
                        type=str,
                        default=data_inf['file_name'],
                        help='Dataset sample indices list.')
    parser.add_argument('--is_testing',type=bool,
                        default=False,
                        help='Put script in testing mode.')
    # parser.add_argument('--use_prev_trained',
    #                     type=bool,
    #                     default=True,
    #                     help='use previous trained data')  # Just for test
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='24/24_model.pth',
                        help='Checkpoint path from which to restore model weights from.')
    parser.add_argument('--test_img_width',
                        type=int,
                        default=data_inf['img_width'],
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default=data_inf['img_height'],
                        help='Image height for testing.')
    parser.add_argument('--res_dir',
                        type=str,
                        default='result',
                        help='Result directory')
    parser.add_argument('--log_interval_vis',
                        type=int,
                        default=50,
                        help='The number of batches to wait before printing test predictions.')

    # Optimization parameters
    # parser.add_argument('--optimizer',
    #                     type=str,
    #                     choices=['adam', 'sgd'],
    #                     default='adam',
    #                     help='The optimizer to use (default: adam).')
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        metavar='N',
                        help='Number of training epochs (default: 25).')
    parser.add_argument('--lr',
                        default=1e-4,
                        type=float,
                        help='Initial learning rate.')
    parser.add_argument('--wd',
                        type=float,
                        default=1e-4,
                        metavar='WD',
                        help='weight decay (default: 1e-4)')
    # parser.add_argument('--lr_stepsize',
    #                     default=1e4,
    #                     type=int,
    #                     help='Learning rate step size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard',type=bool,
                        default=True,
                        help='Use Tensorboard for logging.'),
    parser.add_argument('--img_width',
                        type=int,
                        default=400,
                        help='Image width for training.')
    parser.add_argument('--img_height',
                        type=int,
                        default=400,
                        help='Image height for training.')
    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)
    parser.add_argument('--crop_img',
                        default=False,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_pixel_values',
                        default=[103.939,116.779,123.68, 137.86],
                        type=float)  # [103.939,116.779,123.68] [104.00699, 116.66877, 122.67892]
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    # Tensorboard summary writer

    tb_writer = None
    if args.tensorboard and not args.is_testing:
        # from tensorboardX import SummaryWriter  # previous torch version
        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    # Instantiate model and move it to the computing device
    model = DexiNet().to(device)
    # model = nn.DataParallel(model)

    if not args.is_testing:
        dataset_train = BipedDataset(args.input_dir,
                                     img_width=args.img_width,
                                     img_height=args.img_height,
                                     mean_bgr=args.mean_pixel_values[0:3] if len(
                                         args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                     train_mode='train',
                                     #    arg=args
                                     )
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)

    dataset_val = TestDataset(args.input_val_dir,
                              test_data=args.test_data,
                              img_width=args.test_img_width,
                              img_height=args.test_img_height,
                              mean_bgr=args.mean_pixel_values[0:3] if len(
                                  args.mean_pixel_values) == 4 else args.mean_pixel_values,
                              test_list=args.test_list
                              )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)
    # Testing
    if args.is_testing:
        checkpoint_path = os.path.join(args.output_dir, args.checkpoint_data)
        output_dir = os.path.join(args.res_dir, "BIPED2" + args.test_data)
        print(f"output_dir: {output_dir}")

        test(checkpoint_path, dataloader_val, model, device, output_dir, args)
        return

    # Criterion, optimizer, lr scheduler
    criterion = weighted_cross_entropy_loss
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)
    # lr_schd = lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize,
    #                               gamma=args.lr_gamma)

    # Main training loop
    for epoch in range(args.epochs):
        # Create output directories
        output_dir_epoch = os.path.join(args.output_dir, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.test_data + '_res')
        create_directory(output_dir_epoch)
        create_directory(img_test_dir)

        train_one_epoch(epoch,
                        dataloader_train,
                        model,
                        criterion,
                        optimizer,
                        device,
                        args.log_interval_vis,
                        tb_writer,
                        args=args)
        validate_one_epoch(epoch,
                           dataloader_val,
                           model,
                           device,
                           img_test_dir,
                           arg=args)

        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
