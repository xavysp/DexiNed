import os

import cv2
import numpy as np
import torch
import kornia as kn


def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def count_parameters(model=None):
    if model is not None:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        print("Error counting model parameters line 32 img_processing.py")
        raise NotImplementedError

def save_image_batch_to_disk(tensor, output_dir, file_names, img_shape=None, arg=None, is_inchannel=False):

    os.makedirs(output_dir, exist_ok=True)
    if not arg.is_testing:
        assert len(tensor.shape) == 4, tensor.shape
        img_shape = tensor.shape
        img_height, img_width = img_shape[2], img_shape[3]
        for tensor_image, file_name in zip(tensor, file_names):
            image_vis = kn.utils.tensor_to_image(
                torch.sigmoid(tensor_image))#[..., 0]
            image_vis = (255.0*(1.0 - image_vis)).astype(np.uint8)
            output_file_name = os.path.join(output_dir, file_name)
            image_vis =cv2.resize(image_vis, (img_width, img_height))
            assert cv2.imwrite(output_file_name, image_vis)
    else:
        if is_inchannel:

            tensor, tensor2 = tensor
            fuse_name = 'fusedCH'
            av_name='avgCH'
            is_2tensors=True
            edge_maps2 = []
            for i in tensor2:
                tmp = torch.sigmoid(i).cpu().detach().numpy()
                edge_maps2.append(tmp)
            tensor2 = np.array(edge_maps2)
        else:
            fuse_name = 'fused'
            av_name = 'avg'
            tensor2=None
            tmp_img2 = None

        output_dir_f = os.path.join(output_dir, fuse_name)
        output_dir_a = os.path.join(output_dir, av_name)
        os.makedirs(output_dir_f, exist_ok=True)
        os.makedirs(output_dir_a, exist_ok=True)

        # 255.0 * (1.0 - em_a)
        edge_maps = []
        for i in tensor:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            edge_maps.append(tmp)
        tensor = np.array(edge_maps)
        # print(f"tensor shape: {tensor.shape}")

        image_shape = [x.cpu().detach().numpy() for x in img_shape]
        # (H, W) -> (W, H)
        image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

        assert len(image_shape) == len(file_names)

        idx = 0
        for i_shape, file_name in zip(image_shape, file_names):
            tmp = tensor[:, idx, ...]
            tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
            # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
            tmp = np.squeeze(tmp)
            tmp2 = np.squeeze(tmp2) if tensor2 is not None else None

            # Iterate our all 7 NN outputs for a particular image
            preds = []
            for i in range(tmp.shape[0]):
                tmp_img = tmp[i]
                tmp_img = np.uint8(image_normalization(tmp_img))
                tmp_img = cv2.bitwise_not(tmp_img)
                # tmp_img[tmp_img < 0.0] = 0.0
                # tmp_img = 255.0 * (1.0 - tmp_img)
                if tmp2 is not None:
                    tmp_img2 = tmp2[i]
                    tmp_img2 = np.uint8(image_normalization(tmp_img2))
                    tmp_img2 = cv2.bitwise_not(tmp_img2)

                # Resize prediction to match input image size
                if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                    tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
                    tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None


                if tmp2 is not None:
                    tmp_mask = np.logical_and(tmp_img>128,tmp_img2<128)
                    tmp_img= np.where(tmp_mask, tmp_img2, tmp_img)
                    preds.append(tmp_img)

                else:
                    preds.append(tmp_img)

                if i == 6:
                    fuse = tmp_img
                    fuse = fuse.astype(np.uint8)
                    if tmp_img2 is not None:
                        fuse2 = tmp_img2
                        fuse2 = fuse2.astype(np.uint8)
                        # fuse = fuse-fuse2
                        fuse_mask=np.logical_and(fuse>128,fuse2<128)
                        fuse = np.where(fuse_mask,fuse2, fuse)

                        # print(fuse.shape, fuse_mask.shape)

            # Get the mean prediction of all the 7 outputs
            average = np.array(preds, dtype=np.float32)
            average = np.uint8(np.mean(average, axis=0))
            output_file_name_f = os.path.join(output_dir_f, file_name)
            output_file_name_a = os.path.join(output_dir_a, file_name)
            cv2.imwrite(output_file_name_f, fuse)
            cv2.imwrite(output_file_name_a, average)

            idx += 1


def restore_rgb(config, I, restore_rgb=False):
    """
    :param config: [args.channel_swap, args.mean_pixel_value]
    :param I: and image or a set of images
    :return: an image or a set of images restored
    """

    if len(I) > 3 and not type(I) == np.ndarray:
        I = np.array(I)
        I = I[:, :, :, 0:3]
        n = I.shape[0]
        for i in range(n):
            x = I[i, ...]
            x = np.array(x, dtype=np.float32)
            x += config[1]
            if restore_rgb:
                x = x[:, :, config[0]]
            x = image_normalization(x)
            I[i, :, :, :] = x
    elif len(I.shape) == 3 and I.shape[-1] == 3:
        I = np.array(I, dtype=np.float32)
        I += config[1]
        if restore_rgb:
            I = I[:, :, config[0]]
        I = image_normalization(I)
    else:
        print("Sorry the input data size is out of our configuration")
    return I


def visualize_result(imgs_list, arg):
    """
    data 2 image in one matrix
    :param imgs_list: a list of prediction, gt and input data
    :param arg:
    :return: one image with the whole of imgs_list data
    """
    n_imgs = len(imgs_list)
    data_list = []
    for i in range(n_imgs):
        tmp = imgs_list[i]
        # print(tmp.shape)
        if tmp.shape[0] == 3:
            tmp = np.transpose(tmp, [1, 2, 0])
            tmp = restore_rgb([
                arg.channel_swap,
                arg.mean_pixel_values[:3]
            ], tmp)
            tmp = np.uint8(image_normalization(tmp))
        else:
            tmp = np.squeeze(tmp)
            if len(tmp.shape) == 2:
                tmp = np.uint8(image_normalization(tmp))
                tmp = cv2.bitwise_not(tmp)
                tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
            else:
                tmp = np.uint8(image_normalization(tmp))
        data_list.append(tmp)
        # print(i,tmp.shape)
    img = data_list[0]
    if n_imgs % 2 == 0:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1]
                         * (n_imgs // 2) + ((n_imgs // 2 - 1) * 5), 3))
    else:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1]
                         * ((1 + n_imgs) // 2) + ((n_imgs // 2) * 5), 3))
        n_imgs += 1

    k = 0
    imgs = np.uint8(imgs)
    i_step = img.shape[0] + 10
    j_step = img.shape[1] + 5
    for i in range(2):
        for j in range(n_imgs // 2):
            if k < len(data_list):
                imgs[i * i_step:i * i_step+img.shape[0],
                     j * j_step:j * j_step+img.shape[1],
                     :] = data_list[k]
                k += 1
            else:
                pass
    return imgs