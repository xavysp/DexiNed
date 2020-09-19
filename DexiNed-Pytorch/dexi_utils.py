import cv2 as cv
import numpy as np
import torch


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
    epsilon = 1e-12  # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min) / \
        ((np.max(img)-np.min(img))+epsilon)+img_min
    return img


# def visualize_result(imgs_list, arg):
#     """
#     function for tensorflow results
#     :param imgs_list: a list of prediction, gt and input data
#     :param arg:
#     :return: one image with the whole of imgs_list data
#     """
#     n_imgs = len(imgs_list)
#     data_list = []
#     for i in range(n_imgs):
#         tmp = imgs_list[i]

#         if tmp.shape[1] == 3:
#             tmp = np.transpose(np.squeeze(tmp), [1, 2, 0])
#             tmp = restore_rgb(
#                 [arg.channel_swap, arg.mean_pixel_values[:3]], tmp)
#             tmp = np.uint8(image_normalization(tmp))
#         else:
#             tmp = np.squeeze(tmp)
#             if len(tmp.shape) == 2:
#                 tmp = np.uint8(image_normalization(tmp))
#                 tmp = cv.bitwise_not(tmp)
#                 tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
#             else:
#                 tmp = np.uint8(image_normalization(tmp))
#         data_list.append(tmp)
#     img = data_list[0]
#     if n_imgs % 2 == 0:
#         imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1]
#                          * (n_imgs // 2) + ((n_imgs // 2 - 1) * 5), 3))
#     else:
#         imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1]
#                          * ((1 + n_imgs) // 2) + ((n_imgs // 2) * 5), 3))
#         n_imgs += 1

#     k = 0
#     imgs = np.uint8(imgs)
#     i_step = img.shape[0]+10
#     j_step = img.shape[1]+5
#     for i in range(2):
#         for j in range(n_imgs//2):
#             if k < len(data_list):
#                 imgs[i*i_step:i*i_step+img.shape[0], j*j_step:j *
#                      j_step+img.shape[1], :] = data_list[k]
#                 k += 1
#             else:
#                 pass
#     return imgs


# def cv_imshow(title='image', img=None):
#     cv.imshow(title, img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
def tensor2edge(tensor):
    print(tensor.shape)
    tensor =torch.squeeze(tensor) if len(tensor.shape)>2 else tensor
    tmp = torch.sigmoid(tensor)
    tmp = tmp.cpu().detach().numpy()
    # tmp = np.transpose(np.squeeze(tmp[1]), [1, 2, 0])
    tmp = np.uint8(image_normalization(tmp))
    tmp = cv.bitwise_not(tmp)
    tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
    cv.imshow('test_img', tmp)
    cv.waitKey(0)
    cv.destroyAllWindows()