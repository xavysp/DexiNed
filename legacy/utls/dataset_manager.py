"""

"""
import sys
import cv2 as cv
from PIL import Image

from legacy.utls.utls import *


def cv_imshow(title='default',img=None):
    print(img.shape)
    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rotated_img_extractor(x=None, gt=None,img_width=None, img_height=None,i=None, two_data=False):
    if two_data:
        if img_width==img_height:
            # for images whose sizes are the same

            if i % 90 == 0:
                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (100, 100), (720 - 100, 720 - 100), (0, 0, 255), (2))
                    rot_x = rot_x[100:720 - 100, 100:720 - 100, :]
                    rot_gt = rot_gt[100:720 - 100, 100:720 - 100]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    rot_x = rot_x[75:720 - 75, 75:720 - 75, :]
                    rot_gt = rot_gt[75:720 - 75, 75:720 - 75]
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    rot_x = rot_x[95:720 - 95, 95:720 - 95, :]
                    rot_gt = rot_gt[95:720 - 95, 95:720 - 95]
            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    rot_x = rot_x[85:720 - 85, 85:720 - 85, :]
                    rot_gt = rot_gt[85:720 - 85, 85:720 - 85]
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    rot_gt = rot_gt[105:720 - 105, 105:720 - 105]
                    # print("just for check 23:", i, rot_x.shape)

            return rot_x, rot_gt
        else:
            # # for images whose sizes are ***not*** the same *********************************
            img_size = img_width if img_width < img_height else img_height
            if i % 90 == 0:
                if i==180:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+250, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+250, img_width))
                    # a = np.copy(rot_x)
                    rot_x = rot_x[10:img_size-90, 10:img_size+110, :]
                    rot_gt = rot_gt[10:img_size-90, 10:img_size+110]
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 450, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height + 450, img_width))
                    # a = np.copy(rot_x)
                    rot_x = rot_x[100:img_size + 200, 300:img_size + 200, :]
                    rot_gt = rot_gt[100:img_size + 200, 300:img_size + 200]
            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+i+5, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+i+5, img_width))
                    # a = np.copy(rot_x)
                    # #                 x    y             x           y
                    # cv.rectangle(a, (275, 275), (img_size+55, img_size+55), (0, 0, 255), (2))
                    #                   y                   x
                    rot_x = rot_x[275:img_size+55, 275:img_size+55, :]
                    rot_gt = rot_gt[275:img_size+55, 275:img_size+55]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+i, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+i, img_width))
                    rot_x = rot_x[100:img_size-50,355:img_size+205, :]
                    rot_gt = rot_gt[100:img_size-50,355:img_size+205]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i==19:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+200, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+200, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (150, 150), (img_size+30, img_size-70), (0, 0, 255), (2))
                    rot_x = rot_x[150:img_size-70, 150:img_size+30, :]
                    rot_gt = rot_gt[150:img_size-70, 150:img_size+30]
                    # print("just for check 19: ", i, rot_x.shape)

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+250, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (400, 115), (img_size+180, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size-105, 400:img_size+180, :]
                    rot_gt = rot_gt[115:img_size-105, 400:img_size+180]
                    # print("just for check 19: ", i, rot_x.shape)

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+i+200, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+i, img_width))
                    # a = rot_x
                    # cv.rectangle(a, (95, 50), (img_size+75, img_size-170), (0, 0, 255), (2))
                    rot_x = rot_x[50:img_size-170, 95:img_size+75, :]
                    rot_gt = rot_gt[50:img_size-170, 95:img_size+75]
                    # print("just for check 23: ", i, rot_x.shape)
                elif i==207:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (200, 185), (img_size + 160, img_size - 95), (0, 0, 255), (2))
                    rot_x = rot_x[185:img_size - 95, 200:img_size + 160, :]
                    rot_gt = rot_gt[185:img_size - 95, 200:img_size + 160]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height+250, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height+250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (390, 115), (img_size+170, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size-105, 390:img_size+170, :]
                    rot_gt = rot_gt[115:img_size-105, 390:img_size+170]
            return rot_x,rot_gt
    else:
        # For  NIR imagel but just NIR (ONE data)
        if img_height==img_width:

            if i % 90 == 0:
                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                # print("just for check 90: ", i)
            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (100, 100), (720 - 100, 720 - 100), (0, 0, 255), (2))
                    rot_x = rot_x[100:720 - 100, 100:720 - 100, :]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (75, 75), (720 - 75, 720 - 75), (0, 0, 255), (2))
                    rot_x = rot_x[75:720 - 75, 75:720 - 75, :]
                    # print("just for check 19: ", i, rot_x.shape)
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(95,95),(720-95,720-95),(0,0,255),(2) )
                    rot_x = rot_x[95:720 - 95, 95:720 - 95, :]
                    # print("just for check 19: ", i, rot_x.shape)
            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(85,85),(720-85,720-85),(0,0,255),(2) )
                    rot_x = rot_x[85:720 - 85, 85:720 - 85, :]
                    # print("just for check 23: ", i, rot_x.shape)
                elif i==207:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    # print("just for check 23:", i, rot_x.shape)
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    # print("just for check 23:", i, rot_x.shape)
            else:
                print("Error line 221 in dataset_manager")
                sys.exit()
        else:
            img_size = img_width if img_width < img_height else img_height
            if i % 90 == 0:
                if i == 180:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    rot_x = rot_x[10:img_size - 90, 10:img_size + 110, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 450, img_width))
                    rot_x = rot_x[100:img_size + 200, 300:img_size + 200, :]

            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i + 5, img_width))
                    # #                 x    y             x           y
                    # cv.rectangle(a, (275, 275), (img_size+55, img_size+55), (0, 0, 255), (2))
                    #                   y                   x
                    rot_x = rot_x[275:img_size + 55, 275:img_size + 55, :]
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (355, 100), (img_size+205, img_size-50), (0, 0, 255), (2))
                    rot_x = rot_x[100:img_size - 50, 355:img_size + 205, :]
                elif i == 19:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 200, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (150, 150), (img_size+30, img_size-70), (0, 0, 255), (2))
                    rot_x = rot_x[150:img_size - 70, 150:img_size + 30, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (400, 115), (img_size+180, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size - 105, 400:img_size + 180, :]

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i + 200, img_width))
                    # a = rot_x
                    # cv.rectangle(a, (95, 50), (img_size+75, img_size-170), (0, 0, 255), (2))
                    rot_x = rot_x[50:img_size - 170, 95:img_size + 75, :]
                elif i == 207:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (200, 185), (img_size + 160, img_size - 95), (0, 0, 255), (2))
                    rot_x = rot_x[185:img_size - 95, 200:img_size + 160, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (390, 115), (img_size+170, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size - 105, 390:img_size + 170, :]
        return rot_x

def augment_data(args):

    data_for = 'train'  # choice [train validation, test]
    imgs_splitted = True
    imgs_rotated = True
    imgs_flipped = True
    imgs_gamma_corrected = True

    # degrees
    #          [19, 46, 57,  90,  114,   138, 161, 180,  207, 230,   247   270,  285,  322, 342]
    degrees = [19, 23*2, 19*3, 90, 19*6, 23*6,23*7,180, 23*9, 23*10,19*13, 270, 19*15,23*14,19*18]
    # if data_for=='train':
    # **************** training data ************************#
    if data_for=='train' and not args.use_nir:
        base_dataset_dir = args.dataset_dir.lower() + args.train_dataset + '/edges'
        GT_dir = os.path.join(base_dataset_dir, 'edge_maps/train/rgbr/aug')
        X_dir = os.path.join(base_dataset_dir, 'imgs/train/rgbr/aug')
        # this implementation is just for BIPED dataset
        gt_list = os.listdir(os.path.join(GT_dir,'real'))  #
        gt_list.sort()
        x_list = os.listdir(os.path.join(X_dir,'real'))
        x_list.sort()
        n = len(gt_list) if len(x_list)==len(gt_list) else 0
        if n==0:
            print('there is some inconsistence in the size of lists')
            sys.exit()

        # making 720x720 image size (splitting) ******************
        # ********************************************************
        if not imgs_splitted:
            tmp_img = cv.imread(os.path.join(
                os.path.join(X_dir, 'real'), x_list[0]))
            img_width = tmp_img.shape[1]
            img_height = tmp_img.shape[0]
            if not os.path.exists(os.path.join(GT_dir, 'p1')):
                os.makedirs(os.path.join(GT_dir, 'p1'))
            if not os.path.exists(os.path.join(GT_dir, 'p2')):
                os.makedirs(os.path.join(GT_dir, 'p2'))
            if not os.path.exists(os.path.join(X_dir, 'p1')):
                os.makedirs(os.path.join(X_dir, 'p1'))
            if not os.path.exists(os.path.join(X_dir, 'p2')):
                os.makedirs(os.path.join(X_dir, 'p2'))

            for i in range(n):
                x_tmp = cv.imread(os.path.join(
                    os.path.join(X_dir,'real'), x_list[i]))
                gt_tmp = cv.imread(os.path.join(
                    os.path.join(GT_dir,'real'),gt_list[i]))
                x_tmp1 = x_tmp[:,0:img_height,:]
                x_tmp2 = x_tmp[:,img_width-img_height:img_width,:]

                gt_tmp1 = gt_tmp[:,0:img_height]
                gt_tmp2 = gt_tmp[:,img_width-img_height:img_width]

                cv.imwrite(os.path.join(X_dir,os.path.join('p1',x_list[i])), x_tmp1)
                cv.imwrite(os.path.join(X_dir, os.path.join('p2', x_list[i])), x_tmp2)
                cv.imwrite(os.path.join(GT_dir, os.path.join('p1', gt_list[i])), gt_tmp1)
                cv.imwrite(os.path.join(GT_dir, os.path.join('p2', gt_list[i])), gt_tmp2)
                print('saved image: ', x_list[i], gt_list[i])
            print('...split done')

        # *************** image rotation *******************************
        if not imgs_rotated:

            # for p1 ***********
            folder_name='real'  # choice [p1,p2,real] which are the source of files previously prepared
            # folder_name_x = 'real'
            if folder_name=='p1':
                x_aug_list = os.listdir(os.path.join(X_dir, 'p1'))
                x_aug_list.sort()
                gt_aug_list = os.listdir(os.path.join(GT_dir, 'p1'))
                gt_aug_list.sort()
            elif folder_name == 'p2':
                x_aug_list = os.listdir(os.path.join(X_dir, 'p2'))
                x_aug_list.sort()
                gt_aug_list = os.listdir(os.path.join(GT_dir, 'p2'))
                gt_aug_list.sort()
            elif folder_name=='real':
                x_aug_list = os.listdir(os.path.join(X_dir, 'real'))
                x_aug_list.sort()
                gt_aug_list = os.listdir(os.path.join(GT_dir, 'real'))
                gt_aug_list.sort()
            else:
                print("error reading folder name")
                sys.exit()
            if n==len(x_aug_list) and n== len(gt_aug_list):
                pass
            else:
                print("Error reading data. The is an inconsistency in the data ")
                sys.exit()

            tmp_img = cv.imread(os.path.join(X_dir,
                                             os.path.join(folder_name,x_aug_list[1])))
            img_width = tmp_img.shape[1]
            img_height = tmp_img.shape[0]

            for i in (degrees):
                if folder_name=='p1':
                    current_X_dir =  X_dir+'/p1_rot_'+str(i)
                    current_GT_dir = GT_dir+'/p1_rot_'+str(i)
                elif folder_name=='p2':

                    current_X_dir = X_dir + '/p2_rot_' + str(i)
                    current_GT_dir = GT_dir + '/p2_rot_' + str(i)
                elif folder_name=='real':
                    current_X_dir = X_dir + '/real_rot_' + str(i)
                    current_GT_dir = GT_dir + '/real_rot_' + str(i)
                else:
                    print('error')
                    sys.exit()
                if not os.path.exists(current_X_dir):
                    os.makedirs(current_X_dir)
                if not os.path.exists(current_GT_dir):
                    os.makedirs(current_GT_dir)

                for j in range(n):
                    # i = degrees[j]
                    if folder_name=='real':
                        tmp_x = cv.imread(os.path.join(X_dir,
                                                       os.path.join(folder_name, x_aug_list[j])))
                        tmp_gt = cv.imread(os.path.join(GT_dir,
                                                        os.path.join(folder_name, gt_aug_list[j])))
                        rot_x,rot_gt=rotated_img_extractor(tmp_x, tmp_gt,img_width, img_height,i, True)
                        # [19, 46, 90, 114, 138, 161, 180, 207, 230, 247, 270, 285, 322, 342]

                    else:
                        tmp_x = cv.imread(os.path.join(X_dir,
                                               os.path.join(folder_name,x_aug_list[j])))
                        tmp_gt = cv.imread(os.path.join(GT_dir,
                                                        os.path.join(folder_name,gt_aug_list[j])))
                        rot_x, rot_gt = rotated_img_extractor(tmp_x, tmp_gt, img_width, img_height, i, True)

                    cv.imwrite(os.path.join(current_GT_dir,gt_aug_list[j]),rot_gt)
                    cv.imwrite(os.path.join(current_X_dir, x_aug_list[j]),rot_x)
                    tmp_imgs = np.concatenate((rot_x,rot_gt), axis=1)
                    cv.imshow("rotated", tmp_imgs)
                    cv.waitKey(400) # 1000= 1 seg
                print("rotation with {} degrees fullfiled ".format(i))
            cv.destroyAllWindows()
            print("... rotation done in ", folder_name)

        # **************** flipping horizontally ***********
        if not imgs_flipped:
            type_aug= '_flip'
            dir_list = os.listdir(X_dir)
            dir_list.sort()

            for i in (dir_list):
                X_list = os.listdir(os.path.join(X_dir,i))
                X_list.sort()
                GT_list = os.listdir(os.path.join(GT_dir,i))
                GT_list.sort()
                save_dir_x = X_dir+'/'+str(i)+type_aug
                save_dir_gt = GT_dir+'/'+str(i)+type_aug

                if not os.path.exists(save_dir_x):
                    os.makedirs(save_dir_x)
                if not os.path.exists(save_dir_gt):
                    os.makedirs(save_dir_gt)

                print("Working on the dir: ", os.path.join(X_dir,i),os.path.join(GT_dir,i) )
                for j in range(n):
                    x_tmp = cv.imread(os.path.join(X_dir,os.path.join(i,X_list[j])))
                    gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
                    flip_x = np.fliplr(x_tmp)
                    flip_gt = np.fliplr(gt_tmp)

                    tmp_imgs = np.concatenate((flip_x, flip_gt), axis=1)
                    cv.imshow("rotated", tmp_imgs)
                    cv.waitKey(350)
                    cv.imwrite(os.path.join(save_dir_gt,GT_list[j]),flip_gt)
                    cv.imwrite(os.path.join(save_dir_x, X_list[j]),flip_x)

                print("End flipping file in {}".format(os.path.join(X_dir,i)))
            cv.destroyAllWindows()
            print("... Flipping  data augmentation finished")

        # ***********Data augmentation based on gamma correction **********
        if not imgs_gamma_corrected:
            gamma30 = '_ga30'
            gamma60 = '_ga60'
            dir_list = os.listdir(X_dir)
            dir_list.sort()
            for i in (dir_list):
                X_list = os.listdir(os.path.join(X_dir,i))
                X_list.sort()
                GT_list = os.listdir(os.path.join(GT_dir,i))
                GT_list.sort()
                save_dir_x30 = X_dir+'/'+str(i)+gamma30
                save_dir_gt30 = GT_dir+'/'+str(i)+gamma30
                save_dir_x60 = X_dir + '/' + str(i)+ gamma60
                save_dir_gt60 = GT_dir + '/'  + str(i)+ gamma60

                if not os.path.exists(save_dir_x30):
                    os.makedirs(save_dir_x30)
                if not os.path.exists(save_dir_gt30):
                    os.makedirs(save_dir_gt30)
                if not os.path.exists(save_dir_x60):
                    os.makedirs(save_dir_x60)
                if not os.path.exists(save_dir_gt60):
                    os.makedirs(save_dir_gt60)
                print("Working on the dir: ", os.path.join(X_dir, i), os.path.join(GT_dir, i))
                for j in range(n):
                    x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
                    gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
                    x_tmp = normalization_data_01(x_tmp)
                    x_tmp = gamma_correction(x_tmp,0.4040,False)

                    gam30_x = gamma_correction(x_tmp,0.3030,True)
                    gam60_x = gamma_correction(x_tmp,0.6060, True)
                    gam30_x = np.uint8(normalization_data_0255(gam30_x))
                    gam60_x = np.uint8(normalization_data_0255(gam60_x))

                    tmp_imgs1 = np.concatenate((gam30_x, gt_tmp), axis=1)
                    tmp_imgs2 = np.concatenate((gam60_x, gt_tmp), axis=1)
                    tmp_imgs = np.concatenate((tmp_imgs2,tmp_imgs1),axis=0)

                    cv.imshow("gamma ", tmp_imgs)
                    cv.waitKey(350)
                    cv.imwrite(os.path.join(save_dir_gt30, GT_list[j]), gt_tmp)
                    cv.imwrite(os.path.join(save_dir_x30, X_list[j]), gam30_x)
                    cv.imwrite(os.path.join(save_dir_gt60, GT_list[j]), gt_tmp)
                    cv.imwrite(os.path.join(save_dir_x60, X_list[j]), gam60_x)

                print("End gamma correction, file in {}".format(os.path.join(X_dir, i)))
            cv.destroyAllWindows()
            print("... gamma correction  data augmentation finished")

    # ************** for validation ********************************
    elif data_for=='validation' and not args.use_nir:

        train_GT_dir = args.dataset_dir + args.train_dataset + '/valid'
        train_X_dir = args.dataset_dir + args.train_dataset + '/valid'

        gt_list = os.listdir(os.path.join(train_GT_dir, 'GT_un'))  #
        gt_list.sort()
        x_list = os.listdir(os.path.join(train_X_dir, 'X_un'))
        x_list.sort()
        n = len(gt_list) if len(x_list) == len(gt_list) else 0
        if n == 0:
            print('there is some inconsistence in the size of lists')
            sys.exit()

        # making 720x720 image size (splitting) ******************
        if not imgs_splitted:
            tmp_img = cv.imread(os.path.join(
                os.path.join(train_X_dir, 'X_un'), x_list[0]))
            img_width = tmp_img.shape[1]
            img_height = tmp_img.shape[0]
            for i in range(n):
                x_tmp = cv.imread(os.path.join(
                    os.path.join(train_X_dir, 'X_un'), x_list[i]))
                gt_tmp = cv.imread(os.path.join(
                    os.path.join(train_GT_dir, 'GT_un'), gt_list[i]))
                x_tmp1 = x_tmp[:, 0:img_height, :]
                x_tmp2 = x_tmp[:, img_width - img_height:img_width, :]

                gt_tmp1 = gt_tmp[:, 0:img_height]
                gt_tmp2 = gt_tmp[:, img_width - img_height:img_width]

                cv.imwrite(os.path.join(train_X_dir, os.path.join('X/p1', x_list[i])), x_tmp1)
                cv.imwrite(os.path.join(train_X_dir, os.path.join('X/p2', x_list[i])), x_tmp2)
                cv.imwrite(os.path.join(train_GT_dir, os.path.join('GT/p1', gt_list[i])), gt_tmp1)
                cv.imwrite(os.path.join(train_GT_dir, os.path.join('GT/p2', gt_list[i])), gt_tmp2)
                print('saved image: ', x_list[i], gt_list[i])

                tmp_imgs = np.concatenate((x_tmp,gt_tmp), axis=1)
                cv.imshow("rotated", tmp_imgs)
                cv.waitKey(500)
            cv.destroyAllWindows()
            print('...split for validation finished')

        # image rotation
        if not imgs_rotated:
            folder_name = 'p2'  # choice [p1,p2,GT_u] which are the source of files previously prepared
            X_dir = args.dataset_dir + args.train_dataset + '/valid'
            GT_dir = args.dataset_dir + args.train_dataset + '/valid'

            folder_name_x = 'X_un'
            if folder_name == 'p1':
                x_aug_list = os.listdir(os.path.join(X_dir, 'X/p1'))
                x_aug_list.sort()
                gt_aug_list = os.listdir(os.path.join(GT_dir, 'GT/p1'))
                gt_aug_list.sort()
            elif folder_name == 'p2':
                x_aug_list = os.listdir(os.path.join(X_dir, 'X/p2'))
                x_aug_list.sort()
                gt_aug_list = os.listdir(os.path.join(GT_dir, 'GT/p2'))
                gt_aug_list.sort()
            elif folder_name == 'GT_u':
                x_aug_list = os.listdir(os.path.join(X_dir, 'X_un'))
                x_aug_list.sort()
                gt_aug_list = os.listdir(os.path.join(GT_dir, 'GT_un'))
                gt_aug_list.sort()
            else:
                print("error reading folder name")
                sys.exit()
            # image size
            tmp_img = cv.imread(os.path.join(X_dir,
                                             os.path.join('X/' + folder_name, x_aug_list[1])))
            img_width = tmp_img.shape[1]
            img_height = tmp_img.shape[0]

            for i in (degrees):
                if folder_name == 'p1':
                    current_X_dir = X_dir + '/X/p1_rot_' + str(i)
                    current_GT_dir = GT_dir + '/GT/p1_rot_' + str(i)
                elif folder_name == 'p2':

                    current_X_dir = X_dir + '/X/p2_rot_' + str(i)
                    current_GT_dir = GT_dir + '/GT/p2_rot_' + str(i)
                elif folder_name == 'GT_u':
                    current_X_dir = X_dir + 'X/un_rot_' + str(i)
                    current_GT_dir = GT_dir + 'GT/un_rot_' + str(i)
                else:
                    print('error')
                    sys.exit()
                if not os.path.exists(current_X_dir):
                    os.makedirs(current_X_dir)
                if not os.path.exists(current_GT_dir):
                    os.makedirs(current_GT_dir)

                for j in range(n):
                    if folder_name == 'GT_un':
                        tmp_x = cv.imread(os.path.join(X_dir,
                                                       os.path.join(folder_name_x, x_aug_list[j])))
                        tmp_gt = cv.imread(os.path.join(GT_dir,
                                                        os.path.join(folder_name, gt_aug_list[j])))
                    else:
                        tmp_x = cv.imread(os.path.join(X_dir,
                                                       os.path.join('X/' + folder_name, x_aug_list[j])))
                        tmp_gt = cv.imread(os.path.join(GT_dir,
                                                        os.path.join('GT/' + folder_name, gt_aug_list[j])))
                    if i % 90 == 0:
                        adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                        rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                        rot_gt = cv.warpAffine(tmp_gt, adjus_M, (img_height, img_width))
                    elif i % 19 == 0:
                        if i == 57:
                            adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                            rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                            rot_gt = cv.warpAffine(tmp_gt, adjus_M, (img_height, img_width))
                            # cv.rectangle(rot_x, (100, 100), (720 - 100, 720 - 100), (0, 0, 255), (2))
                            rot_x = rot_x[100:720 - 100, 100:720 - 100, :]
                            rot_gt = rot_gt[100:720 - 100, 100:720 - 100]
                            # print("just for check 19: ", i, rot_x.shape)
                        elif i == 285:
                            adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                            rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                            rot_gt = cv.warpAffine(tmp_gt, adjus_M, (img_height, img_width))
                            # cv.rectangle(rot_x, (75, 75), (720 - 75, 720 - 75), (0, 0, 255), (2))
                            rot_x = rot_x[75:720 - 75, 75:720 - 75, :]
                            rot_gt = rot_gt[75:720 - 75, 75:720 - 75]
                        else:
                            adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                            rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                            rot_gt = cv.warpAffine(tmp_gt, adjus_M, (img_height, img_width))
                            rot_x = rot_x[95:720 - 95, 95:720 - 95, :]
                            rot_gt = rot_gt[95:720 - 95, 95:720 - 95]
                    elif i % 23 == 0:
                        if i == 161:
                            adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                            rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                            rot_gt = cv.warpAffine(tmp_gt, adjus_M, (img_height, img_width))
                            # cv.rectangle(rot_x,(85,85),(720-85,720-85),(0,0,255),(2) )
                            rot_x = rot_x[85:720 - 85, 85:720 - 85, :]
                            rot_gt = rot_gt[85:720 - 85, 85:720 - 85]
                            # print("just for check 23: ", i, rot_x.shape)
                        else:
                            adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                            rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                            rot_gt = cv.warpAffine(tmp_gt, adjus_M, (img_height, img_width))
                            rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                            rot_gt = rot_gt[105:720 - 105, 105:720 - 105]
                    else:
                        print('Error using degrees for rotation')
                        sys.exit()
                    cv.imwrite(os.path.join(current_GT_dir, gt_aug_list[j]), rot_gt)
                    cv.imwrite(os.path.join(current_X_dir, x_aug_list[j]), rot_x)
                    tmp_imgs = np.concatenate((rot_x, rot_gt), axis=1)
                    cv.imshow("rotated", tmp_imgs)
                    cv.waitKey(1000)
                print("rotation with {} degrees fullfiled ".format(i))
            cv.destroyAllWindows()

            print("... data rotation for validation finished ", folder_name)
        # flipping horizontally
        if not imgs_flipped:
            type_aug = 'flip_'
            X_dir = args.dataset_dir + args.train_dataset + '/valid/X'
            GT_dir = args.dataset_dir + args.train_dataset + '/valid/GT'
            dir_list = os.listdir(X_dir)
            dir_list.sort()

            for i in (dir_list):
                X_list = os.listdir(os.path.join(X_dir, i))
                X_list.sort()
                GT_list = os.listdir(os.path.join(GT_dir, i))
                GT_list.sort()
                save_dir_x = X_dir + '/' + type_aug + str(i)
                save_dir_gt = GT_dir + '/' + type_aug + str(i)

                if not os.path.exists(save_dir_x):
                    os.makedirs(save_dir_x)
                if not os.path.exists(save_dir_gt):
                    os.makedirs(save_dir_gt)

                print("Working on the dir: ", os.path.join(X_dir, i), os.path.join(GT_dir, i))
                for j in range(n):
                    x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
                    gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
                    flip_x = np.fliplr(x_tmp)
                    flip_gt = np.fliplr(gt_tmp)

                    tmp_imgs = np.concatenate((flip_x, flip_gt), axis=1)
                    cv.imshow("rotated", tmp_imgs)
                    cv.waitKey(1000)
                    cv.imwrite(os.path.join(save_dir_gt, GT_list[j]), flip_gt)
                    cv.imwrite(os.path.join(save_dir_x, X_list[j]), flip_x)

                print("End flipping file in {}".format(os.path.join(X_dir, i)))

            cv.destroyAllWindows()

            print("... Flipping   validation stage data augmentation finished")

    # ========================================================
    # =====================Just for NIR ======================
    # ========================================================
    elif args.use_nir:

        if data_for == 'train':
            base_dataset_dir = args.dataset_dir + args.train_dataset + '/edges'
            X_dir = os.path.join(base_dataset_dir, 'imgs/train/nir/aug')

            x_list = os.listdir(os.path.join(X_dir, 'real'))
            x_list.sort()
            n = len(x_list)
            if n == 0:
                print('there is some inconsistence in the size of lists')
                sys.exit()

            # making 720x720 image size (splitting) ******************
            if not imgs_splitted:
                tmp_img = cv.imread(os.path.join(
                    os.path.join(X_dir, 'real'), x_list[0]))
                img_width = tmp_img.shape[1]
                img_height = tmp_img.shape[0]

                for i in range(n):
                    x_tmp = cv.imread(os.path.join(
                        os.path.join(X_dir, 'real'), x_list[i]))
                    x_tmp1 = x_tmp[:, 0:img_height, :]
                    x_tmp2 = x_tmp[:, img_width - img_height:img_width, :]

                    cv.imwrite(os.path.join(X_dir, os.path.join('p1', x_list[i])), x_tmp1)
                    cv.imwrite(os.path.join(X_dir, os.path.join('p2', x_list[i])), x_tmp2)
                    print('saved image: ', x_list[i])
                print('...split done')

            #  ***************** image rotation ******************
            if not imgs_rotated:

                # for p1 ***********
                folder_name = 'real'  # choice [p1,p2,real] which are the source of files previously prepared
                # firstly X_un is not used
                folder_name_x = 'real'
                if folder_name == 'p1':
                    x_aug_list = os.listdir(os.path.join(X_dir, 'p1'))
                    x_aug_list.sort()
                elif folder_name == 'p2':
                    x_aug_list = os.listdir(os.path.join(X_dir, 'p2'))
                    x_aug_list.sort()
                elif folder_name == 'real':
                    x_aug_list = os.listdir(os.path.join(X_dir, 'real'))
                    x_aug_list.sort()
                else:
                    print("error reading folder name")
                    sys.exit()

                # image size
                tmp_img = cv.imread(os.path.join(
                    X_dir, os.path.join(folder_name, x_aug_list[1])))
                img_width = tmp_img.shape[1]
                img_height = tmp_img.shape[0]

                for i in (degrees):
                    if folder_name == 'p1':
                        current_X_dir = X_dir + '/p1_rot_' + str(i)
                    elif folder_name == 'p2':
                        current_X_dir = X_dir + '/p2_rot_' + str(i)
                    elif folder_name == 'real':
                        current_X_dir = X_dir + '/real_rot_' + str(i)
                    else:
                        print('error')
                        sys.exit()
                    if not os.path.exists(current_X_dir):
                        os.makedirs(current_X_dir)

                    for j in range(n):
                        # i = degrees[j]
                        if folder_name == 'real':
                            tmp_x = cv.imread(os.path.join(X_dir,
                                                           os.path.join(folder_name_x, x_aug_list[j])))
                            rot_x = rotated_img_extractor(x=tmp_x, gt=None, img_width=img_width,
                                                          img_height=img_height, i=i, two_data=False)
                        else:
                            tmp_x = cv.imread(os.path.join(X_dir,
                                                           os.path.join(folder_name, x_aug_list[j])))
                            rot_x = rotated_img_extractor(x=tmp_x, gt=None, img_width=img_width,
                                                          img_height=img_height, i=i, two_data=False)

                        cv.imwrite(os.path.join(current_X_dir, x_aug_list[j]), rot_x)
                        cv.imshow("rotated", rot_x)
                        cv.waitKey(450)
                    print("rotation with {} degrees fullfiled ".format(i))
                cv.destroyAllWindows()

                print("... rotation done in ", folder_name)

            # flipping horizontally
            if not imgs_flipped:
                type_aug = '_flip'
                dir_list = os.listdir(X_dir)
                dir_list.sort()

                for i in (dir_list):
                    X_list = os.listdir(os.path.join(X_dir, i))
                    X_list.sort()
                    save_dir_x = X_dir + '/' + str(i)+ type_aug

                    if not os.path.exists(save_dir_x):
                        os.makedirs(save_dir_x)

                    print("Working on the dir: ", os.path.join(X_dir, i), i)
                    for j in range(n):
                        x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
                        flip_x = np.fliplr(x_tmp)

                        cv.imshow("Flipping", x_tmp)
                        cv.waitKey(450)
                        cv.imwrite(os.path.join(save_dir_x, X_list[j]), flip_x)

                    print("End flipping file in {}".format(os.path.join(X_dir, i)))

                cv.destroyAllWindows()

                print("... Flipping  data augmentation finished")

            if not imgs_gamma_corrected:
                gamma30 = '_ga30'
                gamma60 = '_ga60'
                dir_list = os.listdir(X_dir)
                dir_list.sort()
                for i in (dir_list):
                    X_list = os.listdir(os.path.join(X_dir, i))
                    X_list.sort()
                    save_dir_x30 = X_dir + '/' + str(i) + gamma30
                    save_dir_x60 = X_dir + '/' + str(i) + gamma60

                    if not os.path.exists(save_dir_x30):
                        os.makedirs(save_dir_x30)
                    if not os.path.exists(save_dir_x60):
                        os.makedirs(save_dir_x60)

                    print("Working on the dir: ", os.path.join(X_dir, i))
                    for j in range(n):
                        x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
                        x_tmp = normalization_data_01(x_tmp)
                        x_tmp = gamma_correction(x_tmp, 0.4040, False)

                        gam30_x = gamma_correction(x_tmp, 0.3030, True)
                        gam60_x = gamma_correction(x_tmp, 0.6060, True)
                        gam30_x = np.uint8(normalization_data_0255(gam30_x))
                        gam60_x = np.uint8(normalization_data_0255(gam60_x))

                        tmp_imgs = np.concatenate((gam30_x, gam60_x), axis=1)

                        cv.imshow("gamma ", tmp_imgs)
                        cv.waitKey(450)
                        cv.imwrite(os.path.join(save_dir_x30, X_list[j]), gam30_x)
                        cv.imwrite(os.path.join(save_dir_x60, X_list[j]), gam60_x)

                    print("End gamma correction, file in {}".format(os.path.join(X_dir, i)))
                cv.destroyAllWindows()
                print("... gamma correction  data augmentation finished")


        elif data_for == 'validation':

            train_X_dir = args.dataset_dir + args.train_dataset + '/nir_valid'

            x_list = os.listdir(os.path.join(train_X_dir, 'X_un'))
            x_list.sort()
            n = len(x_list)
            if n == 0:
                print('there is some inconsistence in the size of lists')
                sys.exit()
                # making 720x720 image size (splitting) ******************
            if not imgs_splitted:
                tmp_img = cv.imread(os.path.join(
                    os.path.join(train_X_dir, 'X_un'), x_list[0]))
                img_width = tmp_img.shape[1]
                img_height = tmp_img.shape[0]

                for i in range(n):
                    x_tmp = cv.imread(os.path.join(
                        os.path.join(train_X_dir, 'X_un'), x_list[i]))
                    x_tmp1 = x_tmp[:, 0:img_height, :]
                    x_tmp2 = x_tmp[:, img_width - img_height:img_width, :]

                    cv.imwrite(os.path.join(train_X_dir, os.path.join('X/p1', x_list[i])), x_tmp1)
                    cv.imwrite(os.path.join(train_X_dir, os.path.join('X/p2', x_list[i])), x_tmp2)
                    print('saved image: ', x_list[i])
                print('... validation split done')

            # image rotation
            if not imgs_rotated:
                X_dir = args.dataset_dir + args.train_dataset + '/nir_valid'

                # for p1 ***********
                folder_name = 'p2'  # choice [p1,p2,X_un] which are the source of files previously prepared
                # firstly X_un is not used
                folder_name_x = 'X_un'
                if folder_name == 'p1':
                    x_aug_list = os.listdir(os.path.join(X_dir, 'X/p1'))
                    x_aug_list.sort()
                elif folder_name == 'p2':
                    x_aug_list = os.listdir(os.path.join(X_dir, 'X/p2'))
                    x_aug_list.sort()
                elif folder_name == 'X_un':
                    x_aug_list = os.listdir(os.path.join(X_dir, 'X_un'))
                    x_aug_list.sort()
                else:
                    print("error reading folder name")
                    sys.exit()

                # image size
                tmp_img = cv.imread(os.path.join(X_dir,
                                                 os.path.join('X/' + folder_name, x_aug_list[1])))
                img_width = tmp_img.shape[1]
                img_height = tmp_img.shape[0]

                for i in (degrees):
                    if folder_name == 'p1':
                        current_X_dir = X_dir + '/X/p1_rot_' + str(i)
                    elif folder_name == 'p2':
                        current_X_dir = X_dir + '/X/p2_rot_' + str(i)
                    elif folder_name == 'X_un':
                        current_X_dir = X_dir + 'X/un_rot_' + str(i)
                    else:
                        print('error')
                        sys.exit()
                    if not os.path.exists(current_X_dir):
                        os.makedirs(current_X_dir)

                    for j in range(n):
                        # i = degrees[j]
                        if folder_name == 'X_un':
                            tmp_x = cv.imread(
                                os.path.join(X_dir, os.path.join(folder_name_x, x_aug_list[j])))
                        else:
                            tmp_x = cv.imread(
                                os.path.join(X_dir, os.path.join('X/' + folder_name, x_aug_list[j])))

                        if i % 90 == 0:
                            adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                            rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))

                        elif i % 19 == 0:
                            if i == 57:
                                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                                rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                                # cv.rectangle(rot_x, (100, 100), (720 - 100, 720 - 100), (0, 0, 255), (2))
                                rot_x = rot_x[100:720 - 100, 100:720 - 100, :]
                                # print("just for check 19: ", i, rot_x.shape)
                            elif i == 285:
                                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                                rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                                # cv.rectangle(rot_x, (75, 75), (720 - 75, 720 - 75), (0, 0, 255), (2))
                                rot_x = rot_x[75:720 - 75, 75:720 - 75, :]
                                # print("just for check 19: ", i, rot_x.shape)
                            else:
                                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                                rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                                # cv.rectangle(rot_x,(95,95),(720-95,720-95),(0,0,255),(2) )
                                rot_x = rot_x[95:720 - 95, 95:720 - 95, :]
                                # print("just for check 19: ", i, rot_x.shape)

                        elif i % 23 == 0:
                            if i == 161:
                                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                                rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                                # cv.rectangle(rot_x,(85,85),(720-85,720-85),(0,0,255),(2) )
                                rot_x = rot_x[85:720 - 85, 85:720 - 85, :]
                                # print("just for check 23: ", i, rot_x.shape)
                            else:
                                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                                rot_x = cv.warpAffine(tmp_x, adjus_M, (img_height, img_width))
                                # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                                rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                                # print("just for check 23:", i, rot_x.shape)
                        else:
                            print('Error using degrees for rotation')
                            sys.exit()
                        cv.imwrite(os.path.join(current_X_dir, x_aug_list[j]), rot_x)
                        cv.imshow("rotated", rot_x)
                        cv.waitKey(350)
                    print("rotation with {} degrees fullfiled ".format(i))
                cv.destroyAllWindows()

                print("... rotation done in ", folder_name)

            # flipping horizontally
            if not imgs_flipped:
                type_aug = 'flip_'
                X_dir = args.dataset_dir + args.train_dataset + '/nir_valid/X'
                dir_list = os.listdir(X_dir)
                dir_list.sort()

                for i in (dir_list):
                    X_list = os.listdir(os.path.join(X_dir, i))
                    X_list.sort()
                    save_dir_x = X_dir + '/' + type_aug + str(i)

                    if not os.path.exists(save_dir_x):
                        os.makedirs(save_dir_x)

                    print("Working on the dir: ", os.path.join(X_dir, i), i)
                    for j in range(n):
                        x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
                        flip_x = np.fliplr(x_tmp)

                        cv.imshow("rotated", x_tmp)
                        cv.waitKey(350)
                        cv.imwrite(os.path.join(save_dir_x, X_list[j]), flip_x)

                    print("End flipping file in {}".format(os.path.join(X_dir, i)))

                cv.destroyAllWindows()

                print("... Flipping  validation data augmentation finished")
        else:
            print("This part is not finished yet")
    else:

        print("Error, just train and validation code have written")

def data_parser(args):

    if args.model_state=='train':

        train_files_name = args.train_list # dataset base dir

        base_dir = os.path.join(args.dataset_dir, args.train_dataset,'edges') \
            if args.train_dataset.lower() == 'biped' else os.path.join(args.dataset_dir, args.train_dataset)

        train_list_path = os.path.join(base_dir, train_files_name)

        train_list = read_files_list(train_list_path)

        train_samples = split_pair_names(args,train_list, base_dir=base_dir)
        n_train = len(train_samples)
        print_info(" Enterely training set-up from {}, size: {}".format(train_list_path, n_train))

        all_train_ids = np.arange(n_train)
        np.random.shuffle(all_train_ids)

        train_ids = all_train_ids[:int(args.train_split * len(train_list))]
        valid_ids = all_train_ids[int(args.train_split * len(train_list)):]

        print_info("Training set-up from {}, size: {}".format(train_list_path, len(train_ids)))
        print_info("Validation set-up from {}, size: {}".format(train_list_path, len(valid_ids)))
        cache_info = {
            "files_path": train_samples,
            "n_files": n_train,
            "train_indices": train_ids,
            "validation_indices": valid_ids
        }
        return cache_info

# ************** for testing **********************
    elif args.model_state=='test':
        base_dir = os.path.join(args.dataset_dir,args.test_dataset,'edges')\
            if args.test_dataset.upper()=='BIPED' else os.path.join(args.dataset_dir,args.test_dataset)

        if args.test_dataset.upper() == "BIPED":
            test_files_name = args.test_list
            test_list_path = os.path.join(base_dir,test_files_name)
            test_list = read_files_list(test_list_path)

            test_samples = split_pair_names(args, test_list, base_dir)
            n_test = len(test_samples)
            print_info(" Enterely testing set-up from {}, size: {}".format(test_list_path, n_test))

            test_ids = np.arange(n_test)
            # np.random.shuffle(test_ids)

            print_info("testing set-up from {}, size: {}".format(test_list_path, len(test_ids)))
            cache_out = [test_samples, test_ids]

            return cache_out

        elif args.test_dataset == "BSDS":
            test_files_name = args.test_list
            test_list_path = os.path.join(args.dataset_dir + args.test_dataset,
                                           test_files_name)
            test_list = read_files_list(test_list_path)

            test_samples = split_pair_names(args,test_list, args.dataset_dir + args.test_dataset)
            n_test = len(test_samples)
            print_info(" Enterely testing set-up from {}, size: {}".format(test_list_path, n_test))

            test_ids = np.arange(n_test)
            # np.random.shuffle(test_ids)

            print_info("testing set-up from {}, size: {}".format(test_list_path, len(test_ids)))
            cache_out = [test_samples, test_ids]
            return cache_out

        else:
            # for NYUD
            test_files_name = args.test_list
            test_list_path = os.path.join(args.dataset_dir + args.test_dataset,
                                          test_files_name)
            test_list = read_files_list(test_list_path)

            test_samples = split_pair_names(args,test_list, args.dataset_dir + args.test_dataset)
            n_test = len(test_samples)
            print_info(" Enterely testing set-up from {}, size: {}".format(test_list_path, n_test))

            test_ids = np.arange(n_test)
            # np.random.shuffle(test_ids)

            print_info("testing set-up from {}, size: {}".format(test_list_path, len(test_ids)))
            cache_out = [test_samples, test_ids]
            return cache_out
    else:
        print_error("The model state is just train and test")
        sys.exit()

# ___________batch management ___________
def get_batch(arg,file_list, batch=None, use_batch=True):

    if use_batch:
        file_names =[]
        images=[]
        edgemaps=[]
        for idx, b in enumerate(batch):
            x = cv.imread(file_list[b][0]) #  Image.open(file_list[b][0])
            y = cv.imread(file_list[b][1]) #  Image.open(file_list[b][1])
            if arg.model_state=='test':
                pass
            else:
                x = cv.resize(x, dsize=(arg.image_width,arg.image_height)) # x.resize((arg.image_width, arg.image_height))
                y = cv.resize(y,dsize=(arg.image_width, arg.image_height))# y.resize((arg.image_width, arg.image_height))
            # pay attention here
            x = np.array(x, dtype=np.float32)
            # x = x[:, :, arg.channel_swap] # while using opencv it is not necessary
            x -= arg.mean_pixel_values[0:3]

            y = cv.cvtColor(y, cv.COLOR_BGR2GRAY)# np.array(y.convert('L'), dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            if arg.train_dataset.lower()!='biped':
                y[y < 107] = 0  # first nothing second <50 third <30
                y[y >= 107] = 255.0
            # else:
                # y[y < 51] = 0  # first 100

            if arg.target_regression:
                bin_y = y/255.0
            else:
                bin_y = np.zeros_like(y)
                bin_y[np.where(y)]=1

            bin_y = bin_y if bin_y.ndim ==2 else bin_y[:,:,0]
            bin_y = np.expand_dims(bin_y,axis=2)

            images.append(x)
            edgemaps.append(bin_y)
            file_names.append(file_list[b])

        return images, edgemaps, file_names

    else:
        # for testing re-coding is needed
        if arg.test_dataset=='BIPED' and (arg.model_state=='test' and arg.use_nir):
            x_nir = Image.open(file_list[0])
            x_rgb = Image.open(file_list[1])
            real_size = x_rgb.shape
            # y = Image.open(file_list[2])
            if not arg.image_width % 6 == 0 and arg.image_width > 1000:
                x_nir = x_nir.resize((arg.image_width, arg.image_height)) #--
                x_rgb = x_rgb.resize((arg.image_width, arg.image_height)) # --
            else:
                x_nir = x_nir.resize((arg.image_width, arg.image_height))
                x_rgb = x_rgb.resize((arg.image_width, arg.image_height))

            # y = y.resize((arg.image_width, arg.image_height))

            x_nir = x_nir.convert("L")
            # pay attention here
            x_nir = np.array(x_nir, dtype=np.float32)
            x_nir = np.expand_dims(x_nir, axis=2)
            x_rgb = np.array(x_rgb, dtype=np.float32)

            x_rgb = x_rgb[:, :, arg.channel_swap]
            x = np.concatenate((x_rgb, x_nir), axis=2)
            x -= arg.mean_pixel_values
            # y = np.array(y.convert('L'), dtype=np.float32)
            # if arg.target_regression:
            #     bin_y = y / 255.0
            # else:
            #     bin_y = np.zeros_like(y)
            #     bin_y[np.where(y)] = 1
            #
            # bin_y = bin_y if bin_y.ndim == 2 else bin_y[:, :, 0]
            # bin_y = np.expand_dims(bin_y, axis=2)
            images =x
            edgemaps  = None
            file_info = (file_list[2], real_size)

        else:
            x = cv.imread(file_list[0])
            y = cv.imread(file_list[1])
            real_size = x.shape

            if arg.test_dataset.lower()=='biped' or arg.test_dataset.lower()=='multicue':
                pass
            else:
                x = cv.resize(x, dsize=(arg.image_width, arg.image_height))
                y = cv.resize(y, dsize=(arg.image_width, arg.image_height))
            x = np.array(x, dtype=np.float32)
            # x = x[:, :, arg.channel_swap] # while using opencv it is not necessary
            x -= arg.mean_pixel_values[:-1]
            y = cv.cvtColor(y, cv.COLOR_BGR2GRAY)
            y = np.array(y, dtype=np.float32)
            if arg.target_regression:
                bin_y = y / 255.0
            else:
                bin_y = np.zeros_like(y)
                bin_y[np.where(y)] = 1

            bin_y = bin_y if bin_y.ndim == 2 else bin_y[:, :, 0]
            bin_y = np.expand_dims(bin_y, axis=2)

            images = x
            edgemaps = bin_y
            file_info = (file_list[1],real_size)
        return images, edgemaps, file_info

def get_training_batch(arg, data_ids):
    train_ids = data_ids['train_indices']
    file_list= data_ids['files_path']
    batch_ids = np.random.choice(train_ids,arg.batch_size_train)

    return get_batch(arg, file_list, batch_ids)

def get_validation_batch(arg, data_ids):
    if arg.use_nir:
        valid_ids = data_ids['validation_indices']
        file_list = data_ids['files_path']
        batch_ids = np.random.choice(valid_ids, arg.batch_size_val)
    else:
        valid_ids = data_ids['validation_indices']
        file_list= data_ids['files_path']
        batch_ids = np.random.choice(valid_ids,arg.batch_size_val)
    return get_batch(arg,file_list,batch_ids)

def get_testing_batch(arg,list_ids, use_batch=True, i=None):
    if use_batch:
        test_ids = list_ids[1]
        file_list = list_ids[0]
        batch_ids = test_ids[i:i + arg.batch_size_test]
        return get_batch(arg,file_list,batch_ids)
    else:
        return get_batch(arg, list_ids[0],list_ids[1], use_batch=False)

def open_images(file_list):
    if len(file_list)>2 and not len(file_list)==3:
        imgs=[]
        file_names = []
        for i in range(len(file_list)):
            tmp = Image.open(file_list[i])
            imgs.append(tmp)
            file_names.append(file_list[i])

    elif len(file_list)>2 and len(file_list)==3:

        imgs = Image.open(file_list[2])
        file_names = file_list
    else:
        imgs = Image.open(file_list[1])
        file_names= file_list

    return imgs, file_names

# for testing on single images
def get_single_image(args,file_path=None):

    if file_path is None:
        imgs_name = ["CLASSIC", None]
        img_dir = 'data' if args.test_dataset in imgs_name else os.path.join(args.dataset_dir, args.test_dataset)
        file_list = os.listdir(img_dir)
        data =[]
        for i in file_list:
            data.append(os.path.join(img_dir,i))
        return data
    else:
        img = cv.imread(file_path)
        h,w,c=img.shape
        x = np.array(img, dtype=np.float32)
        if h==args.image_height and w==args.image_width:
            pass
        else:
            x = cv.resize(x, dsize=(args.image_width, args.image_height))
        # x = x[:, :, arg.channel_swap] # while using opencv it is not necessary
        x -= args.mean_pixel_values[:-1]
        img_info = (file_path,img.shape)
        return x, img_info

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
        if len(tmp.shape)==3 or len(tmp.shape)==2:
            if len(tmp.shape)==2:
                tmp = np.uint8(image_normalization(tmp))
                tmp=cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
                tmp = cv.bitwise_not(tmp)
            else:
                if tmp.shape[-1]==3:
                    tmp=restore_rgb([arg.channel_swap,arg.mean_pixel_values[:3]],tmp)
                    tmp = np.uint8(image_normalization(tmp))
                else:
                    tmp = np.squeeze(tmp)
                    tmp = np.uint8(image_normalization(tmp))
                    tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
                    tmp = cv.bitwise_not(tmp)
        else:
            tmp= np.squeeze(tmp)
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