
import numpy as np
import os
import cv2 as cv
import h5py

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

def make_dirs(paths): # make path or paths dirs
    if not os.path.exists(paths):
        os.makedirs(paths)
        print("Directories have been created: ",paths)
        return True
    else:
        print("Directories already exists: ", paths)
        return False

def read_files_list(list_path,dataset_name=None):
    mfiles = open(list_path)
    file_names = mfiles.readlines()
    mfiles.close()

    file_names = [f.strip() for f in file_names] # this is for delete '\n'
    return file_names

def data_parser(dataset_dir,dataset_name, list_name=None,training=True):

    if dataset_name.upper()!="CLASSIC":
        files_name = list_name # dataset base dir
        list_path = os.path.join(dataset_dir, files_name)
        data_list = read_files_list(list_path)
        tmp_list = data_list[0]
        tmp_list = tmp_list.split(' ')
        n_lists = len(tmp_list)
    if dataset_name.upper()=='BIPED':
        in_dir = os.path.join(dataset_dir,'imgs','train') if training else \
            os.path.join(dataset_dir, 'imgs', 'test')
        gt_dir = os.path.join(dataset_dir,'edge_maps','train') if training else \
            os.path.join(dataset_dir, 'edge_maps', 'test')
    elif dataset_name.upper()=="CLASSIC":
        list_path= dataset_dir
        data_list = os.listdir(dataset_dir)
        data_list = [os.path.join(dataset_dir,i)for i in data_list]
        n_lists = None
    else:

        in_dir = dataset_dir
        gt_dir = dataset_dir

    if n_lists==1:
        data_list = [c.split(' ') for c in data_list]
        data_list = [(os.path.join(in_dir, c[0])) for c in data_list]
    elif n_lists==2:
        data_list = [c.split(' ') for c in data_list]
        data_list = [(os.path.join(in_dir, c[0]),
                       os.path.join(gt_dir, c[1])) for c in data_list]
    else:
        print('There are just two entry files, dataset:', dataset_name)

    num_data = len(data_list)
    print(" Enterely training set-up from {}, size: {}".format(list_path, num_data))

    all_train_ids = np.arange(num_data)
    np.random.shuffle(all_train_ids)
    if training:

        train_ids = all_train_ids[:int(0.9 * len(data_list))]
        valid_ids = all_train_ids[int(0.9 * len(data_list)):]

        print("Training set-up from {}, size: {}".format(list_path, len(train_ids)))
        print("Validation set-up from {}, size: {}".format(list_path, len(valid_ids)))
        train_list = [data_list[i] for i in train_ids]
        val_list = [data_list[i] for i in valid_ids]
        cache_info = {
            "files_path": data_list,
            "train_paths":train_list,
            "val_paths": val_list,
            "n_files": num_data,
            "train_indices": train_ids,
            "val_indices": valid_ids
        }
    else:
        data_shape = []
        data_name=[]
        for tmp_path in data_list:
            tmp_img = cv.imread(tmp_path[0]) if len(tmp_path)<4 else cv.imread(tmp_path)
            tmp_name = os.path.basename(tmp_path[0])if len(tmp_path)<4 else\
                os.path.basename(tmp_path)
            tmp_shape = tmp_img.shape[:2]
            data_shape.append(tmp_shape)
            data_name.append(tmp_name)
            is_gt = True if len(tmp_path)<4 else False

        print("Testing set-up from {}, size: {}".format(list_path, len(all_train_ids)))
        cache_info = {
            "files_path": data_list,
            "n_files": num_data,
            "is_gt":is_gt,
            "data_name": data_name,
            "data_shape":data_shape
        }

    return cache_info

def cv_imshow(title='image',img=None):

    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

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

def visualize_result(x,y,p, img_title):
    """
    function for tensorflow results
    :param imgs_list: a list of prediction, gt and input data
    :param arg:
    :return: one image with the whole of imgs_list data
    """
    imgs_list = []
    imgs_list.append(x)
    imgs_list.append(y)
    for i in p:
        tmp = i.numpy()[2]
        imgs_list.append(tmp)

    n_imgs = len(imgs_list)
    data_list =[]
    for i in range(n_imgs):
        tmp = imgs_list[i]
        if tmp.shape[-1]==3:
            # tmp = np.transpose(np.squeeze(tmp),[1,2,0])
            # tmp=restore_rgb([arg.channel_swap,arg.mean_pixel_values[:3]],tmp)
            tmp = tmp[:,:,[2,1,0]]
            tmp = np.uint8(image_normalization(tmp))
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

    # last processing, resize and add title
    vis_imgs = cv.resize(imgs, (int(imgs.shape[1] * 0.8), int(imgs.shape[0] * 0.8)))
    BLACK = (0, 0, 255)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_size = 1.1
    font_color = BLACK
    font_thickness = 2
    x, y = 30, 30
    vis_imgs = cv.putText(vis_imgs, img_title, (x, y), font, font_size, font_color, font_thickness,
                          cv.LINE_AA)
    return vis_imgs

def tensor2image(image, img_path =None,img_shape=None):
    tmp_img = np.squeeze(image_normalization(image))
    tmp_img = np.uint8(cv.resize(tmp_img,(img_shape[1],img_shape[0])))
    tmp_img = cv.bitwise_not(tmp_img)
    cv.imwrite(img_path,tmp_img)
    print("Prediction saved in:",img_path)

def tensor2image(image, img_path =None,img_shape=None):
    tmp_img = np.squeeze(image_normalization(image))
    tmp_img = np.uint8(cv.resize(tmp_img,(img_shape[1],img_shape[0])))
    tmp_img = cv.bitwise_not(tmp_img)
    cv.imwrite(img_path,tmp_img)
    print("Prediction saved in:",img_path)

def h5_writer(vars,path, is_multivar=False):
    # vars is a single variable o a list of multiple variables

    with h5py.File(path, 'w') as hf:
        if is_multivar:
            assert isinstance(vars, list)
            for idx, var in enumerate(vars):
                hf.create_dataset('data'+str(idx+1), data=var)
            # print('Set of data',len(vars),'saved in ', path)
        else:
            hf.create_dataset('data', data=vars)
            # print("Data [", len(vars), "] saved in: ", path)
