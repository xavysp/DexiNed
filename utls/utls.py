""" Utilities

"""
import numpy as np
import time
from termcolor import colored
import os,glob
import h5py

def get_model_trained_name(sess=None,data_dir=None, last_update=True):

    if sess is not None or data_dir is not None:
        path = os.path.join(data_dir,'*.ckpt')
        if last_update:
            mfiles= glob.glob(path)
            last_file=max(mfiles,key=os.path.getctime)
            return os.path.join(path,last_file)
        else:
            print('Not performet yet')

def gamma_correction(i, g,gamma=True):
    """Gamma correction
    This function is for gamma corrections and de-correction 0.4040 0.3030 0.6060
    :param i: image data
    :param g: gamma value
    :param gamma: if true do gamma correction if does not degamma correction
    :return:if gamma gamma corrected image else image without gamma correction
    """
    i = np.float32(i)
    if gamma:
        img=i**g
    else:
        img=i**(1/g)
    return img

def image_normalization(img, img_min=0, img_max=255):
    """ Image normalization given a minimum and maximum

    This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image given a scale
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img


def normalization_data_0255(data):
    """
    data normalization in 0 till 1 range
    :param data:
    :return:
    """
    epsilon = 1e-12
    if not len(data.shape)==2:
        n_imgs = data.shape[0]
        # data = np.float32(data)
        if data.shape[-1]==3 and len(data.shape)==3:
            data = ((data - np.min(data)) * 255 / ((np.max(data) - np.min(data)) + epsilon))
            # data = ((data - np.min(data)) * 254 / (np.max(data) - np.min(data)))+1

        elif data.shape[-1]==3 and len(data.shape)==4:
            for i in range(n_imgs):
                img = data[i,...]
                data[i,:,:,:] = ((img - np.min(img)) * 255 / ((np.max(img) - np.min(img))+epsilon))
        # print("Data normalized with:", data.shape[-1], "channels")
        elif data.shape[-1]>3 or len(data.shape)>=4:
            print('error normalizin 0-255 line 30')
        else:
            print('error normalizin 0-255 line 32')

        return data
    elif len(data.shape)==2:
        # data = ((data-np.min(data))*255/(np.max(data)-np.min(data)))
        data = ((data-np.min(data))*255/((np.max(data) - np.min(data)) + epsilon))

        return data
    else:
        print('error normalization 0-255')

def normalization_data_01(data):
    """
    data normalization in 0 till 1 range
    :param data:
    :return:
    """
    epsilon = 1e-12
    if np.sum(np.isnan(data))>0:
        print('NaN detected before Normalization')
        return 'variable has NaN values'
    if len(data.shape)>3:
        n_imgs = data.shape[0]
        data = np.float32(data)
        if data.shape[-1]==3:
            for i in range(n_imgs):
                img = data[i,:,:,:]
                data[i,:,:,:] = ((img - np.min(img)) * 1 / ((np.max(img) - np.min(img))+epsilon))

        elif data.shape[-1]==4:
            print('it is a  little naive, check it in line 64 seg utils.py')
            for i in range(n_imgs):
                nir = data[i, :, :, -1]
                nir = ((nir - np.min(nir)) * 1 / ((np.max(nir) - np.min(nir)) + epsilon))
                img = data[i, :, :, 0:3]
                img = ((img - np.min(img)) * 1 / ((np.max(img) - np.min(img)) + epsilon))
                data[i, :, :, 0:3] = img
                data[i, :, :, -1] = nir
        elif data.shape[-1]==2:
            #normalization according to channels
            print('check line 70 utils_seg.py')
            for i in range(n_imgs):
                im = data[i,:,:,0]
                N = data[i,:,:,-1]
                data[i,:,:,0]= ((im-np.min(im))*1/(np.max(im)-np.min(im)))
                data[i, :, :, -1] = ((N - np.min(N)) * 1 / (np.max(N) - np.min(N)))

        elif data.shape[-1]==1:
            for i in range(n_imgs):
                img = data[i, :, :, 0]
                data[i, :, :, 0] = ((img - np.min(img)) * 1 / ((np.max(img) - np.min(img))+epsilon))
        else:
            print("error normalizing line 83")
        if np.sum(np.isnan(data)) > 0:
            print('NaN detected after normalization')
            return 'variable has NaN values'
        return data

    else:
        if np.max(data) ==0 and np.min(data)==0:
            return data
        if np.sum(np.isnan(data)) > 0:
            print('NaN detected before normalization')
            return 'variable has NaN values'

        data = ((data - np.min(data)) * 1 / ((np.max(data) - np.min(data))+epsilon))
        if np.sum(np.isnan(data)) > 0:
            print('NaN detected after normalization')
            return 'variable has NaN values'
        return data

# _________ text visualization ____________
def get_local_time():
    return time.strftime("%d %b %Y %Hh%Mm%Ss", time.localtime())

def print_info(info_string, quite=False):

    info = '[{0}][INFO]{1}'.format(get_local_time(), info_string)
    print(colored(info, 'green'))

def print_error(error_string):

    error = '[{0}][ERROR] {1}'.format(get_local_time(), error_string)
    print (colored(error, 'red'))

def print_warning(warning_string):

    warning = '[{0}][WARNING] {1}'.format(get_local_time(), warning_string)

    print (colored(warning, 'blue'))
# ___________ End text visualization

# ___________ read list of files
def read_files_list(list_path,dataset_name=None):
    mfiles = open(list_path)
    file_names = mfiles.readlines()
    mfiles.close()

    file_names = [f.strip() for f in file_names]
    return file_names

def split_pair_names(opts, file_names, base_dir=None):
    # If base_dir is None, it assume that the list have the complete image source
    if opts.model_state=='train':

        if base_dir==None:
            file_names =[c.split(' ') for c in file_names]
        else:
            if opts.train_dataset.lower()=='biped':
                x_base_dir=os.path.join(base_dir,'imgs',opts.model_state)
                y_base_dir =os.path.join(base_dir,'edge_maps',opts.model_state)
                file_names = [c.split(' ') for c in file_names]
                file_names = [(os.path.join(x_base_dir, c[0]),
                               os.path.join(y_base_dir, c[1])) for c in file_names]
            else:
                file_names = [c.split(' ') for c in file_names]
                file_names = [(os.path.join(base_dir, c[0]),
                               os.path.join(base_dir, c[1])) for c in file_names]
        return file_names
    else:
        # ******************* for data testing ****************************
        if base_dir == None:
            file_names = [c.split(' ') for c in file_names]
        else:
            if opts.test_dataset.lower() == 'biped':
                x_base_dir = os.path.join(base_dir,'imgs', opts.model_state)
                y_base_dir = os.path.join(base_dir,'edge_maps', opts.model_state)
                file_names = [c.split(' ') for c in file_names]
                file_names = [(os.path.join(x_base_dir, c[0]),
                               os.path.join(y_base_dir, c[1])) for c in file_names]
            else:
                file_names = [c.split(' ') for c in file_names]
                file_names = [(os.path.join(base_dir, c[0]),
                               os.path.join(base_dir, c[1])) for c in file_names]

        return file_names
# ____________ End reading files list

# _____________ H5 file manager _________

def h5_reader(path):
    """ Read H5 file
    Read .h5 file format data h5py <<.File>>
    :param path:file path of desired file
    :return: dataset -> contain images data for training;
    label -> contain  training label values (ground truth)
    """
    with h5py.File(path, 'r') as hf:
        n_variables = len(list(hf.keys()))
        # choice = True  # write
        if n_variables==3:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            test = np.array(hf.get('test'))
        elif n_variables==2:

            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            test=None
        elif n_variables == 1:
            data = np.array(hf.get('data'))
            label=None
            test=None
        else:
            data = None
            label = None
            test = None
            print("Error reading path: ",path)

        print(n_variables, " vars opened from: ", path)
        return data, label, test

def save_h5_data(savepath,data, label, predi = None, data_name=None,
                 label_name=None, predi_name=None):
    if data_name==None or label_name==None:
        if np.any(predi == None):

            with h5py.File(savepath, 'w') as hf:
                hf.create_dataset('data', data=data)
                hf.create_dataset('label', data=label)
                print("Data [", data.shape, "and label ", label.shape, "] saved in: ", savepath)
        else:
            with h5py.File(savepath, 'w') as hf:
                hf.create_dataset('data', data=data)
                hf.create_dataset('label', data=label)
                hf.create_dataset('predi', data=predi)
                print("Input data [", data.shape, ", label ", label.shape, "and predi ", predi.shape,"] saved in: ", savepath)


    else:
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset(data_name, data=data)
            hf.create_dataset(label_name, data=label)
            hf.create_dataset(predi_name,data=predi)
            print("[",data_name, data.shape, ", ",
                  label_name, label.shape," and ",predi_name, predi.shape, "] saved in: ", savepath)

def save_variable_h5(savepath, data):

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)

        print("Data [", len(data), "] saved in: ", savepath)
# ___________ End h5 file manager ____________

# ____________ Restoring RGB former values _______

def restore_rgb(config,I):
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
            x = x[:, :, config[0]]
            x = normalization_data_0255(x)
            I[i,:,:,:]=x
    elif len(I.shape)==3 and I.shape[-1]==3:
        I = np.array(I, dtype=np.float32)
        I += config[1]
        I = I[:, :, config[0]]
        I = normalization_data_0255(I)
    else:
        print_error("Sorry the input data size is out of our configuration")
    print_info("The enterely I data {} restored".format(I.shape))
    return I

def restore_edgemap(config,I):
    """ Not finished coding ***
    :param config: args.target_regression = True or False
    :param I:  input image data
    :return: restored image data
    """
    print_error("Sorry this function is not ready")
    if len(I.shape)>3 and I.shape[3]==1:
        n = I.shape[0]
        for i in range(n):
            y=I[i,...]
    elif len(I.shape)==3 and I.shape[-1]==1:

        I = np.array(I.convert('L'), dtype=np.float32)
        if config:
            bin_I = I / 255.0
        else:
            bin_I = np.zeros_like(I)
            bin_I[np.where(I)] = 1

        bin_I = bin_I if bin_I.ndim == 2 else bin_I[:, :, 0]
        bin_y = np.expand_dims(bin_I, axis=2)
    else:
        print_error("Sorry the input data size is out of our configuration")
    return I

def tensor_norm_01(data):
    """
    tensor means that the size image is [batch-size,img_width, img_height, num_channels]
    :param data:
    :return:
    """
    data = np.array(data)
    if np.sum(np.isnan(data))>0:
        print('NaN detected before Normalization')
        return 'variable has NaN values'
    if len(data.shape)>3:
        n_imgs = data.shape[0]
        data = np.float32(data)
        if data.shape[-1]==3:
            for i in range(n_imgs):
                img = data[i,:,:,:]
                data[i,:,:,:] = image_normalization(img,img_min=0,img_max=1)

        elif data.shape[-1]==4:
            print('it is a  little naive, check it in line 64 seg utils.py')
            for i in range(n_imgs):
                nir = data[i, :, :, -1]
                nir = image_normalization(nir,img_min=0,img_max=1)
                img = data[i, :, :, 0:3]
                img = image_normalization(img,img_min=0,img_max=1)
                data[i, :, :, 0:3] = img
                data[i, :, :, -1] = nir
        elif data.shape[-1]==2:
            #normalization according to channels
            print('check line 70 utils_seg.py')
            for i in range(n_imgs):
                im = data[i,:,:,0]
                N = data[i,:,:,-1]
                data[i,:,:,0]= image_normalization(im,img_min=0,img_max=1)
                data[i, :, :, -1] = image_normalization(N,img_min=0,img_max=1)

        elif data.shape[-1]==1:
            x=[]
            for i in range(n_imgs):
                img = data[i, :, :, 0]
                img= image_normalization(img,img_min=0,img_max=1)
                x.append(img)
            data=x
        else:
            print("error normalizing line 83")
        if np.sum(np.isnan(data)) > 0:
            print('NaN detected after normalization')
            return 'variable has NaN values'
        return data
    else:
        print('Please use image_normalization() function')