import skimage.io as io
import os
from scipy.fftpack import fftfreq
import numpy as np
import time

def load_image(url):

    img = io.imread(url)
    return img

def load_images_parallel(urls = []):
    if urls == []:
        return None
    """using concurrent.futures"""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(load_image, urls)
        images = list(results)
    return images

def get_image(path, idx = None, file_type=None):
    if file_type is None:
        file_type = 'tif'
    if type(idx) is not list:
        idx = [idx]
    if type(path) is not list or type(path) is not tuple:
        if type(path) is str:
            if os.path.isdir(path):
                images = list(io.imread_collection(path + '/*.' + file_type).files)

                if idx[0] is None:
                    image_path = images
                    idx = list(range(len(images)))
                    
                else:
                    image_path = [images[i] for i in idx]
    
                image = load_images_parallel(image_path)
                
            elif os.path.isfile(path):
                image = io.imread(path)
                image_path = path
            
                if len(images.shape) == 2:
                    image = images
                elif len(images.shape) == 3:
                    image = images[idx,:,:] if idx is not None else images
                else:
                    image = images[idx,:,:,:] if idx is not None else images
                image = [image]
        elif 'numpy' in str(type(path)) or 'torch' in str(type(path)) or 'jax' in str(type(path)):
            if len(path.shape) == 2:
                image = path
            elif len(path.shape) == 3:
                image = path[idx,:,:] if idx is not None else path
            else:
                # print('path shape', path, 'idx', idx)
                image = path#[idx,:,:,:] if idx is not None else path
            # image = [image]
            image_path = None
        elif 'ImageCollection' in str(type(path)):
            image_path = path.files
            image_path = [image_path[i] for i in idx] if idx is not None else image_path
            image = load_images_parallel(image_path)
        elif 'npy' in str(type(path)):
            image = np.load(path).astype('float32')
            image_path = None
            
        elif 'npz' in str(type(path)):
            image = np.load(path)['arr_0'].astype('float32')
            image_path = None

        else:
            image = path
            image_path = None
            # print("couldn't load image from path", torch_reshape(image).shape)
            pass
    else:
        image = []
        image_path = []
        for p in path:
            image_, image_path_ = get_image(p, idx, file_type)
            image.append(image_)
            image_path.append(image_path_)
    return image, image_path


def check_if_processed(path):
    print(path)
    if os.path.exists(path):
        return True
    else:
        return False
    
def in_progress(text_file, path):
    if os.path.exists(text_file):
        with open(text_file, 'r') as f:
            paths = f.readlines()
        if path in paths:
            return True
        
    with open(text_file, 'a') as f:
        f.write(path+'\n')
    return None
   
def join_dictionaries(dict2, base_dict):
    res = base_dict.copy()
    res.update(dict2)
    return res
   
def get_setup_info(dict = {}):
    #rearrange them in a descending order based on length
    dict = {k: v for k, v in sorted(dict.items(), key=lambda item: len(item[0]) + len(str(item[1])), reverse=True)}
    len_line = 0
    for key, value in dict.items():
        if type(value) == str or  type(value) == int or type(value) == float or type(value) == bool: 
            if len(key) > len_line:
                len_line = len(key)
        elif type(value) == np.ndarray:
            if len(value.shape) == 0:
                if len(key) > len_line:
                    len_line = len(key)
        else: 
            try:
                from torch_utils import tensor_to_np
                if type(tensor_to_np(value)) == np.ndarray and len(tensor_to_np(value).shape) == 0:
                    if len(key) > len_line:
                        len_line = len(key)
            except:
                pass
    len_line += 10
    line = '_'*len_line 
    information = line + '\n'
    for key, value in dict.items():
        if type(value) == str or type(value) == int or type(value) == float or type(value) == bool:
            information += '| ' +key +': '+ str(value) +' \n'
        elif type(value) == np.ndarray and len(value.shape) == 0:
            information += '| ' +key +': '+ str(value) +' \n'
        else:
            try:
                # from torch_utils import tensor_to_np
                if type(tensor_to_np(value)) == np.ndarray and len(tensor_to_np(value).shape) == 0:
                    information += '| ' +key +': '+ str(tensor_to_np(value)) +' \n'
            except:
                pass
    information += line + ' \n'
    return information, len_line

def get_file_nem(dict):
    name = ''
    important_keys = ['experiment_name', 'abs_ratio', 'iter_num', 'downsampling_factor', 'l1_ratio', 'contrast_ratio', 'normalized_ratio', 'brightness_ratio', 'contrast_normalize_ratio', 'brightness_normalize_ratio', 'l2_ratio', 'fourier_ratio']
    for key in important_keys:
        if key in dict.keys():
            name += key + '_' + str(dict[key]) + '__'
    return name
  
def create_table_info(dict={}):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame()
    for key, value in dict.items():
        if type(value) != np.ndarray:
            df[key] = [value]
        elif type(value) == np.ndarray and len(value.shape) == 0:
            df[key] = [value]
    df = df.T
    #create a plot with the information
    fig, ax = plt.subplots(figsize=(20, 10))
    #make the rows and columns look like a table
    ax.axis('tight')
    ax.axis('off')
    #create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', rowLabels=df.index, cellLoc='center')
    #change the font size
    table.set_fontsize(14)
    #change the cell height
    table.scale(1, 2)
    
    return df,ax, table

def shorten(string):
    if 'e' in string:
        left = string.split('e')[0][:7]
        right = string.split('e')[1][:7]
        return left + 'e' + right
    else:
        if '.' in string:
            count = 0
            for i in range(len(string.split('.')[1])):
                if string[i] == '0':
                    count += 1
            return string[:count+5]
        else:
            return string[:7]
        
def give_title(image, title = '', idx = '', min_max = True):    
    if min_max:
        min_val_orig = np.min(image)
        max_val_orig = np.max(image)
        txt_min_val = ' (' + shorten(str(min_val_orig)) +'), '
        txt_max_val = ' (' + shorten(str(max_val_orig)) +')/n'
    else:
        txt_min_val = ''
        txt_max_val = ''    
    title = str(idx+1).zfill(4) if title == '' else title
    return title  + txt_min_val + txt_max_val

def give_titles(images, titles = [], min_max = True):
    titles = [titles] if type(titles) is not list else titles
    if len(titles) <= len(images):
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(titles))]
        n_for_rest = np.arange(len(titles), len(images))
        titles.extend([give_title(images[i], idx=i, min_max = min_max) for i in n_for_rest])
    else:
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(images))]
    return titles

def time_to_string(time):
    if time > 60:
        if time > 3600:
            if time > 3600*24:
                return str(int(time//(3600*24))) + ' days ' + str(int((time%(3600*24))//3600)) + ' hours ' + str(int((time%3600)//60)) + ' minutes ' + str(int(time%60)) + ' seconds'
            else:
                return str(int(time//3600)) + ' hours ' + str(int((time%3600)//60)) + ' minutes ' + str(int(time%60)) + ' seconds'
        else:
            return str(int(time//60)) + ' minutes ' + str(int(time%60)) + ' seconds'
    else:
        return str(int(time%60)) + ' seconds'

def get_list_of_possibilities(value, gap = None, number_of_elements = None):
    if gap is None:
        gap = value * 0.1
    if number_of_elements is None:
        number_of_elements = 6
    values = [value - gap*(i+1) for i in range(number_of_elements//2)]
    values2 = [value + gap*(i+1) for i in range(number_of_elements//2)]
    values.extend([value])
    values.extend(values2)
    values.sort()
    return values

def np_zero_at_boundary(img, width=2):
    img[:width, :] = 0
    img[-width:, :] = 0
    img[:, :width] = 0
    img[:, -width:] = 0
    return img

def rgb2gray(img):
    """ Convert RGB image to grayscale using the colorimetric (luminosity-preserving) method
    
    See e.g. discussion in https://poynton.ca/PDFs/ColorFAQ.pdf page 6 on the benefit of this
    method compared to the classical [0.299, 0.587, 0.114] weights.
    
    """
    return img @ np.array([0.2125, 0.7154, 0.0721])


import scipy.ndimage as spnd
def imresize(image, size, **kwargs):
    """Resize an image to a new size.
    
    kwargs are passed to scipy.ndimage.zoom.
    """
    zoom_factor = size/np.array(image.shape)
    return spnd.zoom(image, zoom_factor, **kwargs)


def positioning_comp(rows, cols, index, pad_by = 0):
    """
    This function takes the index of a compartment and returns how many neighbors it has for each side.
    :param rows: number of rows of the compartments
    :param cols: number of columns of the compartments
    :param index: index of the compartment
    :return: dictionary with the number of neighbors for each side
    """
    if index % cols == 0:
        left = 0
    else:
        left = 1
    if index % cols == cols - 1:
        right = 0
    else:
        right = 1
    if index // cols == 0:
        top = 0
    else:
        top = 1
    if index // cols == rows - 1:
        bottom = 0
    else:
        bottom = 1
    # if pad_by > 0:
    #     left = left * pad_by if right == 1 else left * 2 * pad_by
    #     right = right * pad_by if left == 1 else right * 2 * pad_by
    #     top = top * pad_by if bottom == 1 else top * 2 * pad_by
    #     bottom = bottom * pad_by if top == 1 else bottom * 2 * pad_by
    return {'left': left, 'right': right, 'top': top, 'bottom': bottom}

def compartment_image(image, rows, cols, pad_by = 0, replace = False):
    """
    This function takes an image and splits it into compartments
    :param image: image to split
    :param rows: number of rows to split the image
    :param cols: number of columns to split the image
    :return: list of images
    """
    compartments = []
    height, width = image.shape

    compartment_height = height // rows 
    compartment_width = width // cols
    for i in range(rows):    
        for j in range(cols):
            cell = np.ones((compartment_height + 2 * pad_by, compartment_width+ 2 * pad_by))
            if not replace:
                compartment = image[i * compartment_height:(i + 1) * compartment_height,
                    j * compartment_width:(j + 1) * compartment_width]
                cell[pad_by:compartment_height + pad_by, pad_by:compartment_width + pad_by] = compartment
            else:
                "compartments will have cover bigger area that can be shared with other compartments"
                neighbors = positioning_comp(rows, cols, i * cols + j)
                if neighbors['left'] == 0:
                    left = 0
                    right = 2 * pad_by
                elif neighbors['right'] == 0:
                    left = 2 * pad_by
                    right = 0 
                else:
                    left = pad_by
                    right =  pad_by
                if neighbors['top'] == 0:
                    top = 0
                    bottom = 2 * pad_by
                elif neighbors['bottom'] == 0:
                    top =  2 * pad_by
                    bottom = 0
                else:
                    top = pad_by
                    bottom = pad_by
                compartment = image[i * compartment_height - top:(i + 1) * compartment_height + bottom,
                                    j * compartment_width - left:(j + 1) * compartment_width + right]
                cell = compartment
                
            compartments.append(cell)
    return compartments

def join_compartments(compartments, rows, cols, padded_by = 0, replace = False):
    """
    This function takes a list of images and joins them into a single image
    :param compartments: list of images to join
    :param rows: number of rows to join the images
    :param cols: number of columns to join the images
    :param padded_by: padding that has been added to the compartments
    :return: single image
    """
    compartment_height, compartment_width = compartments[0].shape
    
    if not replace:
                
        compartment_height -= 2 * padded_by
        compartment_width -= 2 * padded_by
        height = compartment_height * rows
        width = compartment_width * cols
        image = np.zeros((height, width))
        unpadded_compartments = [compartments[i][padded_by:compartment_height + padded_by, padded_by:compartment_width + padded_by] for i in range(rows * cols)]
        for i in range(rows):
            for j in range(cols):
                image[i * compartment_height:(i + 1) * compartment_height,
                    j * compartment_width:(j + 1) * compartment_width] = unpadded_compartments[i * cols + j]
    else:
        
        unpadded_compartments = []
        for i in range(rows * cols):
            neighbors = positioning_comp(rows, cols, i)
            cell_height, cell_width = compartments[i].shape
            if neighbors['left'] == 0:
                left = 0
                right = 2 * padded_by
            elif neighbors['right'] == 0:
                left = 2 * padded_by
                right = 0
            else:
                left = padded_by
                right = padded_by
            if neighbors['top'] == 0:
                top = 0
                bottom = 2 * padded_by
            elif neighbors['bottom'] == 0:
                top = 2 * padded_by
                bottom = 0
            else:
                top = padded_by
                bottom = padded_by
            cell_height -= (left + right)
            cell_width -= (top + bottom)
            cell = np.zeros((cell_height, cell_width))
            cell = compartments[i][top:cell_height + top, left:cell_width + left]
            unpadded_compartments.append(cell)
        compartment_height = cell_height
        compartment_width = cell_width
        width = compartment_width * cols
        height = compartment_height * rows
        image = np.zeros((height, width))
        
        for i in range(rows):
            for j in range(cols):
                image[i * compartment_height:(i + 1) * compartment_height,
                    j * compartment_width:(j + 1) * compartment_width] = unpadded_compartments[i * cols + j]
                
    return image

def filter_values(image, min_value = None, max_value = None, replace = None, replace_value = None):
    #replace nan values with the min value
    image = np.nan_to_num(image)
    if min_value is not None:
        image[image < min_value] = min_value 
    if max_value is not None:
        image[image > max_value] = max_value
    if replace is not None:
        image[image == replace] = replace_value
    return image
