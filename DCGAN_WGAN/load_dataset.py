from __future__ import division
from PIL import Image
import numpy as np
import os
from config import config



def load_train_dataset(path, train_size, image_size):
    """
    :path: dataset path
    :train_size: number of images to train with
    :image_size: size of each image
    """
    
    train_path_original = path + '/train/original/'
    train_path_style = path + '/train/style/'

    train_image_num = len([name for name in os.listdir(train_path_original)
                           if os.path.isfile(os.path.join(train_path_original, name))])

    if train_size == -1:  
        # use all training set
        train_size = train_image_num
        train_image = np.arange(0, train_size)
    else:                
        # use specified number of training set
        train_image = np.random.choice(np.arange(0, train_image_num), size=train_size, replace=False)

    train_original = np.zeros((train_size, image_size))
    train_style = np.zeros((train_size, image_size))

    idx = 0
    for img in train_image:
        img_array = np.asarray(Image.open(train_path_original + str(img) + '.tif'))    
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        train_original[idx, :] = img_array

        img_array = np.asarray(Image.open(train_path_style + str(img) + '.tif'))   
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        train_style[idx, :] = img_array

        idx += 1
        if idx % 100 == 0:
            print('image / train_size : %d / %d = %.2f percent done' % (idx, train_size, idx/train_size))

    return train_original, train_style


def load_test_dataset_patches(path, test_start, test_end, image_size):
    """
    load test dataset with image patches
    
    :path: dataset path
    :test_start: start index of the test images
    :test_end: end index of the test images
    :image_size: size of each image
    """
    if config.val_patches:
        test_path_original = path + '/val/original/'
        test_path_style = path + '/val/style/'    
    elif config.test_patches:
        test_path_original = path + '/test/original/'
        test_path_style = path + '/test/style/'

    test_image_num = len([name for name in os.listdir(test_path_original)
                         if os.path.isfile(os.path.join(test_path_original, name))])

    test_size = test_end - test_start
    if test_size == -1:  
        # use all test set
        test_size = test_image_num
        test_image = np.arange(0, test_size)
    else:                
        # use specified number of test set
        test_image = np.arange(test_start, test_end)

    test_original = np.zeros((test_size, image_size))
    test_style = np.zeros((test_size, image_size))

    idx = 0
    for img in test_image:
        img_array = np.asarray(Image.open(test_path_original + str(img) + '.tif'))
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        test_original[idx, :] = img_array

        img_array = np.asarray(Image.open(test_path_style + str(img) + '.tif'))
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        test_style[idx, :] = img_array

        idx += 1
        if idx % 100 == 0:
            print('image / test_size : %d / %d = %.2f percent done' % (idx, test_size, idx/test_size))

    return test_original, test_style


def load_test_dataset(ind):
    """
    load test dataset with full images
    
    :ind: index of the test images
    """
    test_path_original = '../data/full/original/'
    test_path_style = '../data/full/style/'

    img_array = np.asarray(Image.open(test_path_original + str(ind) + '.tif'))
    image_height = img_array.shape[0]
    image_width = img_array.shape[1]
    test_original = np.float32(img_array.reshape(1, -1)) / 255

    img_array = np.asarray(Image.open(test_path_style + str(ind) + '.tif'))
    test_style = np.float32(img_array.reshape(1, -1)) / 255

    return test_original, test_style, image_height, image_width
