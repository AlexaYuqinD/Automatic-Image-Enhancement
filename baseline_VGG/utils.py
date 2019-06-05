# coding: utf-8

import os

from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from six.moves import urllib


def download(download_link, file_name, expected_bytes):
    """
    Download pre-trained VGG-19
    
    :param download_link: download link
    :param file_name: file name
    :param expected_bytes: file size
    """
    if os.path.exists(file_name):
        print("VGG-19 pre-trained model is ready")
        return
    print("Downloading the VGG pre-trained model. This might take a while ...")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded VGG-19 pre-trained model', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')


def get_resized_photo(photo_path, width, height, save=True):
    """
    standardize the size of photos
    
    :param photo_path: photo path
    :param width: photo width
    :param height: photo height
    :param save: saving path
    :return: 
    """
    photo = Image.open(photo_path)
    # PIL is column major so you have to swap the places of width and height
    photo = ImageOps.fit(photo, (width, height), Image.ANTIALIAS)
    if save:
        photo_dirs = photo_path.split('/')
        photo_dirs[-1] = 'resized_' + photo_dirs[-1]
        out_path = '/'.join(photo_dirs)
        if not os.path.exists(out_path):
            photo.save(out_path)
    photo = np.asarray(photo, np.float32)
    return np.expand_dims(photo, 0)


def generate_noise_photo(content_photo, width, height, noise_ratio=0.6):
    """
    add white noise to photos
    
    :param content_photo: content photo
    :param width: photo width
    :param height: photo height
    :param noise_ratio: ratio of white noise
    :return: content photo with noise
    """
    noise_photo = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)
    return noise_photo * noise_ratio + content_photo * (1 - noise_ratio)


def save_photo(path, photo):
    photo = photo[0]
    photo = np.clip(photo, 0, 255).astype('uint8')
    scipy.misc.imsave(path, photo)


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass