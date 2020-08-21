# -*- coding: utf-8 -*-

"""
preprocess_image.py

created: 14:56 - 18/08/2020
author: kornel
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# standard lib imports
import logging

# 3rd party imports
import numpy as np
from keras.preprocessing import image
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tqdm import tqdm
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# project imports


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------------------------------------------------------------
# FUNCTIONS AND CLASSES
# --------------------------------------------------------------------------------------------------


def scale_tensor(tensor):
    """
    Norms values between 0 and 255 to vlaues between 0 and 1.

    Parameters
    ----------
    tensor: np.array
        array representing img-data

    Returns
    -------
    np.array
        normed array with values between 0 and 1
    """
    return tensor.astype('float32') / 255


def path_to_tensor(img_arg, scale=False):
    """
    Loads a image and transforms it into a 4D-numpy of shape (1, 224, 224, 3) array with its RGB-values.

    Parameters
    ----------
    img_arg: filehandler or str
        path to the image to load or filehandler dependent on read-argument
    scale: bool
        whether to scale the resulting values of the tensor to values between 0 and 1

    Returns
    -------
    np.array
    """
    # if opened:
    #     img = read_buffer(img_arg)
    # else:
    #     img = load_image(img_arg)
    img = Image.open(img_arg)
    img.convert('RGB')
    img = img.resize((224, 224), Image.NEAREST)

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    arr = image.img_to_array(img)
    if scale:
        arr = scale_tensor(arr)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(arr, axis=0)


def paths_to_tensor(img_paths, scale=False):
    """
    Loads a list of images and transforms them into a 4D-numpy array with their RGB-values.

    Parameters
    ----------
    img_paths: iterable
        iterable with pathes to images
    scale: bool
        whether to scale the resulting values of the tensor to values between 0 and 1

    Returns
    -------
    np.array
        array of images (RGB-arrays)
    """
    list_of_tensors = [path_to_tensor(img_path, scale=scale) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def preprocess_resnet50(img_path):
    """
    Loads image and applies preprocessing based on Resnet50-models and returns 4D-array.

    Parameters
    ----------
    img_path: str
        pth to image

    Returns
    -------
    np.array
    """
    return resnet50_preprocess(path_to_tensor(img_path, scale=False))


def preprocess_vgg16(img_path):
    """
    Loads image and applies preprocessing based on VGG16-models and returns 4D-array.

    Parameters
    ----------
    img_path: str
        pth to image

    Returns
    -------
    np.array
    """
    return vgg16_preprocess(path_to_tensor(img_path, scale=False))


def preprocess_xception(img_path):
    """
    Loads image and applies preprocessing based on Xception-models and returns 4D-array.

    Parameters
    ----------
    img_path: str
        pth to image

    Returns
    -------
    np.array
    """
    return xception_preprocess(path_to_tensor(img_path, scale=False))
