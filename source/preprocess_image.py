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

# 3rd party imports
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# project imports


# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
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


def path_to_tensor(img_path, scale=False):
    """
    Loads a image and transforms it into a 4D-numpy of shape (1, 224, 224, 3) array with its RGB-values.

    Parameters
    ----------
    img_path: str
        path to the image to load

    Returns
    -------
    np.array
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    if scale:
        x = scale_tensor(x)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths, scale=False):
    """
    Loads a list of images and transforms them into a 4D-numpy array with their RGB-values.

    Parameters
    ----------
    img_paths: iterable
        iterable with pathes to images

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
    from keras.applications.resnet50 import preprocess_input
    return preprocess_input(path_to_tensor(img_path, scale=False))


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
    from keras.applications.vgg16 import preprocess_input
    return preprocess_input(path_to_tensor(img_path, scale=False))


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
    from keras.applications.xception import preprocess_input
    return preprocess_input(path_to_tensor(img_path, scale=False))
