# -*- coding: utf-8 -*-

"""
apply_cnn.py

created: 15:56 - 18/08/2020
author: Cornelius
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# stamdard lib imports
import json
from abc import ABC, abstractmethod
import logging
import os.path

# 3rd party imports
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
import cv2

# project imports
from source import preprocess_image

# --------------------------------------------------------------------------------------------------
# CONSTANTS AND MODULE-LVL REFERENCES
# --------------------------------------------------------------------------------------------------

# configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

filedir = os.path.dirname(__file__)

# path to json with dog-classes
DOG_CLASSES_PTH = os.path.join(filedir, './dog_classes_en.json')

# path to model used for face/human detection
FACE_CLASSIFIER_PTH = os.path.join(filedir, '../haarcascade/haarcascade_frontalface_alt.xml')

# pathes to trained CNNs for classification of dog-breeds
MODEL_PATHES = dict(
    resnet50=os.path.join(filedir, '../models/model.best.resnet50.hdf5'),
    from_scratch=os.path.join(filedir, '../models/model.best.from_scratch.hdf5'),
    xception=os.path.join(filedir, '../models/model.best.xception.hdf5'),
    vgg16=os.path.join(filedir, '../models/model.best.vgg16.hdf5'),
)

# reference to current model loaded
__model = None

# model used for dog-detection
__ResNet50_model = ResNet50(weights='imagenet')

# --------------------------------------------------------------------------------------------------
# FUNCTIONS AND CLASSES
# --------------------------------------------------------------------------------------------------


def face_detector(img_arg):
    """
    loads image and detects faces in images and returns boolean whether a face was found or not.

    Parameters
    ----------
    img_arg: str or filehandler
        path to image or filelike object

    Returns
    -------
    bool
    """
    if hasattr(img_arg, 'read') and hasattr(img_arg, 'seek'):
        # jump to start of file
        img_arg.seek(0)
        # transform bytes
        file_bytes = np.asarray(bytearray(img_arg.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_arg)
    # initialize pre-trained face detector
    face_cascade = cv2.CascadeClassifier(FACE_CLASSIFIER_PTH)
    # transform to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces if at least one detected return True
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_arg):
    """
    loads image and uses resnet50-model trained on imagenet for detection of dogs. If a dog is contained in the image
    True will be returned; False otherwise.

    Parameters
    ----------
    img_arg: str or filehandler
        path to image or filehandler

    Returns
    -------
    bool
    """
    img = preprocess_image.preprocess_resnet50(img_arg)
    # get index of maximum from prediction vector (most likely class)
    resnet50pred = np.argmax(__ResNet50_model.predict(img))
    # dog-classes are in imagenet between index 151 to 268
    return (resnet50pred <= 268) & (resnet50pred >= 151)


def predict_dog_breed(img_arg, model='resnet50'):
    """
    Loads an image of a dog, classifies it's breed and returnes it's breed name. If image doesn't contain a dog, the
    CNN will return the most likely dog class anyway.

    Parameters
    ----------
    img_arg: str or filehandler
        path to image or filehandler
    model: ['resnet50', 'from_scratch', 'xception', 'vgg16']
        trained CNN to be used for classification of the dog-breed

    Returns
    -------
    str
        name of the dog-breed
    """
    global __model
    if not __model:
        __model = DogPredictor(model)
    __model.change_type(model)
    return __model.predict_name(img_arg)


def classify_image(img_arg, model='resnet50'):
    """
    identifies whether a dog or a human is in the image. If one is in the image the most likely dog-breed will be
    determined and message returned. If neither is the case a error-message will be returned.

    Parameters
    ----------
    img_arg: str
        path to the image
    model: ['resnet50', 'from_scratch', 'xception', 'vgg16']
        trained CNN to be used for classification of the dog-breed

    Returns
    -------
    None
    """
    # preprocess img
    if dog_detector(img_arg):
        breed = predict_dog_breed(img_arg, model)
        print('Aawww!!! What a cutie! If that is not a %s!' % breed)
    elif face_detector(img_arg):
        breed = predict_dog_breed(img_arg, model)
        print('You look like a %s! Are you sure you are a human and not a dog??' % breed)
    else:
        print('What is that?! Neither a dog nor a human - I am sure!')


class DogPredictor:
    """
    class for loading different kinds of CNNs with knowledge transfer and appliing them to images of dogs for
    determination of the most likely dog-breed.

    Attributes
    ----------
    model: tensorflow.python.keras.engine.sequential.Sequential
        trained model on top of the model used for transfered learning
    tf_model: KnowledgeTransfer
        model used for transfer learning
    type: str
        type of the currently loaded model

    Methods
    -------
    change_type
    load_model
    predict_props
    predict_name
    """

    # load json will class-names
    with open(DOG_CLASSES_PTH, 'r') as fin:
        __dog_names = json.load(fin)

    def __init__(self, model='from_scratch'):
        """
        initialize DogPredictor

        Parameters
        ----------
        model: ['resnet50', 'from_scratch', 'xception', 'vgg16']
            which model (CNN) to load
        """
        self.model = None
        self.tf_model = None
        self.type = None
        self.change_type(model)

    def change_type(self, model):
        """
        change currently active CNN.

        Parameters
        ----------
        model: ['resnet50', 'from_scratch', 'xception', 'vgg16']
            which model (CNN) to load

        Returns
        -------
        None
        """
        if self.type == model:
            return

        if model == 'resnet50':
            self.tf_model = Resnet50()
        elif model == 'from_scratch':
            self.tf_model = NullTransfer()
        elif model == 'xception':
            self.tf_model = Xception()
        elif model == 'vgg16':
            self.tf_model = VGG16()
        else:
            raise ValueError
        self.load_model(MODEL_PATHES.get(model))
        self.type = model

    def load_model(self, pth):
        """
        load stored CNN

        Parameters
        ----------
        pth: str
            path to trained CNN

        Returns
        -------
        None
        """
        self.model = load_model(pth)

    def predict_props(self, inp):
        """
        predict probabilities of different dog-breeds in image based on the current loaded CNN.

        Parameters
        ----------
        inp: str or filehandler
            path to image or filehandler

        Returns
        -------
        np.array
            vector with probabilities of the dog-breed classes
        """
        # feed image to model used for knowledge transfer
        knowledge = self.tf_model.transfer(inp, need_preprocessing=True)
        # the knowledge from previous model is feeded to main CNN
        return self.model.predict(knowledge)

    def predict_name(self, inp):
        """
        predict most likely dog breed in image based on the current loaded CNN.

        Parameters
        ----------
        inp: str or filehandler
            path to image or filehandler

        Returns
        -------
        str
            name of the most likely dog-breed
        """
        props = self.predict_props(inp)
        # get most likely index
        maxidx = np.argmax(props)
        return DogPredictor.__dog_names[maxidx]


class KnowledgeTransfer(ABC):
    """
    abstract class used for preprocessing images according to requirements of model used for transfered learning and
    making data aggregation based on the model.
    """

    @property
    @abstractmethod
    def model(self):
        """ abstract property with model used for tf (must have a predict-method) """
        pass

    @abstractmethod
    def preprocess(self, image_path):
        """ abstract method for preprocessing images based oon the model's needs """
        pass

    def transfer(self, inp, need_preprocessing=False):
        """
        Apply the transfer model on the input.

        Parameters
        ----------
        inp: str or np.array
            path to image or already preprocessed image in form of an array
        need_preprocessing: bool
            whether input contains path to an image or is already preprocessed input

        Returns
        -------
        np.array
            aggregated data passed through tf-model
        """
        if need_preprocessing:
            inp = self.preprocess(inp)
        return self.model.predict(inp)


class VGG16(KnowledgeTransfer):
    """ VGG16-TF-model trained on imagenet """

    def __init__(self):
        from keras.applications.vgg16 import VGG16 as VGG16Local
        self.__model = VGG16Local(weights='imagenet', include_top=False)

    @property
    def model(self):
        return self.__model

    def preprocess(self, image_path):
        """ delegate to VGG16's preprocess function """
        return preprocess_image.preprocess_vgg16(image_path)


class Resnet50(KnowledgeTransfer):
    """ Resnet50-TF-model trained on imagenet """

    def __init__(self):
        # ResNet50 already imported in outer scope
        self.__model = ResNet50(weights='imagenet', include_top=False)

    @property
    def model(self):
        return self.__model

    def preprocess(self, image_path):
        """ delegate to Resnet50's preprocess function """
        return preprocess_image.preprocess_resnet50(image_path)


class Xception(KnowledgeTransfer):
    """ Xception-TF-model trained on imagenet """

    def __init__(self):
        from keras.applications.xception import Xception as XceptionLocal
        self.__model = XceptionLocal(weights='imagenet', include_top=False)

    @property
    def model(self):
        return self.__model

    def preprocess(self, image_path):
        """ delegate to xception's preprocess function """
        return preprocess_image.preprocess_xception(image_path)


class NullModel:
    """ pseudo-model class passing input through without any changes. Used for CNNs without Knowledge transfer """

    def predict(self, arr):
        """ implementation of required predict method called by KnowledgeTransfer-class """
        return arr


class NullTransfer(KnowledgeTransfer):
    """
    Model passing through input without transfer of knowledge. Used for CNNs without Knowledge transfer.
    """
    def __init__(self):
        self.__model = NullModel()

    @property
    def model(self):
        return self.__model

    def preprocess(self, image_path):
        """
        loads image and transfers to tensor

        Parameters
        ----------
        image_path: str

        Returns
        -------
        np.array
        """
        return preprocess_image.path_to_tensor(image_path)
