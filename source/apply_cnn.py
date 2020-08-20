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
import preprocess_image

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

filedir = os.path.dirname(__file__)

DOG_CLASSES_PTH = os.path.join(filedir, './dog_classes_en.json')
FACE_CLASSIFIER_PTH = os.path.join(filedir, '../haarcascades/haarcascade_frontalface_alt.xml')
MODEL_PATHES = dict(
    resnet50=os.path.join(filedir, '../models/model.best.resnet50.hdf5'),
    from_scratch=os.path.join(filedir, '../models/model.best.resnet50.hdf5'),
    xception=os.path.join(filedir, '../models/model.best.resnet50.hdf5'),
    vgg16=os.path.join(filedir, '../models/model.best.resnet50.hdf5'),
)

__model = None

__ResNet50_model = ResNet50(weights='imagenet')

# --------------------------------------------------------------------------------------------------
# FUNCTIONS AND CLASSES
# --------------------------------------------------------------------------------------------------


def face_detector(img_path):
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier(FACE_CLASSIFIER_PTH)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    img = preprocess_image.preprocess_resnet50(img_path)
    # get index of maximum from prediction vector (most likely class)
    resnet50pred = np.argmax(__ResNet50_model.predict(img))
    # dog-classes are in imagenet between index 151 to 268
    return (resnet50pred <= 268) & (resnet50pred >= 151)


def predict_dog_breed(img_pth, model='resnet50'):
    """
    """
    if not __model:
        __model = DogPredictor(model)
    __model.change_type(model)
    return __model.predict_name(img_pth)


def classify_breed(imgpth):
    # preprocess img
    if dog_detector(imgpth):
        breed = predict_dog_breed(imgpth)
        print('Aawww!!! What a cutie! If that is not a %s!' % breed)
    elif face_detector(imgpth):
        breed = predict_dog_breed(imgpth)
        print('You look like a %s! Are you sure you are a human and not a dog??' % breed)
    else:
        print('What is that?! Neither a dog nor a human - I am sure!')


class DogPredictor:

    with open(DOG_CLASSES_PTH, 'r') as fin:
        __dog_names = json.load(fin)

    def __init__(self, type='from_scratch'):
        self.model = None
        self.tf_model = None
        self.type = None
        self.change_type(type)

    def change_type(self, type):
        if self.type == type:
            return

        if type == 'restnet50':
            self.tf_model = Resnet50()
        elif type == 'from_scratch':
            self.tf_model = NullTransfer()
        elif type == 'xception':
            self.tf_model = Xception()
        elif type == 'vgg16':
            self.tf_model = VGG16()
        else:
            raise ValueError
        load_model(MODEL_PATHES.get(type))
        self.type = type

    def load_model(self, pth):
        self.model = load_model(pth)

    def predict_props(self, inp):
        tf = self.tf_model.transfer(inp, image_path=True)
        return self.model.predict(tf)

    def predict_name(self, inp):
        props = self.predict_props(inp)
        maxidx = np.argmax(props)
        return DogPredictor.__dog_names[maxidx]


class KnowledgeTransfer(ABC):

    @property
    @abstractmethod
    def _model(self):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image_path):
        pass

    def transfer(self, inp, image_path=False):
        if image_path:
            inp = self.preprocess(inp)
        return self._model.predict(inp)


class VGG16(KnowledgeTransfer):
    def __init__(self):
        from keras.applications.vgg16 import VGG16 as VGG16Local
        self._model = VGG16Local(weights='imagenet', include_top=False)

    def preprocess(self, image_path):
        return preprocess_image.preprocess_vgg16(image_path)


class Resnet50(KnowledgeTransfer):
    def __init__(self):
        from keras.applications.resnet50 import ResNet50 as ResNetLocal
        self._model = ResNetLocal(weights='imagenet', include_top=False)

    def preprocess(self, image_path):
        return preprocess_image.preprocess_resnet50(image_path)


class Xception(KnowledgeTransfer):
    def __init__(self):
        from keras.applications.xception import Xception as XceptionLocal
        self._model = XceptionLocal(weights='imagenet', include_top=False)

    def preprocess(self, image_path):
        return preprocess_image.preprocess_xception(image_path, scale=True)


class NullModel:
    def predict(self, x):
        return x


class NullTransfer(KnowledgeTransfer):
    def __init__(self):
        self._model = NullModel()

    def preprocess(self, image_path):
        return preprocess_image.path_to_tensor(image_path)
