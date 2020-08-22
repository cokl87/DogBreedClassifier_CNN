# DogBreedClassifier_CNN

### Table of Contents
1. [Project Motivation and Description](#project-motivation-and-description)
2. [Installation](#installation)
3. [File Descriptions](#file-descriptions)
4. [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)


## Project Motivation and Description
< Coming Soon >

## Installation
For the project python3.8 was used. For getting the pipelines running, install the required packages listed in [requirements.txt](./requirements.txt).

### Local installation for running the webapp on your local machine
1. `git clone https://github.com/cokl87/jamdocs project-name`
2. `cd project-name`
3. `virtualenv venv`
4. `source venv/bin/activate`
5. `pip install -r requirements.txt`
6. `python webappp/run.py`

For running the web-app you might have to configure the server-settings defined in the constants-section of run.py:
```
DEBUG = False
HOST = '0.0.0.0'
PORT = 3001
```


## File Descriptions
One can find 4 directories in the project.
#### webapp:
contains the files for the webapp. The run.py runs the flask-app and defines the views. In templates you can find the HTML-templates. Static content for the app can be found in static-directory.
#### source:
* [apply_cnn.py](./source/apply_cnn.py) - functions and classes for determination of humans/dogs and making predictions about dog-breeds
* [preprocess_image.py](./source/preprocess_image.py) - functions for preprocessing images before feeding them into CNNs
* [dog_classes_en.json](./source/dog_classes_en.json) - json file containing list of english dogbreed-names ordered by output classes of CNNs. 

<!--
* [log_config.py](./source/log_config.py) - includes function for configuration of logging module
* [logging.json](./source/logging.json) - json file where loggers, handlers and formatters are defined
-->

#### models:
The trained keras-models (CNNs):
* model.best.from_scratch.hdf5: stand alone model without transfer learning. Expects scaled RGB-arrays of shape (X, 224, 224, 3) as input (you might use preprocess_image.paths_to_tensor with kwarg 'scale=True'.
* model.best.resnet50.hdf5: model uses knowledge from on imagenet trained resnet-model. (You might use output of apply_cnn.Resnet50.transfer as input)
* model.best.vgg16.hdf5: model uses knowledge from in imagenet trained vgg16-model. (You might use output of apply_cnn.VGG16.transfer as input)
#### haarcascade:
Contains Intel's haarcascade-classifier used for detection of human-faces in image. Consider it's license.


## Licensing, Authors, Acknowledgements
The project was part of the [Udacity's DataScientist program]('https://www.udacity.com/course/data-scientist-nanodegree--nd025'). 

The webapp uses for detection of human-faces Intel's haarcascade-classifier. Consider it's [license agreement](./haarcascade/haarcascade_frontalface_alt.xml).
The other parts of the project you may use as you like.
