# dog breedClassifier_CNN

### Table of Contents
1. [Project Motivation and Description](#project-motivation-and-description)
2. [Installation](#installation)
3. [File Descriptions](#file-descriptions)
4. [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)


## Project Motivation and Description

### Project Definition
The goal of this project was to build a convolutional neural network (CNN) that classifies images of dogs, regarding the 
dog breed they depict. 8351 images of dogs were available for training and testing of the CNN. This is a rather small 
dataset for optimizing millions of weights needed within a CNN and to achieve good final classification results. 
Therefor it was intended to make use of Transfer Learning.

In Transfer Learning a pre-trained model is used, which is able to classify similar objects, and it's 
knowledge in form of the pretrained weights and the architecture is used as input to the final network-architecture. 

In this project models trained on Imagenet were used for Transfer Learning. Imagenet is a DataSet containing millions 
of images depicting 1000 different classes, including 120-dog-classes which is ideal for the project's goals.
Because of the small DataSet and the similarity between our classification goal and the classes used in training these
CNNs, the end of these pre-trained models were simply 'cut off' without readjusting their weights. The ends are replaced 
by an own set of Dense layers mapping to the final desired 133 dog-breed classes. In parts also additional convolutional 
layers were added. You can see the individual architectures of the final 'on top' models in the html export of the 
[jupyter notebook](./webapp/static/notebook/notebook.html).

The final trained models were deployed in this webapp where users can upload own images and see whether they contain a
dog or a human and which dog-breed it resembles the most.

### Conclusion
Until now three models were trained. One model without usage of transfer learning, one model built on top of the
VGG16 model and one build upon the Resnet50-model. The DataSet of the dog-images was subdivided in a training, 
validation and testing set. For measuring the final quality of the models, the accuracy on the testing-set was assessed.
* The model without TL ('from_scratch') achieved an accuracy of 28.0%
* The model built upon VGG16 an accuracy of 62.3%
* The model built upon Resnet50 an accuracy of 80.6%

The numbers might seem small, but given 133 classes, random choice would lead to an average accuracy of only
0.75%. I find it very astonishing that with a very small, relatively simple model, like I build myself from scratch 
and the limitations based on time, amount of reference data and computational power, accuracies of ~30% were 
achieved.


## Installation
### Local installation for running the webapp on your local machine
For this project python3.8 was used. To get the app started you first you have to get all project files and install 
all required packages:
1. `git clone https://github.com/cokl87/jamdocs project-name`
2. `cd project-name`
3. `virtualenv venv -p=3.8`
4. `source venv/bin/activate`
5. `pip install -r requirements.txt`

Then you need to create the database for the webapp with the required tables and schemas. You can create it using the 
helperfuncs.py module:
```
import source.helperfuncs as hf
hf.create_new_maintable(<Path2DB>, <tablename1>)
hf.create_new_probtable(<Path2DB>, <tablename2>)
```

Configure your App-settings in webapp/run.py:
```
DEBUG = True
HOST = '0.0.0.0'
PORT = 3001

DOG_IMAGES_ROOT = './static/img/dog_breeds'
DB_PTH = './classifications.sqlite3'
PROB_TABLE = 'probs'
CLASS_TABLE = 'queries'
```

And you're good to go: `python webappp/run.py`

## File Descriptions
One can find 4 directories in the project.
#### webapp:
contains the files for the webapp. The run.py runs the flask-app and defines the views.
* The templates directory contains the html-templates.
* The static directory contains content linked in the html-templates.
* the logging directory contains a module and configuration for webapp's logging.

In templates you can find the 
HTML-templates. Static content for the app can be found in static-directory.
#### source:
* [apply_cnn.py](./source/apply_cnn.py) - functions and classes for determination of humans/dogs and making predictions 
about dog-breeds
* [preprocess_image.py](./source/preprocess_image.py) - functions for preprocessing images before feeding them into CNNs
* [dog_classes_en.json](./source/dog_classes_en.json) - json file containing list of english dog breed-names ordered by 
output classes of CNNs. 
* [helperfuncs.py](./source/helperfuncs.py) - optional functions helping to create the database and other contents for
the webapp not actual part of the app/ project however and therefor optional.

<!--
* [log_config.py](./source/log_config.py) - includes function for configuration of logging module
* [logging.json](./source/logging.json) - json file where loggers, handlers and formatters are defined
-->

#### models:
The trained keras-models (CNNs):
* model.best.from_scratch.hdf5: stand alone model without transfer learning. Expects scaled RGB-arrays of shape 
(X, 224, 224, 3) as input (you might use preprocess_image.paths_to_tensor with kwarg 'scale=True'.
* model.best.resnet50.hdf5: model uses knowledge from on imagenet trained resnet-model. (You might use output of 
apply_cnn.Resnet50.transfer as input)
* model.best.vgg16.hdf5: model uses knowledge from in imagenet trained vgg16-model. (You might use output of 
apply_cnn.VGG16.transfer as input)


#### haarcascade:
Contains Intel's haarcascade-classifier used for detection of human-faces in image. Consider it's license.


## Licensing, Authors, Acknowledgements
The project was part of the [Udacity's DataScientist program](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
The html-version of the [notebook](./webapp/static/notebook) included in the code.html template, is a cleaned version 
of a project of Udacity in which I designed and trained the CNNs.

The webapp uses for detection of human-faces Intel's haarcascade-classifier. Consider it's 
[license agreement](./haarcascade/haarcascade_frontalface_alt.xml).
The other parts of the project you may use as you like.
