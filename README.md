# dog breedClassifier_CNN

### Table of Contents
1. [Project Motivation and Description](#project-motivation-and-description)
    1. [Preprocessing](#preprocessing)
    2. [Metrics](#metrics)
    3. [Models applied](#methods-models-appplied)
    4. [Conclusion](#conclusion)
2. [Installation](#installation)
3. [File Descriptions](#file-descriptions)
4. [TODOs](#todos)
5. [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)


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

### Preprocessing
Before the images where passed to the Input-layer of the neural networks the size had to be transformed to a uniform
shape. Therefor the images were read and scaled to an array of 224x224 pixles with their RGB values which give a input 
array of size 224x224x3. If images with a very different height and width ratio compared to the training data, simple 
scaling could be problematic.
To avoid very large or very small weights the R,G and B values were normed to a value range between 0 and 1 by dividing 
them by 255.

The images were also used as is without any further input-augmentation like rotation, horizontal mirroring etc. in 
order to artificially 'increase' the amount of input data available for training the CNNs. By Input-augmentation the 
Neural Network has the change to learn on 'different' versions of one image different representations of dogs and 
abstract from them the patterns needed for correct classification of the images. By using this technique the accuracies 
at least of the model 'from scratch' could be further improved.
For the models using learning transfer the transfer-model's specific preprocessing steps had to be used in order to 
make correct predictions with these models. For further information on the preprocessing steps used by these models 
please refer to their corresponding documentation.

### Metrics
At the beginning of the project the available data was analysed. Images of all 133 dog-classes had been available. The 
dog classes were in all datasets (training, validation and testing) more or less equally represented (the whole dataset 
showed a mean of 62.79 images per class with a standard deviation of 14.80).
Because there was no majority classes dominating all other classes and therefor the accuracy paradoxon wasn't an issue,
the accuracy measure was used for the evaluation of the models:
<!-- $$ acc = \frac{# correct classified dogs}{# all images} $$ -->

![equation](http://latex.codecogs.com/gif.latex?acc%3D%5Cfrac%7B%5C%23%20Correct%20Classified%20Dogs%7D%7B%5C%23%20All%20Images%7D)

Other possibilities would have been to calculate f1 scores for every class:

![equation](http://latex.codecogs.com/gif.latex?f1%3D%5Cfrac%7B2%2Arecall%2Aprecicion%7D%7Brecall%2Bprecicion%7D)
<!-- $$f1 = \frac{2 * recall * precision}{recall + precision}$$. -->
Or to calculate a confusion matrix.

### Methods/ Models appplied

#### Without usage Transfer Learning
First a CNN without Transfer Learning was created from scratch:

Therefor the 224x224x3 matrix was passed to a sequence of convolutional layers of size 3x3, stride 1 and a 
relu-activation function and maxpooling-layers of size 2x2 and stride 2. The First can be seen as filters, which 
extract patterns from the image and increase the 'depth' of the matrix, the latter aggregate the image and decrease the 
size of the data in each sequence by half. After 4 sequences the remaining x/y extend was aggregated via a 
GlobalAveragePooling layer to transform the matrix to a 1d-vector containing the extracted 'patterns' which then was 
input to two Dense layers. 

The first Dense layer had a tangens-hyberbolicus activation function and a l2-regularization for avoiding overfitting 
by punishing large weights. The second Dense layer had 133 nodes with a softmax-activation function for mapping the 
final output to the dog-classes. The error function was determined by the categorical-crossentropy. This function was 
reduced by the adam-optimizer which includes a learningrate adaption as well as momentum. Adam showed faster 
descending then other optimizers like RMSprop.

This model achieved an accuracy of 28.0%. This value might seem low, but given 133 classes, random chance would result
in a average accuracy of 0.75%.

#### Usage of Transfer Learning
For further improving the achieved accuracies of the classification task, it was decided to make usage of transfer 
learning. Two models were picked: The VGG16 and the Resnet50 model which were trained on imagenet.

Because the amount of the available training data was rather small (8351 dog-images) and the pretrained models were
trained on data including similar classes (imagenet has 120 dog classes), transfer Learning was used by simply taking the
models, removing their end of dense layers and adding additional layers which are then trained without changeing the
original architecture or weights. In fact the old model was used for preprocessing of the training data. The so 
preprocessed or aggregated data was used for training the new 'top'-model.

##### VGG16:
The output of the cut-off VGG16-model has a shape of 7x7x512. This output was input to a MaxPooling2D-Layer of size 4x4
and stride 4 with padding for further aggregation. The so aggregated  data was flattened and fet to two Dense Layers
similar to the 'from scratch'-model with first tanh and l2-regularization and second softmax activation function.
This model achieved with 62.3% a significant higher accuracy then the model without transfer learning.

##### Resnet50:
The output of the Resnet50 model has a shape of 1x1x2048. A Flattening Layer was therefor not necessary. Unlike the
VGG16 model a composition of three Dense Layers of sizes 512, 256 and 133 showed best results. The first two layers used
 tanh as activation-function with l2-regularization and the last layer a softmax activation function for calculating the
 final probabilities. The model was able to achieve an accuracy of 80.6%

### Conclusion
Within this project three CNNs were trained for classifying 133 dog-breeds in images. For training of these CNNs only 
8351 labeled images were available. First a model from scratch was build and trained on these images. It already 
reached an accuracy of 28% which is surprisingly good given that random change would lead only to an accuracy of 0.75%.

For further improvements on the classification, transfer learning was used. Therefore, based on two pretrained CNNs,
individual on-top-models were build and trained on the data. This lead to an great increase of the accuracies: The model
 based on Vgg16 reached an accuracy on test data of 62.3% and the model build upon Resnet50 and accuracy of impressive 
 80.6%. This shows impressingly what can be achieved with little effort by just using previous work and knowledge in 
 form of pretrained models.

There are still many ways however, how even better results might be achieved within this project:

It became apparent that, given the relatively small amount of data, overfitting was an issue. When training the CNNs 
within this project overfitting was avoided by breaking the training process (before the last epoch was reached), once 
the validation-error was increasing in a series of three epochs. For most of the models this criteria was fullfilled 
after only a view epochs. From this point on further training would result in better classifications within the training
 image-set, but poor results for unseen images.

To avoid overfitting and therefore come to even better classifications several improvements could be done:

* As discussed in the previous sections image augmentation techniques were not used in this project but could be a very 
promising way to further improve training of the models. By image augmentation the 'amount' of images can be 
artificially increased (by rotation/mirroring etc.). Therefore misleading factors for the classification, like were in 
the image the dog is depicted can be decreased.
* as Alexis Cook discusses in this [article](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)
Some of the models used for transfer learning (e.g. VGG16) does not use Dropout-Layers in order to decrease the chance 
of overfitting. DO-Layers could be added to their architecture before retraining their weights. Retraining of their
weights would however require more training-data (images).
* In this project a Max-Pooling Layer + Flattening Layer was added in the VGG16-Transfer-model. One way to improve it, 
would be to replace the MaxPooling and Flatteing Layers by a GlobalAveralgePooling Layer, as these layers reduce 
overfitting as pointed out in the [first comment](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)
 of Alexis Cook article. (The GAP-Layer makes the CNN invariant towards on which area the dog is within the image)
* Other pretrained models like Xception could be assessed for Transfer Learning in this classification task.

On more way to decrease overfitting and improve the training of the models is obvioulsly to gather more labeled data 
available for training. With more training-data even other techniques for transfer learning could be assessed like
readjusting the weights of the pretrained-models, so that they can reflect better to the specifics in dogbreed-
classification.
More data would be especially helpful for classes difficult to identify (e.g. dog breeds with a high in-class 
variability or dog breeds with a small inter-class variability (similar dog-breeds)). If the amount of data per class
has a high variability however the choice of the metrics might have to be reassessed.

One way to gather more label data could be to use the images uploaded in the webapp. Therefore the images would have 
to be stored and somehow correctly labeled (either by the uploading user if he already knows the breed and is 
trustworthy, or by some specialists). With the new gathered data the weights of the CNNs could be then from time to time 
readjusted.

## Installation
### Local installation for running the webapp on your local machine
For this project python3.8 was used. To get the app started you first you have to get all project files and install 
all required packages:
1. `git clone https://github.com/cokl87/DogBreedClassifier_CNN project-name`
2. `cd project-name`
3. `virtualenv venv -p=3.8`
4. `source venv/bin/activate`
5. `pip install -r requirements.txt`

Then you need to create the database for the webapp with the required tables and schemas. Therefor you need postgres
isntalled. On Linux you can do that with `sudo apt-get install postgresql postgresql-contrib`. Then you can create the
 needed database by using the migrate.py module:
```
# add the needed environment-variables (you might want to edit the variables first)
source .env

# create the migrations directory in your project directory 
python migrate.py db init

# migrate the database
python migrate.py db migrate

# apply migrations to the database
python migrate.py db upgrade
```

Configure your App-settings in run_app.py:
```python
app.run(host='0.0.0.0', port=3001, debug=False)
```

And configure the table names and the path to your database in routes.py if you didn't use the default values
```python
DOG_IMAGES_ROOT = './webapp/static/img/dog_breeds'
```

Now you're good to go: `python run_app.py`


## File Descriptions
In the main directory you find several files. The most important ones are:
* [run_app.py](./run_app.py) - The main file for running the app
* [migrate.py](./migrate.py) - The script for creating the postgres-database tables and managing migrations
* [config.py](./config.py) - definitions of several configurations for the flask-app
* [.env](./.env) - definition of environment variables used by the app (e.g. db-location, configuration to use)

Furthermore you find 4 directories in the project:

#### models:
The trained keras-models (CNNs):
* model.best.from_scratch.hdf5: stand alone model without transfer learning. Expects scaled RGB-arrays of shape 
(X, 224, 224, 3) as input (you might use preprocess_image.paths_to_tensor with kwarg 'scale=True'.
* model.best.resnet50.hdf5: model uses knowledge from on imagenet trained resnet-model. (You might use output of 
apply_cnn.Resnet50.transfer as input)
* model.best.vgg16.hdf5: model uses knowledge from in imagenet trained vgg16-model. (You might use output of 
apply_cnn.VGG16.transfer as input)

#### source:
* [apply_cnn.py](./source/apply_cnn.py) - functions and classes for determination of humans/dogs and making predictions 
about dog-breeds
* [preprocess_image.py](./source/preprocess_image.py) - functions for preprocessing images before feeding them into CNNs
* [dog_classes_en.json](./source/dog_classes_en.json) - json file containing list of english dog breed-names ordered by 
output classes of CNNs. 
* [helperfuncs.py](./source/helperfuncs.py) - optional functions helping to create a sqlite3-database and other 
contents for the webapp. No actual part of the app/ project however and therefor optional.

<!--
* [log_config.py](./source/log_config.py) - includes function for configuration of logging module
* [logging.json](./source/logging.json) - json file where loggers, handlers and formatters are defined
-->

#### webapp:
In the webapp the files used for the webapp are included:
* [__init__.py](./webapp/__init__.py) - makes the directory to a python package; defines the app- and db-object
* [routes.py](./webapp/routes.py) - defines the routes of the webapp and contains some functions used only within these 
routes
* [models.py](./webapp/models.py) - definitions of postgres-tables used

Included are also three directories:
* templates: directory contains the html-templates used in the web-app
* static: directory contains content linked in the html-templates
* logging directory contains a module and configuration for the webapp's logging

#### haarcascade:
Contains Intel's haarcascade-classifier used for detection of human-faces in image. Consider it's license.


## TODOs:
* Images by the Users are accepted as is and not checked for right format, reasonable size etc. Functionality needs to
be implemented to check this user-input before further processing.
* When hosting the webapp on a free platform like heroku, the memory consumption is causing shutdowns. Optimizations
could be done to decrease the memory required
* Trained models can be further optimized e.g. by usage of Image-augmentation.
* The following bugs need to be fixed:
    * code.html-template. When importing the notebook.html via JavaScript, the css of bootstrap's navbar becomes altered
    * The file-name displayed after selection, is only shown if a previous classification had been made.


## Licensing, Authors, Acknowledgements
The project was part of the [Udacity's DataScientist program](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
The html-version of the [notebook](./webapp/static/notebook) included in the code.html template, is a cleaned version 
of a project of Udacity in which I designed and trained the CNNs which are deployed in this webapp.

The webapp uses, for the detection of human-faces, Intel's haarcascade-classifier. Consider it's 
[license agreement](./haarcascade/haarcascade_frontalface_alt.xml).
The other parts of the project you may use as you like.
