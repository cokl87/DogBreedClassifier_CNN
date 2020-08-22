# -*- coding: utf-8 -*-

"""
run.py script for running flask-app
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# stamdard lib imports
import os.path
import logging
import glob
from random import sample
from io import BytesIO
from base64 import b64encode
import sys
sys.path.append('./..')

# 3rd party imports
from flask import Flask, render_template, request
from keras.preprocessing.image import array_to_img

# project imports
from webapp.logging.log_config import config_logging
import source.apply_cnn as apply_cnn
from source.preprocess_image import path_to_tensor


# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

DEBUG = True
HOST = '0.0.0.0'
PORT = 3001

# DOG_IMAGES_ROOT = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     './static/img/dog_breeds'
# )
DOG_IMAGES_ROOT = './static/img/dog_breeds'

# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

app = Flask(__name__)
# app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
# app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024


def tensor_to_image_string(array):
    """
    transforms a numpy-array representation of image to Bytes64-string for sending it to html.

    Parameters
    ----------
    array: np.array
        RGB array of shape (X, X, 3)

    Returns
    -------
    str
    """
    pil_img = array_to_img(array)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return b64encode(buff.getvalue()).decode("utf-8")


def get_dog_images(class_nr, number=3):
    """
    takes a dog-class number and returns n images of this dog

    Parameters
    ----------
    class_nr: int
    number: int

    Returns
    -------
    list
        list of img-pathes
    """
    # create filepat based on class-nr
    dirfilepat = '%03i.*/*.jpg' % class_nr
    # create path from static
    dirfilepath = os.path.join(DOG_IMAGES_ROOT, dirfilepat)
    # apply pattern and pick randomly n images
    dog_images = glob.glob(dirfilepath)
    return sample(dog_images, min(number, len(dog_images)))


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/classify-image', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        if request.files:
            img = request.files['image']
            model = request.form.get('model')
            logger.debug(model)

            # TODO: Danger! check image before further usage!!!!!!!

            # classify image
            bytes_image = BytesIO(img.read())
            try:
                species, breed = apply_cnn.classify_image(
                    bytes_image, model=model, breedname_only=False, bestbreed_only=False)
            except Exception as e:
                logger.error(e)
                # raise ValueError(e)
                return render_template('classify.html', image=True, image_err=True)

            # if breed was not determined (e.g. species 3)
            if breed is not None:
                dog_images = get_dog_images(breed['nr'][0])
                dog_name = breed['name'][0]
                top10 = breed[['name', 'p']][:10]
            else:
                # iterable needed. dog_name will be replaced in template via JavaScript, so None is fine.
                dog_images = ()
                top10 = ()
                dog_name = None

            # transform the image to how the CNN sees it
            bytes_image.seek(0)
            img_str = tensor_to_image_string(path_to_tensor(bytes_image)[0])
            return render_template(
                'classify.html',
                image=img_str,
                dog_images=dog_images,
                dog_name=dog_name,
                species=species,
                pred_array=top10,
            )

        else:
            logger.debug('No file uploaded')

    return render_template('classify.html')


@app.route('/statistics')
def stats():
    # TODO: change render template and implement view
    #figures = return_figures()

    # plot ids for the html id tag
    #ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    #figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)


    # logger.info('upsi')
    # TODO: does not work behind proxy
    # ip_address = request.remote_addr
    # logger.info(ip_address)

    return render_template('index.html')  #, ids=ids, figuresJSON=figuresJSON)


# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------------------


def main():
    """ main routine """
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == '__main__':
    # configure logging
    config_logging(
        os.path.join(os.path.dirname(__file__), './logging/logging.json')
    )
    logger = logging.getLogger(__name__)
    # call main routine
    logger.info('Starting web-app...')
    main()
