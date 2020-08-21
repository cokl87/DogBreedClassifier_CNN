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
from io import BytesIO
from base64 import b64encode

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

            # TODO: check image before further usage!!!!!!!

            bytes_image = BytesIO(img.read())
            apply_cnn.classify_image(bytes_image, model=model)

            bytes_image.seek(0)
            # transform the image to how the CNN sees it
            img_str = tensor_to_image_string(path_to_tensor(bytes_image)[0])
            return render_template('classify.html', image=img_str)
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
