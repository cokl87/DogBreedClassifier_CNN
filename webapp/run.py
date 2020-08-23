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
import sqlite3
import json
from functools import wraps
from random import sample
from io import BytesIO
from base64 import b64encode
import sys
sys.path.append('./..')

# 3rd party imports
from flask import Flask, render_template, request
from keras.preprocessing.image import array_to_img
import numpy as np
import plotly
import plotly.graph_objects as go

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
DB_PTH = './classifications.sqlite3'
PROB_TABLE = 'probs'
CLASS_TABLE = 'queries'

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


def connected(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        con = sqlite3.connect(DB_PTH)
        cur = con.cursor()
        result = func(cur, *args, **kwargs)
        con.commit()
        con.close()
        return result
    return decorated


@connected
def write_class2db(cursor, name, species, dogbreed, model):
    t = (name, species, dogbreed, model)
    # do not specify qid to let the sqlite3 engine increment the id by one
    sql_str = "INSERT INTO %s (name, species, dogbreed, model) VALUES (?,?,?,?);" % CLASS_TABLE
    cursor.execute(sql_str, t)


@connected
def write_props2db(cursor, querry_id, probs):
    column_str = 'qid,' + ', '.join(('p_%03i' % idx for idx in range(1, 134)))
    sql_str = "INSERT INTO %s (%s) VALUES (%s);" % (PROB_TABLE, column_str, ', '.join(('?' for _ in range(0, 133+1))))
    t = (querry_id,) + tuple(probs)
    cursor.execute(sql_str, t)


@connected
def write_class_and_props2db(cursor, name, species, dogbreed, model, probs):
    write_class2db.__wrapped__(cursor, name, species, dogbreed, model)
    querry_id = next(cursor.execute("SELECT max(qid) FROM %s" % CLASS_TABLE))[0]
    write_props2db.__wrapped__(cursor, querry_id, probs)


@connected
def execute_querry(cursor, querry):
    return [row for row in cursor.execute(querry)]


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/classify-image', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        if request.files:
            img = request.files['image']
            model = request.form.get('model', '')
            # Not yet implemented but needed for DataBase (allowed to be None)
            name = request.form.get('name')
            if name is not None:
                name.translate({ord(x): '' for x in ';,()"\''})
            logger.debug(name)

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
                write_class_and_props2db(
                    name,
                    species,
                    breed['nr'][0],
                    model,
                    breed[np.argsort(breed['nr'])]['p']
                )
            else:
                # iterable needed. dog_name will be replaced in template via JavaScript, so None is fine.
                write_class2db(name, species, None, model)
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
    '''
    # get data from db.classification
    # create plotly plot for classified dogbreeds (two groups - humans, dogs)
    # humans
    hum_clss_breed, hum_clss_count = list(zip(
        *execute_querry("SELECT dogbreed, COUNT(ALL) FROM %s WHERE species==1 GROUP BY dogbreed;" % CLASS_TABLE)
    ))
    #dogs
    hdog_clss_breed, hdog_clss_count = list(zip(
        *execute_querry("SELECT dogbreed, COUNT(ALL) FROM %s WHERE species==0 GROUP BY dogbreed;" % CLASS_TABLE)
    ))

    # create plot for species
    execute_querry("SELECT species, COUNT(ALL) FROM %s GROUP BY species;" % CLASS_TABLE)

    execute_querry('SELECT model, COUNT(ALL) FROM %s GROUP BY model;' % CLASS_TABLE)



    logger.debug(human_clss)
    logger.debug(list(zip(*human_clss)))

    data = [
        go.Bar(
            x=[x for x in range(1, 134)],
            y=[list(zip(*human_clss))],
        )]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    '''

    # create result table: name - dogbreed-name - model
    graphJSON = create_dogbreed_histogram()
    logger.debug(graphJSON)

    usr_results = execute_querry('SELECT name, dogbreed, model FROM %s WHERE name IS NOT NULL;' % CLASS_TABLE)

    return render_template('statistics.html', graphJSON=graphJSON, results=usr_results)  #, ids=ids, figuresJSON=figuresJSON)


def create_dogbreed_histogram():

    hum_qres = execute_querry("SELECT dogbreed, COUNT(ALL) FROM %s WHERE species==1 GROUP BY dogbreed;" % CLASS_TABLE)
    h_x, h_y = list(zip(*hum_qres)) if hum_qres else ((), ())
    dog_qres = execute_querry("SELECT dogbreed, COUNT(ALL) FROM %s WHERE species==0 GROUP BY dogbreed;" % CLASS_TABLE)
    d_x, d_y = list(zip(*dog_qres)) if dog_qres else ((), ())

    graph = go.Figure()
    graph.add_trace(go.Bar(x=h_x, y=h_y, name='humans'))
    graph.add_trace(go.Bar(x=d_x, y=d_y, name='dogs'))
    graph.update_layout(dict(
        xaxis=dict(title='Dog Breed'),
        yaxis=dict(title='Number of classifications'),
        title='Number of Classifications per Dog Breed and Image-Type',
        autosize=True,
    ))

    return json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)


# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------------------


def main():
    """ main routine """
    sqlite3.register_adapter(np.int32, lambda x: int(x))
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
