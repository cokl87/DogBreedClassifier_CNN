# -*- coding: utf-8 -*-

"""
routes.py script for running flask-app
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

# 3rd party imports
from flask import render_template, request
from keras.preprocessing.image import array_to_img
import numpy as np
import plotly
import plotly.graph_objects as go

# project imports
from webapp import app
import source.apply_cnn as apply_cnn
from source.preprocess_image import path_to_tensor

# configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# configure sqlite3 adapter for numpy-ints
sqlite3.register_adapter(np.int32, lambda x: int(x))


# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

DOG_IMAGES_ROOT = './webapp/static/img/dog_breeds'
DB_PTH = './webapp/classifications.sqlite3'
PROB_TABLE = 'probs'
CLASS_TABLE = 'queries'


# --------------------------------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------------------------------


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
    # apply pattern
    dog_images = glob.glob(dirfilepath)
    # make relative to this files dir and pick randomly n images
    dog_images = [os.path.relpath(dopi, os.path.dirname(__file__)) for dopi in dog_images]
    return sample(dog_images, min(number, len(dog_images)))


def connected(func):
    """
    wrapper-function for wrapping sql-queries to ensure db-connection and closing of the db.
    """
    @wraps(func)
    def decorated(*args, **kwargs):
        """
        Connect to Database and give cursor-object to wrapped function. Close DB after database commit.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        return value from wrapped function
        """
        con = sqlite3.connect(DB_PTH)
        cur = con.cursor()
        result = func(cur, *args, **kwargs)
        con.commit()
        con.close()
        return result
    return decorated


@connected
def write_class2db(cursor, name, species, dogbreed_int, dogbreed_name, model):
    """
    write entry to query-table.

    Parameters
    ----------
    cursor: cursor-obj
    name: str or None
        value for name-attribute
    species: int
        value for species-attribute
    dogbreed_int: int
        value for dogbreed_int-attribute
    dogbreed_name: str
        value for dogbreed_name-attribute
    model: str
        value for model-attribute

    Returns
    -------
    None
    """
    t = (name, species, dogbreed_int, dogbreed_name, model)
    # do not specify qid to let the sqlite3 engine increment the id by one
    sql_str = "INSERT INTO %s (name, species, dogbreed_int, dogbreed_name, model) VALUES (?,?,?,?,?);" % CLASS_TABLE
    cursor.execute(sql_str, t)


@connected
def write_props2db(cursor, querry_id, probs):
    """
    write entry to property-table.

    Parameters
    ----------
    cursor: cursor-obj
    querry_id: int
        key for querry-table
    probs: iterable
        iterable of length 133 with prediction probabilities for different classes

    Returns
    -------
    None
    """
    column_str = 'qid,' + ', '.join(('p_%03i' % idx for idx in range(1, 134)))
    sql_str = "INSERT INTO %s (%s) VALUES (%s);" % (PROB_TABLE, column_str, ', '.join(('?' for _ in range(0, 133+1))))
    t = (querry_id,) + tuple(probs)
    cursor.execute(sql_str, t)


@connected
def write_class_and_props2db(cursor, name, species, dogbreed_int, dogbreed_name, model, probs):
    """
    write entry into query and in property table.

    Parameters
    ----------
    cursor: cursor-obj
    name: str
    species: int
    dogbreed_int: int
    dogbreed_name: str
    model: str
    probs: iterable
        iterable with floats of length 133

    Returns
    -------
    None
    """
    write_class2db.__wrapped__(cursor, name, species, dogbreed_int, dogbreed_name, model)
    querry_id = next(cursor.execute("SELECT max(qid) FROM %s" % CLASS_TABLE))[0]
    write_props2db.__wrapped__(cursor, querry_id, probs)


@connected
def execute_querry(cursor, querry):
    """
    execute sql-querry str

    Parameters
    ----------
    cursor: cursor-obj
    querry: str
        sql-querry

    Returns
    -------
    result rows
    """
    return [row for row in cursor.execute(querry)]


def create_dogbreed_histogram():
    """
    create a plotly bar chart ofclassified dog-breeds

    Returns
    -------
    str
        json-representation of chart
    """
    hum_qres = execute_querry(
        "SELECT dogbreed_name, COUNT(ALL) FROM %s WHERE species==1 GROUP BY dogbreed_int;" % CLASS_TABLE)
    h_x, h_y = list(zip(*hum_qres)) if hum_qres else ((), ())
    dog_qres = execute_querry(
        "SELECT dogbreed_name, COUNT(ALL) FROM %s WHERE species==0 GROUP BY dogbreed_int;" % CLASS_TABLE)
    d_x, d_y = list(zip(*dog_qres)) if dog_qres else ((), ())

    graph = go.Figure()
    graph.add_trace(go.Bar(x=h_x, y=h_y, name='humans'))
    graph.add_trace(go.Bar(x=d_x, y=d_y, name='dogs'))
    graph.update_layout(dict(
        xaxis=dict(title='Dog Breed', tickangle=30),
        yaxis=dict(title='Number of classifications'),
        title='Number of Classifications per Dog Breed and Image-Type',
        autosize=True,
    ))
    return json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)


def create_species_pie():
    """
    create a plotly pie chart of species detected in images

    Returns
    -------
    str
        json-representation of chart
    """
    spec_qres = execute_querry("SELECT species, COUNT(ALL) FROM %s GROUP BY species;" % CLASS_TABLE)
    species, counts = list(zip(*spec_qres)) if spec_qres else ((), ())
    label_mapper = {0: 'dogs', 1: 'humans', 2: 'other'}
    graph = go.Figure()
    graph.add_trace(go.Pie(labels=[label_mapper.get(spec) for spec in species], values=counts))
    graph.update_layout(dict(
        title='Uploaded Image Content',
    ))
    return json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)


def create_models_pie():
    """
    create a plotly pie chart of models used in classifications

    Returns
    -------
    str
        json-representation of chart
    """
    model_qres = execute_querry('SELECT model, COUNT(ALL) FROM %s GROUP BY model;' % CLASS_TABLE)
    models, counts = list(zip(*model_qres)) if model_qres else ((), ())
    graph = go.Figure()
    graph.add_trace(go.Pie(labels=models, values=counts))
    graph.update_layout(dict(
        title='Model used for classification',
    ))
    return json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)


# --------------------------------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------------------------------


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/code')
def code():
    return render_template('code.html')


@app.route('/classify-image', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        if request.files:
            img = request.files['image']
            model = request.form.get('model', '')
            # if name is empty string or empty after removal of unwanted character None will be inserted for db
            name = request.form.get('name').translate({ord(x): '' for x in ';,()"\''}) or None
            logger.debug('user-name: %s, data-type: %s', name, type(name))

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
                    dog_name,
                    model,
                    breed[np.argsort(breed['nr'])]['p']
                )
            else:
                # iterable needed. dog_name will be replaced in template via JavaScript, so None is fine.
                write_class2db(name, species, None, None, model)
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
            logger.info('No file uploaded')

    return render_template('classify.html')


@app.route('/statistics')
def stats():
    # create plotly plot for classified dogbreeds (two groups - humans, dogs)
    json_classified_bar = create_dogbreed_histogram()

    # create plotly plot for models used
    json_model_pie = create_models_pie()

    # create plot for species detected in images
    json_species_pie = create_species_pie()

    # create result table: name - dogbreed-name - model
    usr_results = execute_querry('SELECT name, dogbreed_name, model FROM %s WHERE name IS NOT NULL;' % CLASS_TABLE)

    return render_template(
        'statistics.html',
        classifiedBar=json_classified_bar,
        modelPie=json_model_pie,
        speciesPie=json_species_pie,
        results=usr_results
    )
