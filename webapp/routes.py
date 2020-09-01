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
import json
from random import sample
from io import BytesIO
from base64 import b64encode

# 3rd party imports
from flask import render_template, request
from sqlalchemy import func
from keras.preprocessing.image import array_to_img
import numpy as np
import plotly
import plotly.graph_objects as go
from psycopg2.extensions import register_adapter, AsIs

# project imports
from webapp import app, db
import source.apply_cnn as apply_cnn
from source.preprocess_image import path_to_tensor
from webapp.models import Query, Probabilities

# configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# configure postgres adapter for numpy-ints
register_adapter(np.int32, AsIs)


# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

DOG_IMAGES_ROOT = './webapp/static/img/dog_breeds'


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


def write_class2db(name, species, dogbreed_int, dogbreed_name, model):
    """
    write entry to query-table.

    Parameters
    ----------
    name: str or None
        value for name-attribute
    species: int
        value for species-attribute
    dogbreed_int: int or None
        value for dogbreed_int-attribute
    dogbreed_name: str or None
        value for dogbreed_name-attribute
    model: str
        value for model-attribute

    Returns
    -------
    None
    """
    query = Query(
        name, species, dogbreed_int, dogbreed_name, model
    )
    db.session.add(query)
    db.session.commit()


def write_class_and_props2db(name, species, dogbreed_int, dogbreed_name, model, probs):
    """
    write entry into query and in property table.

    Parameters
    ----------
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
    query = Query(
        name, species, dogbreed_int, dogbreed_name, model
    )
    db.session.add(query)
    db.session.flush()

    probs = Probabilities(query.id, probs)
    db.session.add(probs)
    db.session.commit()


def create_dogbreed_histogram():
    """
    create a plotly bar chart ofclassified dog-breeds

    Returns
    -------
    str
        json-representation of chart
    """
    # SELECT dogbreed_name, COUNT(ALL) FROM Query WHERE species==1 GROUP BY dogbreed_int;
    hum_qres = db.session.query(Query.dogbreed_name, func.count(Query.dogbreed_name)).filter(
        Query.species == 1).group_by(Query.dogbreed_name).all()
    h_x, h_y = list(zip(*hum_qres)) if hum_qres else ((), ())

    # SELECT dogbreed_name, COUNT(ALL) FROM Query WHERE species==0 GROUP BY dogbreed_int;
    dog_qres = db.session.query(Query.dogbreed_name, func.count(Query.dogbreed_name)).filter(
        Query.species == 0).group_by(Query.dogbreed_name).all()
    d_x, d_y = list(zip(*dog_qres)) if dog_qres else ((), ())

    # create graph
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

    # SELECT species, COUNT(ALL) FROM Query GROUP BY species;
    spec_qres = db.session.query(Query.species, func.count(Query.species)).group_by(Query.species).all()
    species, counts = list(zip(*spec_qres)) if spec_qres else ((), ())

    label_mapper = {0: 'dogs', 1: 'humans', 2: 'other'}

    # create graph
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

    # SELECT model, COUNT(ALL) FROM Query GROUP BY model;
    model_qres = db.session.query(Query.model, func.count(Query.model)).group_by(Query.model).all()
    models, counts = list(zip(*model_qres)) if model_qres else ((), ())

    # create graph
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
                dog_images = ()
                top10 = ()
                dog_name = None
                write_class2db(name, species, None, dog_name, model)

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
    # SELECT name, dogbreed_name, model FROM Query WHERE name IS NOT NULL;
    usr_results = db.session.query(Query.name, Query.dogbreed_name, Query.model).filter(Query.name.isnot(None))

    return render_template(
        'statistics.html',
        classifiedBar=json_classified_bar,
        modelPie=json_model_pie,
        speciesPie=json_species_pie,
        results=usr_results
    )
