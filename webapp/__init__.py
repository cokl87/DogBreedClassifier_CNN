# -*- coding: utf-8 -*-

"""
__init__.py

created: 11:50 - 26.08.20
author: kornel
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os.path

# configure tensorflow for limiting memory used
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# create app object
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
# app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

db = SQLAlchemy(app)
from webapp.models import Query, Probabilities

# import routes
from webapp import routes
