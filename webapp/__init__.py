# -*- coding: utf-8 -*-

"""
__init__.py

created: 11:50 - 26.08.20
author: kornel
"""

from flask import Flask

# create app object
app = Flask(__name__)
# app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
# app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

# import routes
from webapp import routes
