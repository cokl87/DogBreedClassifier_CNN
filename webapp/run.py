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

# 3rd party imports
from flask import Flask, render_template

# project imports
from webapp.logging.log_config import config_logging


# --------------------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------------------

DEBUG = True
HOST = '0.0.0.0'
PORT = 3001

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():

    #figures = return_figures()

    # plot ids for the html id tag
    #ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    #figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    logger.info('upsi')

    return render_template('index.html', ids=())  #, ids=ids, figuresJSON=figuresJSON)


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
