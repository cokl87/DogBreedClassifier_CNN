# -*- coding: utf-8 -*-

"""
run_app.py

created: 09:37 - 26.08.20
author: kornel
"""

import logging
import os.path
from webapp.logging.log_config import config_logging

# configure logging
config_logging(os.path.join(os.path.dirname(__file__), './webapp/logging/logging.json'))
logger = logging.getLogger(__name__)

# define webapp
from webapp import app


# --------------------------------------------------------------------------------------------------
# IF NOT HOSTED (IMPORTED)
# --------------------------------------------------------------------------------------------------

def main():
    """ routine if not hosted (not imported) """
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()
