# -*- coding: utf-8 -*-

"""
migrate.py

created: 17:08 - 01.09.20
author: kornel
"""

from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

from webapp import app, db

migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)


if __name__ == '__main__':
    manager.run()
