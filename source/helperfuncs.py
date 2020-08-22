# -*- coding: utf-8 -*-

"""
helperfuncs.py

created: 19:10 - 21.08.20
author: kornel
"""

# --------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------

# stamdard lib imports
import logging
import os.path
import sqlite3

# 3rd party imports
from keras.preprocessing.image import array_to_img

# project imports
from source.preprocess_image import path_to_tensor

# --------------------------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------------------------


def process_image(img_pth):
    """ use processing used also by CNNs to decrease img-size """
    arr = path_to_tensor(img_pth)
    return array_to_img(arr[0])


def copy_img_dir(from_pth, to_pth):
    """ copy dir-structure with !only images! to other dir and process all images found in tree """
    # check if it is a dir
    assert os.path.isdir(from_pth)
    assert os.path.isdir(to_pth)
    for root, dirs, files in os.walk(from_pth):
        rel_root = os.path.relpath(root, from_pth)
        from_to_pth = os.path.join(to_pth, rel_root)

        for d in dirs:
            os.mkdir(os.path.join(from_to_pth, d))

        for imgpth in files:
            process_image(os.path.join(root, imgpth)).save(os.path.join(from_to_pth, imgpth))


def count_dogbreed_images(img_dir):
    """ checker for seeing how many images of each dog-breed are available """
    ndogs = []
    for root, dirs, files in os.walk(img_dir):
        if files:
            ndogs.append(len(files))
    print(min(ndogs))
    print(max(ndogs))
    return ndogs


def create_new_maintable(db_pth, tablename, overwrite=False):
    """ create main database-table for webapp - careful with overwrite-arg """
    con = sqlite3.connect(db_pth)
    cur = con.cursor()
    if overwrite:
        cur.execute("DROP TABLE IF EXISTS %s" % tablename)
    cur.execute("CREATE TABLE %s (id INT PRIMARY KEY NOT NULL, name TEXT, species INT NOT NULL, dogbreed INT NOT NULL)"
                % tablename)
    con.commit()
    con.close()


def create_new_probtable(db_pth, tablename, overwrite=False):
    """ create secondary-table for webapp - careful with overwrite-arg """
    con = sqlite3.connect(db_pth)
    cur = con.cursor()
    if overwrite:
        cur.execute("DROP TABLE IF EXISTS %s" % tablename)
    ', p'
    cur.execute("CREATE TABLE %s (id INT PRIMARY KEY NOT NULL, %s)" % (
        tablename, ', '.join(('p_%03i REAL NOT NULL' % idx for idx in range(1, 134)))))
    con.commit()
    con.close()
