#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from os import listdir
from os.path import isfile, join
import pickle


directory = "data"

eyes = [f for f in listdir(directory) if isfile(join(directory, f)) and "eye" in f]

db = {}
for f in eyes:
    parts = f.split(".")
    handle = open(directory+"/"+f)

    # linea da ignorare  
    handle.readline()

    data = handle.readline().rstrip()
    coords = [ int(i) for i in data.split("\t")]
    db[parts[0]] = coords
    handle.close()


pickle.dump(db, open("eyes_db.pickle","wb") )
