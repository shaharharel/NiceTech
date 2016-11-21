
import json
import numpy as np

def load_json_file(path):
    with open(path) as data_file:
        data = json.load(data_file)
    return data

def default(o):
    if isinstance(o, np.integer): return int(o)
    raise TypeError

def append_to_json(path,key,value):
        with open(path) as json_file:
            json_decoded = json.load(json_file)
        json_decoded[key] = value
        with open(path, 'w') as json_file:
            json.dump(json_decoded, json_file,default=default)

def write_to_json(path,data):
        with open(path, 'w') as outfile:
            json.dump(data, outfile,default=default)
