from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import requests
from requests.exceptions import ConnectionError
from urllib3.exceptions import MaxRetryError, NameResolutionError
import json
from datetime import datetime as dt

# ff. imports are for getting secret values from .env file
from pathlib import Path
import os

# import and load model architectures as well as decoder
from modelling.models.arcs import GenPhiloTextA, generate
from modelling.utilities.preprocessors import decode_predictions, map_value_to_index
from modelling.utilities.loaders import load_lookup_table

import tensorflow as tf

# # configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://127.0.0.1:5500", "http://127.0.0.1:5173", "https://project-alexander.vercel.app"])

# global variables
vocab = None
char_to_idx = None
idx_to_char = None
def load_misc():
    """
    loads miscellaneous variables to be used by the model
    """
    global vocab 
    vocab = load_lookup_table('../modelling/saved/misc/char_to_idx')

    global char_to_idx 
    char_to_idx = map_value_to_index(vocab)

    global idx_to_char
    idx_to_char = map_value_to_index(vocab, inverted=True)

load_misc()


@app.route('/')
def index():
    return f"{vocab}"

@app.errorhandler(404)
def page_not_found(error):
    print(error)
    return 'This page does not exist', 404



