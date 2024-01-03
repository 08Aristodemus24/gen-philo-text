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
from modelling.utilities.preprocessors import decode_predictions, map_value_to_index, preprocess
from modelling.utilities.loaders import load_lookup_table, load_hyper_params

import tensorflow as tf

# # configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5000", "https://gen-philo-text.vercel.app", "https://gen-philo-text.onrender.com"])

# global variables
vocab = None
char_to_idx = None
idx_to_char = None
hyper_params = None
model = None


# functions to load miscellaneous variables, hyperparameters, and the model itself
def load_misc():
    """
    loads miscellaneous variables to be used by the model
    """
    global vocab 
    vocab = load_lookup_table('./modelling/final/misc/char_to_idx')

    global char_to_idx 
    char_to_idx = map_value_to_index(vocab)

    global idx_to_char
    idx_to_char = map_value_to_index(vocab, inverted=True)

    global hyper_params
    hyper_params = load_hyper_params('./modelling/final/misc/hyper_params.json')

def load_model():
    """
    prepares and loads sample input and custom model in
    order to use trained weights/parameters/coefficients
    """

    # declare sample input in order ot 
    # use load_weights() method
    sample_input = tf.random.uniform(shape=(1, hyper_params['T_x']), minval=0, maxval=hyper_params['n_unique'] - 1, dtype=tf.int32)
    sample_h = tf.zeros(shape=(1, hyper_params['n_a']))
    sample_c = tf.zeros(shape=(1, hyper_params['n_a']))

    # recreate model architecture
    global model
    model = GenPhiloTextA(
        emb_dim=hyper_params['emb_dim'],
        n_a=hyper_params['n_a'],
        n_unique=hyper_params['n_unique'],
        dense_layers_dims=hyper_params['dense_layers_dims'] + [hyper_params['n_unique']],
        lambda_=hyper_params['lambda_'],
        drop_prob=hyper_params['drop_prob'],
        normalize=hyper_params['normalize'])
    
    # call model on sample input before loading weights
    model(sample_input)

    # load weights
    model.load_weights('./modelling/final/weights/notes_gen_philo_text_a_100_3.0299.h5')

load_misc()
load_model()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    raw_data = request.json
    prompt = [preprocess(raw_data['prompt'])]
    temperature = float(raw_data['temperature'])
    T_x = int(raw_data['sequence_length'])
    print(raw_data)

    pred_ids = generate(model, prompts=prompt, char_to_idx=char_to_idx, T_x=T_x, temperature=temperature)
    decoded_ids = decode_predictions(pred_ids, idx_to_char=idx_to_char)

    return jsonify({'message': decoded_ids})

@app.errorhandler(404)
def page_not_found(error):
    print(error)
    return 'This page does not exist', 404



