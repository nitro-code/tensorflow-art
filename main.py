#!/usr/bin/env python

from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
import json
import urllib
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from network.model import model_fn, decode_jpeg, resize_image, ARTISTS


CHECKPOINT_FILE = "checkpoint"
MODEL_CHECKPOINT_FILE = "model.ckpt.data-00000-of-00001"
MODEL_INDEX_FILE = "model.ckpt.index"
MODEL_META_FILE = "model.ckpt.meta"

CHECKPOINTS_DIR = './checkpoints'

IMAGES = ['static/img/Monet-Jardin_a_Sainte-Adresse.jpg', 'static/img/Picasso-Figure_dans_un_Fauteuil.jpg']


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return super(NumpyEncoder, self).default(obj)

def decode_predictions(predictions, top=3):
  results = []
  predictions = list(predictions)

  for prediction in predictions:
    probabilities = prediction['probabilities']
    classes = prediction['classes']
    result = [{'description' : ARTISTS[classes], 'probability' : probabilities[classes]}]
    results.append(result)

  return results

def classify(file):
  with tf.Session() as sess:
    input_fn_predict = lambda: resize_image(decode_jpeg(file.read()))
    predictions = classifier.predict(input_fn=input_fn_predict)
    result = decode_predictions(predictions)
    return json.dumps(result, cls=NumpyEncoder)


app = Flask(__name__)
CORS(app)

classifier = learn.Estimator(model_fn=model_fn, model_dir=CHECKPOINTS_DIR)


@app.route('/')
def index():
  return app.send_static_file('index.html')

@app.route('/js/<path:path>')
def send_js(path):
  return send_from_directory('js', path)

@app.route('/css/<path:path>')
def send_css(path):
  return send_from_directory('css', path)

@app.route('/api/v1/classify', methods=['GET'])
def classifyOnGet():
  id = int(request.args.get('id'))
  file_path = IMAGES[id]
  file = open(file_path, "r")
  return classify(file)

@app.route('/api/v1/classify', methods=['POST'])
def classifyOnPost():
  file = request.files['file']
  if file:
    return classify(file)
