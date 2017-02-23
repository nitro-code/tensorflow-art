from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
import json
import urllib
from PIL import Image, ImageOps
import StringIO
import numpy as np
from tensorflow.contrib import learn
from network.model import model_fn
from network.image import preprocess_image, ARTISTS, WIDTH, HEIGHT


MODEL_URL = "http://nitro.ai/assets/models/art"

CHECKPOINT_FILE = "checkpoint"
MODEL_CHECKPOINT_FILE = "model.ckpt.data-00000-of-00001"
MODEL_INDEX_FILE = "model.ckpt.index"
MODEL_META_FILE = "model.ckpt.meta"

CHECKPOINTS_DIR = './checkpoints'


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return super(NumpyEncoder, self).default(obj)

def assure_checkpoints_dir():
  if not os.path.isdir(CHECKPOINTS_DIR):
    os.mkdir(CHECKPOINTS_DIR)

def assure_model_file(model_file):
  model_file_path = os.path.join(CHECKPOINTS_DIR, model_file)

  if not os.path.isfile(model_file_path):
    url_opener = urllib.URLopener()
    print "downloading " + model_file
    url_opener.retrieve(MODEL_URL + "/" + model_file, model_file_path)

def assure_model():
  assure_checkpoints_dir()

  assure_model_file(CHECKPOINT_FILE)
  assure_model_file(MODEL_INDEX_FILE)
  assure_model_file(MODEL_META_FILE)
  assure_model_file(MODEL_CHECKPOINT_FILE)

def pil2array(pil_img):
  return np.array(pil_img.getdata(), np.float32).reshape(pil_img.size[1], pil_img.size[0], 3)

def decode_predictions(predictions, top=3):
  results = []
  predictions = list(predictions)

  for prediction in predictions:
    probabilities = prediction['probabilities']
    top_indices = probabilities.argsort()[-top:][::-1]
    result = [{'description' : ARTISTS[i], 'probability' : probabilities[i]} for i in top_indices]
    results.append(result)

  return results



app = Flask(__name__)
CORS(app)

assure_model()
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

@app.route('/api/v1/classify', methods=['POST'])
def classify():
  file = request.files['file']

  if file:
    image_jpeg = StringIO.StringIO(file.read())
    image = preprocess_image(image_jpeg)
    img_array = pil2array(image)
    img_array_batch = np.expand_dims(img_array, axis=0)

    input_fn = lambda: img_array_batch
    predictions = classifier.predict(input_fn=input_fn)
    result = decode_predictions(predictions)

    return json.dumps(result, cls=NumpyEncoder)
