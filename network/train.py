import os
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops import control_flow_ops
from model import model_fn, decode_jpeg, WIDTH, HEIGHT, CHANNELS


TRAIN_DIR = '/home/models/art/preprocessed'
CHECKPOINTS_DIR = './../checkpoints'

BATCH_SIZE = 32
STEPS_TRAIN = 100
STEPS_EVAL = 10
EPOCHS = 100000


def apply_with_random_selector(x, func, num_cases):
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0):
  if color_ordering == 0:
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
  elif color_ordering == 1:
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    image = tf.image.random_hue(image, max_delta=0.05)
  elif color_ordering == 2:
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
  elif color_ordering == 3:
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
  else:
    raise ValueError('color_ordering must be in [0, 3]')

  return tf.clip_by_value(image, 0.0, 1.0)

def input_batch(mode):
  if mode == learn.ModeKeys.TRAIN:
    pattern = os.path.join(TRAIN_DIR, '*.train')
  else:
    pattern = os.path.join(TRAIN_DIR, '*.validation')

  filenames = tf.train.match_filenames_once(pattern)
  filename_queue = tf.train.string_input_producer(filenames, name="filename_queue")

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
        features={
          'image/height': tf.FixedLenFeature([], tf.int64),
          'image/width': tf.FixedLenFeature([], tf.int64),
          'image/channels': tf.FixedLenFeature([], tf.int64),
          'image/label': tf.FixedLenFeature([], tf.int64),
          'image/filename': tf.FixedLenFeature([], tf.string),
          'image/encoded': tf.FixedLenFeature([], tf.string)
      })

  label = features['image/label']
  image_jpeg = features['image/encoded']

  image = decode_jpeg(image_jpeg)
  distorted_image = apply_with_random_selector(image, lambda x, ordering: distort_color(x, ordering), num_cases=4)

  num_threads = 4
  min_after_dequeue = BATCH_SIZE * 4
  capacity = min_after_dequeue + num_threads * BATCH_SIZE

  image_batch, label_batch = tf.train.shuffle_batch(
      [distorted_image, label],
      num_threads=num_threads,
      batch_size=BATCH_SIZE,
      min_after_dequeue=min_after_dequeue,
      capacity=capacity)

  #label_batch = tf.Print(label_batch, [label_batch], message="labels: ", summarize=BATCH_SIZE)
  return image_batch, label_batch


#tf.logging.set_verbosity(tf.logging.INFO)

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  classifier = learn.Estimator(model_fn=model_fn, model_dir=CHECKPOINTS_DIR)

  input_fn_train = lambda: input_batch(learn.ModeKeys.TRAIN)
  input_fn_eval = lambda: input_batch(learn.ModeKeys.EVAL)

  for epoch in range(EPOCHS):
    classifier.fit(input_fn=input_fn_train, steps=STEPS_TRAIN)
    classifier.evaluate(input_fn=input_fn_eval, steps=STEPS_EVAL)
