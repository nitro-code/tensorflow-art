import os
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops import control_flow_ops
from model import model_fn, decode_jpeg, resize_image, WIDTH, HEIGHT, CHANNELS


TRAIN_DIR = '/home/models/art/preprocessed'
CHECKPOINTS_DIR = './../checkpoints'

BATCH_SIZE = 128
STEPS_TRAIN = 100
STEPS_EVAL = 4
EPOCHS = 100000


def apply_with_random_selector(x, func, num_cases):
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0):
  with tf.name_scope('distort_color'):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=0.25)
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.05)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    elif color_ordering == 1:
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
      image = tf.image.random_brightness(image, max_delta=0.25)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.05)
    elif color_ordering == 2:
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
      image = tf.image.random_hue(image, max_delta=0.05)
      image = tf.image.random_brightness(image, max_delta=0.25)
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
    elif color_ordering == 3:
      image = tf.image.random_hue(image, max_delta=0.05)
      image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
      image = tf.image.random_brightness(image, max_delta=0.25)
    else:
      raise ValueError('color_ordering must be in [0, 3]')

    return tf.clip_by_value(image, 0.0, 1.0)

def distort_image(image, height, width):
  with tf.name_scope('distort_image'):
    image = apply_with_random_selector(image, lambda x, ordering: distort_color(x, ordering), num_cases=4)

    rand = tf.random_uniform([], maxval=0.2, dtype=tf.float32) - 0.1
    image = tf.contrib.image.transform(image, [1, rand, rand, rand, 1, rand, 0, 0])

    glimpse_min_height = tf.cast(tf.multiply(tf.cast(height, tf.float32), tf.constant(0.8)), tf.int64)
    glimpse_min_width = tf.cast(tf.multiply(tf.cast(width, tf.float32), tf.constant(0.8)), tf.int64)

    glimpse_height = tf.cast(tf.random_uniform([], minval=glimpse_min_height, maxval=height, dtype=tf.int64), tf.int32)
    glimpse_width = tf.cast(tf.random_uniform([], minval=glimpse_min_width, maxval=width, dtype=tf.int64), tf.int32)

    height_offset = tf.random_uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32)
    width_offset = tf.random_uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32)

    image = tf.image.extract_glimpse(
      tf.expand_dims(image, 0),
      size=[glimpse_height, glimpse_width],
      offsets=tf.expand_dims([height_offset, width_offset], 0),
      normalized=True,
      centered=True)

    return image

def input_batch(mode):
  with tf.name_scope('input_batch'):
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

    height = features['image/height']
    width = features['image/width']
    label = features['image/label']
    image_jpeg = features['image/encoded']

    image = decode_jpeg(image_jpeg)
    image = distort_image(image, height, width)
    image = resize_image(image)

    num_threads = 4
    min_after_dequeue = BATCH_SIZE * 4
    capacity = min_after_dequeue + num_threads * BATCH_SIZE

    image_batch, label_batch = tf.train.shuffle_batch(
      [image, label],
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
