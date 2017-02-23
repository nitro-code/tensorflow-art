import os
import tensorflow as tf
from tensorflow.contrib import learn
from model import model_fn, WIDTH, HEIGHT, CHANNELS


TRAIN_DIR = '/home/models/art/preprocessed'
CHECKPOINTS_DIR = './../checkpoints'

BATCH_SIZE = 128
STEPS_TRAIN = 200
STEPS_EVAL = 10
EPOCHS = 100000


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

  image = tf.image.decode_jpeg(features['image/encoded'], channels=CHANNELS)
  float_image = tf.cast(image, tf.float32)
  float_image = tf.reshape(float_image, [HEIGHT, WIDTH, CHANNELS])

  num_threads = 10
  min_after_dequeue = BATCH_SIZE * 10
  capacity = min_after_dequeue + num_threads * BATCH_SIZE

  image_batch, label_batch = tf.train.shuffle_batch(
      [float_image, label],
      num_threads=num_threads,
      batch_size=BATCH_SIZE,
      min_after_dequeue=min_after_dequeue,
      capacity=capacity)

  label_batch = tf.Print(label_batch, [label_batch], message="labels: ", summarize=BATCH_SIZE)
  return image_batch, label_batch


with tf.Session() as sess:
  classifier = learn.Estimator(model_fn=model_fn, model_dir=CHECKPOINTS_DIR)

  input_fn_train = lambda: input_batch(learn.ModeKeys.TRAIN)
  input_fn_eval = lambda: input_batch(learn.ModeKeys.EVAL)

  for epoch in range(EPOCHS):
    classifier.fit(input_fn=input_fn_train, steps=STEPS_TRAIN)
    print classifier.evaluate(input_fn=input_fn_eval, steps=STEPS_EVAL)
