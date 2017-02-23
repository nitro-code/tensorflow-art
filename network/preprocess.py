import os
from PIL import Image, ImageOps
import cStringIO
import tensorflow as tf


CROP_LEFT = 25
CROP_RIGHT = 25
CROP_BOTTOM = 25
CROP_TOP = 25

SOURCE_DIR = "/home/models/art/original"
TARGET_DIR = "/home/models/art/preprocessed"

TRAIN_EXTENSION = "train"
VALIDATION_EXTENSION = "validation"

WIDTH = 512
HEIGHT = 512
VALIDATION_MOD = 25


def process_artist(artist_dir, artist_number):
  artist_path = os.path.join(SOURCE_DIR, artist_dir)
  image_files = os.listdir(artist_path)

  if not image_files:
    return False
  else:
    os.mkdir(os.path.join(TARGET_DIR, artist_dir))
    image_number = 0

    for image_file in image_files:
      image_path = os.path.join(artist_path, image_file)

      if os.path.isfile(image_path) and image_path.lower().endswith(("jpg", "jpeg")):
        if image_number > 0 and image_number % VALIDATION_MOD == 0:
          extension = VALIDATION_EXTENSION
        else:
          extension = TRAIN_EXTENSION

        write_artist_tfrecord(artist_number, artist_dir, image_number, image_path, extension)
        image_number += 1

    return True

def write_artist_tfrecord(artist_number, artist_dir, image_number, image_path, extension):
  filename = os.path.basename(image_path)
  output_file = os.path.join(TARGET_DIR, '{}_{}.{}'.format(artist_dir, image_number, extension))

  print "writing tfrecord for artist {}, label {} and image {} into {}".format(artist_dir, artist_number, image_number, output_file)

  writer = tf.python_io.TFRecordWriter(output_file)
  image_buffer, height, width = process_image(image_path)
  example = convert_to_example(filename, image_buffer, artist_number, filename, height, width)
  writer.write(example.SerializeToString())

  writer.close()

def process_image(image_path):
  image = Image.open(image_path)
  image = ImageOps.autocontrast(image, 0.1)
  image = image.crop((CROP_LEFT, CROP_TOP, image.width - CROP_RIGHT, image.height - CROP_BOTTOM))
  image = image.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

  image_buffer = cStringIO.StringIO()
  image.save(image_buffer, "JPEG")
  return image_buffer, image.height, image.width

def convert_to_example(filename, image_buffer, label, text, height, width):
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'
  image_data = image_buffer.getvalue()

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/channels': _int64_feature(channels),
      'image/label': _int64_feature(label),
      'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data))}))
  return example

def _int64_feature(value):
  if not isinstance(value, list): value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


dirs = os.listdir(SOURCE_DIR)
dirs.sort()

artists = []
artist_number = 0

for artist_dir in dirs:
  result = process_artist(artist_dir, artist_number)

  if result:
    artists.append(artist_dir)
    artist_number += 1

print artists
