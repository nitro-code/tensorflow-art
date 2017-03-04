import os
from PIL import Image
import tensorflow as tf
from model import decode_jpeg, WIDTH, HEIGHT, CHANNELS


SOURCE_DIR = "/home/models/art/original"
TARGET_DIR = "/home/models/art/preprocessed"

TRAIN_EXTENSION = "train"
VALIDATION_EXTENSION = "validation"
VALIDATION_MOD = 100
FILE_MIN_SIZE_KB = 100


def process_artist(artist_dir, artist_number):
  artist_path = os.path.join(SOURCE_DIR, artist_dir)
  image_files = os.listdir(artist_path)

  if not image_files:
    return False
  else:
    image_number = 0

    for image_file in image_files:
      image_path = os.path.join(artist_path, image_file)

      if os.path.isfile(image_path) and image_path.lower().endswith(("jpg", "jpeg")):
        filesize = os.path.getsize(image_path)

        if filesize > FILE_MIN_SIZE_KB * 1000:
          image = Image.open(image_path)
          width, height = image.size

          if width >= WIDTH and height >= HEIGHT:
            filename = os.path.basename(image_path)

            if image_number > 0 and image_number % VALIDATION_MOD == 0:
              extension = VALIDATION_EXTENSION
            else:
              extension = TRAIN_EXTENSION

            f = file(image_path)
            file_content = f.read()

            write_artist_tfrecord(artist_dir, artist_number, image_number, filename, extension, file_content, width, height)
            image_number += 1

    return image_number > 0

def write_artist_tfrecord(artist_dir, artist_number, image_number, filename, extension, file_content, width, height):
  output_path = os.path.join(TARGET_DIR, '{}_{}.{}'.format(artist_dir, image_number, extension))

  print "writing tfrecord for artist {}, label {} and image {} into {}".format(artist_dir, artist_number, image_number, output_path)
  writer = tf.python_io.TFRecordWriter(output_path)
  example = convert_to_example(filename, file_content, artist_number, width, height)
  writer.write(example.SerializeToString())
  writer.close()

def convert_to_example(filename, file_content, label, width, height):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/channels': _int64_feature(CHANNELS),
      'image/label': _int64_feature(label),
      'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(file_content))}))

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
