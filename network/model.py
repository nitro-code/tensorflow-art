import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.layers import conv2d, dropout, repeat
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import metric_key


ARTISTS = ['abrulloff', 'alekseev', 'almatadema', 'altdorfer', 'altman', 'angelico', 'barth', 'bellini', 'bonnard', 'borisov-musatov', 'bosch', 'braque', 'briullov', 'bruegel', 'burne-jones', 'campin', 'carpaccio', 'cezanne', 'chagall', 'cimabue', 'corot', 'correggio', 'cranach', 'cross', 'dali', 'daumier', 'david', 'davinci', 'degas', 'delacroix', 'dobuzhinsky', 'duccio', 'duchamp', 'durer', 'ernst', 'eyck', 'fabriano', 'fouquet', 'gagarin', 'gainsborough', 'gauguin', 'ghirlandao', 'giotto', 'gossaert', 'goya', 'grosz', 'hals', 'hay', 'heemskerch', 'hogarth', 'holbein', 'huntwh', 'ingres', 'kandinsky', 'klee', 'korovin', 'kramskoy', 'lebedev', 'levitan', 'lippi', 'lorenzetti', 'mantegna', 'martini', 'matisse', 'michelangelo', 'millais', 'miro', 'modigliani', 'monet', 'nesterov', 'orlovsky', 'ostroumova', 'patinir', 'perugino', 'picabia', 'picasso', 'piero', 'pissaro', 'pointillism', 'polenov', 'pollock', 'raphael', 'rembrandt', 'renoir', 'repin', 'rossetti', 'rubens', 'savrasov', 'schongauer', 'scorel', 'serov', 'seurat', 'shishkin', 'signac', 'signorelli', 'somov', 'surikov', 'teniers', 'tissot', 'turner', 'vandyck', 'vangogh', 'vasnetsov', 'venetsianov', 'vereshchagin', 'vermeer', 'vrubel', 'vuillard', 'watteau', 'weyden', 'whistler', 'zubov']

WIDTH = 512
HEIGHT = 512
CHANNELS = 3

LEARN_RATE = 1e-3


def decode_jpeg(image_jpeg):
  image = tf.image.decode_jpeg(image_jpeg, channels=CHANNELS)

  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image

def resize_image(image):
  image = tf.image.resize_images(image, [WIDTH, HEIGHT])
  return tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])

def model_fn(features, labels, mode):
  x = tf.reshape(features, [-1, HEIGHT, WIDTH, CHANNELS])
  tf.summary.image('images', x, 4)

  x = repeat(x, 2, conv2d, 32, 3, 2, scope='conv1', activation_fn=tf.nn.tanh)
  x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

  x = repeat(x, 2, conv2d, 64, 3, 2, scope='conv2', activation_fn=tf.nn.tanh)
  x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)

  x = tf.reshape(x, [-1, 8 * 8 * 64])

  with tf.name_scope('fc1'):
    x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.tanh)

  x = tf.layers.dropout(inputs=x, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  with tf.name_scope('fc2'):
    x = tf.layers.dense(inputs=x, units=len(ARTISTS))

  classes = tf.argmax(x, axis=1)
  probabilities = tf.nn.softmax(x, name="softmax_tensor")

  loss = None
  accuracy = None
  train_op = None

  if mode == learn.ModeKeys.TRAIN or mode == learn.ModeKeys.EVAL:
    #x = tf.Print(x, [x], message="logits: ", summarize=len(ARTISTS))
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=len(ARTISTS))
    #one_hot_labels = tf.Print(one_hot_labels, [one_hot_labels], message="one_hot_labels: ", summarize=len(ARTISTS))

    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=x)

    correct_prediction = tf.equal(classes, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.Print(accuracy, [accuracy], message="accuracy: ")

  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(), optimizer="Adam", learning_rate=LEARN_RATE)
    train_op = tf.Print(train_op, [train_op], message="train_op loss: ")
    tf.summary.scalar("accuracy", accuracy)

  with tf.name_scope('readout'):
    predictions = {
      "probabilities": probabilities,
      "classes": classes
    }

  with tf.name_scope('evaluation'):
    eval_metric_ops = {
      metric_key.MetricKey.ACCURACY: accuracy
    }

  return model_fn_lib.ModelFnOps(mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
