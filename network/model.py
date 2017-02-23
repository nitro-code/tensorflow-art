import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.layers import conv2d, dropout, repeat
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from image import ARTISTS_LEN, WIDTH, HEIGHT, CHANNELS


LEARN_RATE = 1e-3

def model_fn(features, labels, mode):
  x = tf.reshape(features, [-1, HEIGHT, WIDTH, CHANNELS])
  tf.summary.image('images', x, 4)

  x = repeat(x, 2, conv2d, 16, [3, 3], [2, 2], scope='conv1')
  x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

  x = repeat(x, 2, conv2d, 32, [3, 3], [2, 2], scope='conv2')
  x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

  x = tf.reshape(x, [-1, 8 * 8 * 32])

  with tf.name_scope('fc'):
    x = tf.layers.dense(inputs=x, units=ARTISTS_LEN, activation=tf.nn.relu)

  loss = None
  train_op = None

  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(labels, depth=ARTISTS_LEN)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=x)
    train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(), optimizer="Adam", learning_rate=LEARN_RATE)

  with tf.name_scope('readout'):
    predictions = {
      "classes": tf.argmax(input=x, axis=1),
      "probabilities": tf.nn.softmax(x, name="softmax_tensor")
    }

  with tf.name_scope('evaluation'):
    eval_metric_ops = {
      "val_loss": loss
    }

  return model_fn_lib.ModelFnOps(mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
