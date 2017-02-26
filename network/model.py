import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.layers import conv2d, dropout, repeat
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import metric_key
from image import ARTISTS_LEN, WIDTH, HEIGHT, CHANNELS


LEARN_RATE = 1e-3

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
    x = tf.layers.dense(inputs=x, units=ARTISTS_LEN)

  classes = tf.argmax(input=x, axis=1)
  loss = None
  accuracy = None
  train_op = None

  if mode == learn.ModeKeys.TRAIN or mode == learn.ModeKeys.EVAL:
    x = tf.Print(x, [x], message="logits: ", summarize=ARTISTS_LEN)
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=ARTISTS_LEN)
    one_hot_labels = tf.Print(one_hot_labels, [one_hot_labels], message="one_hot_labels: ", summarize=ARTISTS_LEN)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=x)

    correct_prediction = tf.equal(classes, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(), optimizer="Adam", learning_rate=LEARN_RATE)
    train_op = tf.Print(train_op, [train_op], message="train_op loss: ")
    tf.summary.scalar("accuracy", accuracy)

  with tf.name_scope('readout'):
    predictions = {
      "probabilities": tf.nn.softmax(x, name="softmax_tensor"),
      "classes": classes
    }

  with tf.name_scope('evaluation'):
    eval_metric_ops = {
      metric_key.MetricKey.ACCURACY: accuracy
    }

  return model_fn_lib.ModelFnOps(mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
