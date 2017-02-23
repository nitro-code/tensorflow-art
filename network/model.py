import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.layers import conv2d, dropout, repeat
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


WIDTH = 512
HEIGHT = 512
CHANNELS = 3
LEARN_RATE = 1e-3

artists = ['abrulloff', 'aivazovsky', 'alekseev', 'almatadema', 'altdorfer', 'altman', 'angelico', 'antropov', 'arcimboldo', 'arp', 'baldung', 'barth', 'bazzile', 'bellini', 'bellotto', 'bogoliubov', 'bonington', 'bonnard', 'borisov-musatov', 'borovikovsky', 'bosch', 'botticelli', 'boucher', 'boudin', 'bouguereau', 'bouts', 'braque', 'briullov', 'broederlam', 'brouwer', 'bruegel', 'bruni', 'burne-jones', 'campin', 'canaletto', 'caravaggio', 'carpaccio', 'cassatt', 'castagno', 'cezanne', 'chagall', 'chardin', 'chirico', 'christinek', 'christus', 'cima', 'cimabue', 'clouet', 'cole', 'constable', 'copley', 'corot', 'correggio', 'cranach', 'cross', 'daumier', 'david', 'degas', 'delacroix', 'denis', 'dobuzhinsky', 'domenico', 'duccio', 'duchamp', 'durer', 'elgreco', 'ernst', 'eyck', 'fabriano', 'fantin-latour', 'fedotov', 'flemishstilife', 'fouquet', 'fragonard', 'gagarin', 'gainsborough', 'gauguin', 'gay', 'gerardavid', 'ghirlandao', 'giorgione', 'giotto', 'goes', 'gonzales', 'gossaert', 'goya', 'grabar', 'grosz', 'grunewald', 'hals', 'hay', 'heemskerch', 'hemessen', 'hogarth', 'holbein', 'hooch', 'huntwh', 'ingres', 'ivanov', 'jordaens', 'kandinsky', 'kaufman', 'khrutsky', 'kiprensky', 'klee', 'korovin', 'kramskoy', 'kuinji', 'kustodiyev', 'lampi', 'landseer', 'latour', 'lawrence', 'lebedev', 'leonardo', 'levitan', 'levitzky', 'leyster', 'lippi', 'lorenzetti', 'lotto', 'magritte', 'manet', 'mantegna', 'martini', 'martynov', 'massys', 'matisse', 'matveev', 'mayr', 'memling', 'mengs', 'michelangelo', 'millais', 'miro', 'modigliani', 'monet', 'morisot', 'munter', 'murillo', 'nesterov', 'nikitin', 'orlovsky', 'ostade', 'ostroumova', 'ostrovsky', 'paterssen', 'patinir', 'perov', 'perugino', 'pesne', 'petrov-vodkin', 'picabia', 'picasso', 'piero', 'pissaro', 'pointillism', 'polenov', 'poussin', 'quarenghi', 'ramsay', 'raphael', 'ray', 'rembrandt', 'renoir', 'repin', 'reynolds', 'ribera', 'rivera', 'rokotov', 'rossetti', 'rossika', 'rubens', 'ryabushkin', 'sadovnikov', 'sargent', 'savrasov', 'schongauer', 'scorel', 'semiradsky', 'semshchedrin', 'serov', 'serusier', 'seurat', 'shchedrin', 'shibanov', 'shishkin', 'signac', 'signorelli', 'sisley', 'snyders', 'somov', 'soutine', 'steen', 'stubbs', 'surikov', 'tanguy', 'tchistyakov', 'teniers', 'terborch', 'tissot', 'titian', 'tolstoy', 'tropinin', 'turner', 'uccello', 'valckenborch', 'valloton', 'vandyck', 'vangogh', 'vasilyev', 'vasnetsov', 'velazquez', 'venetsianov', 'vereshchagin', 'vermeer', 'verocchio', 'vigeelebrun', 'vishnyakov', 'vorobiev', 'vos', 'vrubel', 'vuillard', 'watteau', 'weyden', 'whistler', 'wilkie', 'winterhalter', 'witz', 'zubov', 'zurbaran']
artists_len = len(artists)


def model_fn(features, labels, mode):
  x = tf.reshape(features, [-1, HEIGHT, WIDTH, CHANNELS])
  tf.summary.image('images', x, 4)

  x = repeat(x, 2, conv2d, 16, [3, 3], [2, 2], scope='conv1')
  x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

  x = repeat(x, 2, conv2d, 32, [3, 3], [2, 2], scope='conv2')
  x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

  x = tf.reshape(x, [-1, 8 * 8 * 32])

  with tf.name_scope('fc'):
    x = tf.layers.dense(inputs=x, units=artists_len, activation=tf.nn.relu)

  loss = None
  train_op = None

  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(labels, depth=artists_len)
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
