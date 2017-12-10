Deep Neural Network Artwork Classifier
--------------------------------------

This is an artwork classifier, which classifies the artist of a painting based on a deep neural network. The neural network has been trainined with paintings of well-known artists such as Claude Monet and Pablo Picasso. A running demo is deployed at [Heroku](http://tensorflow-art.herokuapp.com) (takes 10 seconds to load).

The deep neural network has been implemented and trained with [tensorflow](https://www.tensorflow.org). The model architecture is constituted by a multi-layer [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network), which is followed by a fully connected layer and completed with a [softmax readout layer](https://en.wikipedia.org/wiki/Softmax_function).

Training was done based on a [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) error function.

![screenshot1](http://assets.nitroventures.de/tensorflow-art/tensorflow-art.png)



Install Dependencies
====================

```
$ virtualenv -p python2.7 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```


Start Web Server
================

```
$ gunicorn main:app --log-file=- --timeout=600 --preload
```


Train Model
===========

```
$ python preprocess.py
$ python train.py
```
