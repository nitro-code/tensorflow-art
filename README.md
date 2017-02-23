Deep Neural Network Artwork Classifier
--------------------------------------

This is an artwork classifier, which classifies paintings of more than 200 artists such as Claude Monet and Pablo Picasso.
A running demo is deployed at [Heroku](http://art-dnn.herokuapp.com).

The classifier is based on a deep neural network (DNN) implemented and trained with [Tensorflow](https://www.tensorflow.org).
The model architecture is constituted by a multi-layer [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network),
which is followed by a fully connected layer and completed with a [Softmax Readout Layer](https://en.wikipedia.org/wiki/Softmax_function).

Training was done based on a [Cross-Entropy](https://en.wikipedia.org/wiki/Cross_entropy) error function.


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
$ wget -r --level=3 --accept=jpg,jpeg
$ find . -name 's*' -exec rm {} \;

$ python preprocess.py
$ python train.py
```
