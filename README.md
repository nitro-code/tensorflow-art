Deep Neural Network Artwork Classifier
--------------------------------------

This is an artwork classifier, which classifies paintings of more than 200 artists such as Claude Monet and Pablo Picasso.
A running demo is deployed at [Heroku](http://art-dnn.herokuapp.com).

The classifier is based on a deep neural network (DNN) implemented and trained with [tensorflow](https://www.tensorflow.org).
The model architecture is constituted by a multi-layer [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network),
which is followed by a fully connected layer and completed with a [softmax readout layer](https://en.wikipedia.org/wiki/Softmax_function).

Training was done based on a [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) error function.



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
