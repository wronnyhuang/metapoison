# LEARNERS

This module implements common architectures into the 'meta' framework necessary for this project.
Most of the implementations are modifications of the **keras-applications** package
https://github.com/keras-team/keras-applications

> The module Keras Applications is the `applications` module of
the Keras deep learning library.
It provides model definitions and pre-trained weights for a number
of popular archictures, such as VGG16, ResNet50, Xception, MobileNet, and more.
>
> Read the documentation at: https://keras.io/applications/
>
> Keras Applications is compatible with Python 2.7-3.6
and is distributed under the MIT license.

All networks from keras-applications are modified to be usable with a dictionary of parameters that is given externally
when the network is evaluated and passed through all subfunctions.
To do this without major modifications custom layers are imported from '.modules.py'.

To replicate the modifications to keras-applications:
* Append `params=None` to every function call
* All layers modules with parameters take `params=params` as extra keyword argument
* The output of the network construction call is output, model: `x, model` instead of just the keras-model `model`
* Modify '__init__.py' to use the custom layers