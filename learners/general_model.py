# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted TF Resnets from
# JOG
#
#
"""ResNet56 model for Keras adapted from tf.keras.applications.ResNet50.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import models

from data import tf_standardize

from . import keras_applications
from .modules import Dense

from collections import defaultdict

L2_WEIGHT_DECAY = 2e-4


class KerasModel(tf.keras.Model):
    """Generalized Sub-class.

    This class wraps the a kereas model implementation to trace the model without external parameters. This produces
    a list of weights that can then be used to directly call the model with a given list of weights.
    This approach should work for any net that is sufficiently modified to pass the parameters
    """

    def __init__(self, args, architecture='ResNet50', data='CIFAR10'):
        super().__init__(name=architecture)

        self.args = args
        if self.args.bit64:
            raise NotImplementedError()
        self.architecture = architecture
        self.data = data

        if data == 'CIFAR10':
            self.num_classes = 10
            self.expected_shape = (32, 32, 3)
        elif data == 'ImageNet':
            self.num_classes = 1000
            self.expected_shape = (224, 224, 3)

        self.flatten = layers.Flatten(name='features')
        self.dense = Dense(self.num_classes,
                           activation=None,
                           name='logits',
                           kernel_initializer=initializers.RandomNormal(stddev=0.01),
                           kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                           bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY))

    def call(self, inputs=None, params=defaultdict(lambda: None), preprocessing=True):
        if preprocessing:
            data = self.data
        else:
            data = None
        if data == 'CIFAR10':
            datamean, datastd = np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))
            inputs = tf_standardize(inputs, datamean, datastd)
        elif data == 'ImageNet':
            pass  # depends on the network arch and will be handled separately

        if self.architecture in ['ResNet50', 'ResNet101', 'ResNet152']:
            if data == 'ImageNet':
                inputs = keras.applications.resnet.preprocess_input(inputs)
            x, _ = getattr(keras_applications.resnet, self.architecture)(include_top=False,
                                                                         weights=None,
                                                                         input_tensor=inputs,
                                                                         input_shape=self.expected_shape,
                                                                         pooling='avg',
                                                                         classes=self.num_classes, params=params)
        elif self.architecture in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']:
            if data == 'ImageNet':
                inputs = keras_applications.resnet_v2.preprocess_input(inputs)
            x, _ = getattr(keras_applications.resnet_v2, self.architecture)(include_top=False,
                                                                            weights=None,
                                                                            input_tensor=inputs,
                                                                            input_shape=self.expected_shape,
                                                                            pooling='avg',
                                                                            classes=self.num_classes, params=params)

        elif self.architecture in ['ResNeXt50', 'ResNeXt101']:
            if data == 'ImageNet':
                inputs = keras_applications.resnext.preprocess_input(inputs)
            x, _ = getattr(keras_applications.resnext, self.architecture)(include_top=False,
                                                                          weights=None,
                                                                          input_tensor=inputs,
                                                                          input_shape=self.expected_shape,
                                                                          pooling='avg',
                                                                          classes=self.num_classes, params=params)

        elif self.architecture in ['DenseNet40', 'DenseNet121', 'DenseNet169', 'DenseNet201']:
            if data == 'ImageNet':
                inputs = keras_applications.densenet.preprocess_input(inputs)
            x, _ = getattr(keras_applications.densenet, self.architecture)(include_top=False,
                                                                           input_tensor=inputs,
                                                                           input_shape=self.expected_shape,
                                                                           pooling='avg',
                                                                           classes=self.num_classes, params=params)

        elif self.architecture in ['NASNetMobile', 'NASNetLarge']:
            if data == 'ImageNet':
                inputs = keras_applications.nasnet.preprocess_input(inputs)
            x, _ = getattr(keras_applications.nasnet, self.architecture)(include_top=False,
                                                                         weights=None,
                                                                         input_tensor=inputs,
                                                                         input_shape=self.expected_shape,
                                                                         pooling='avg',
                                                                         classes=self.num_classes, params=params)
        elif self.architecture in ['InceptionResNetV2']:
            if data == 'ImageNet':
                inputs = keras_applications.inception_resnet_v2.preprocess_input(inputs)
            x, _ = getattr(keras_applications.inception_resnet_v2, self.architecture)(include_top=False,
                                                                                      weights=None,
                                                                                      input_tensor=inputs,
                                                                                      input_shape=self.expected_shape,
                                                                                      pooling='avg',
                                                                                      classes=self.num_classes, params=params)
        elif self.architecture in ['InceptionV3']:
            if data == 'ImageNet':
                inputs = keras_applications.inception_v3.preprocess_input(inputs)
            x, _ = getattr(keras_applications.inception_v3, self.architecture)(include_top=False,
                                                                               weights=None,
                                                                               input_tensor=inputs,
                                                                               input_shape=self.expected_shape,
                                                                               pooling='avg',
                                                                               classes=self.num_classes, params=params)
        elif self.architecture in ['VGG19']:
            if data == 'ImageNet':
                inputs = keras_applications.vgg19.preprocess_input(inputs)
            x, _ = keras_applications.vgg19.VGG19(include_top=False,
                                                  weights=None,
                                                  input_tensor=inputs,
                                                  input_shape=self.expected_shape,
                                                  pooling='max',
                                                  classes=self.num_classes, params=params)
        elif self.architecture in ['VGG16']:
            if data == 'ImageNet':
                inputs = keras_applications.vgg16.preprocess_input(inputs)
            x, _ = keras_applications.vgg16.VGG16(include_top=False,
                                                  weights=None,
                                                  input_tensor=inputs,
                                                  input_shape=self.expected_shape,
                                                  pooling='max',
                                                  classes=self.num_classes, params=params)
        elif self.architecture in ['VGG13']:
            if data == 'ImageNet':
                inputs = keras_applications.vgg13.preprocess_input(inputs)
            x, _ = keras_applications.vgg13.VGG13(include_top=False,
                                                  input_tensor=inputs,
                                                  input_shape=self.expected_shape,
                                                  pooling='max',
                                                  classes=self.num_classes, params=params)
        elif self.architecture in ['VGG11']:
            if data == 'ImageNet':
                inputs = keras_applications.vgg11.preprocess_input(inputs)
            x, _ = keras_applications.vgg11.VGG11(include_top=False,
                                                  input_tensor=inputs,
                                                  input_shape=self.expected_shape,
                                                  pooling='max',
                                                  classes=self.num_classes, params=params)
        elif self.architecture in ['Xception']:
            if data == 'ImageNet':
                inputs = keras_applications.xception.preprocess_input(inputs)
            x, _ = keras_applications.xception.Xception(include_top=False,
                                                        weights=None,
                                                        input_tensor=inputs,
                                                        input_shape=self.expected_shape,
                                                        pooling='avg',
                                                        classes=self.num_classes, params=params)
        elif self.architecture in ['MobileNet']:
            if data == 'ImageNet':
                inputs = keras_applications.mobilenet.preprocess_input(inputs)
            x, _ = keras_applications.mobilenet.MobileNet(include_top=False,
                                                          weights=None,
                                                          input_tensor=inputs,
                                                          input_shape=self.expected_shape,
                                                          pooling='avg',
                                                          classes=self.num_classes, params=params)
        elif self.architecture in ['MobileNetV2']:
            if data == 'ImageNet':
                inputs = keras_applications.mobilenet_v2.preprocess_input(inputs)
            x, _ = keras_applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                               alpha=1.0,
                                                               weights=None,
                                                               input_tensor=inputs,
                                                               input_shape=self.expected_shape,
                                                               pooling='avg',
                                                               classes=self.num_classes, params=params)
        else:
            raise NotImplementedError('Unknown architecture.')
        features = self.flatten(x)
        logits = self.dense(features, params=params)
        return logits, features

    def construct_weights(self):
        """Infer temp. weights by tracing a default model and then collecting its parameters.

        Return weights from traced parameters as dictionary with unscoped names.
        """
        input = tf.keras.Input(shape=self.expected_shape)
        model = models.Model(input, self.call(input, preprocessing=False), name=self.architecture)
        current_scope = tf.get_default_graph().get_name_scope()
        scope_len = len(current_scope) + 1
        params = {param.name[scope_len:] : param for param in model.trainable_weights}
        return params, {}

    def forward(self, inputs, params, buffers={}):  # reuse=False, scope='' ?
        """Call Resnet functionally and return output tensor for given input and given weights.

        Alias for self.call
        """
        logits, feats = self.call(inputs, params)
        return logits, feats, buffers
