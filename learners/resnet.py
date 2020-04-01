"""Implement a resnet directly. Inspired by https://keras.io/examples/cifar10_resnet/."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
import tensorflow as tf
# from tensorflow.contrib.layers.python import layers as tf_layers
import numpy as np
from utils import count_params_in_scope
from data import tf_standardize

from .convnet import ConvNet


class ResNet(ConvNet):
    """Implement default resnets in a functional representation.

    Weights are stored separately for easy meta-unrolling.
    """

    def __init__(self, args, structure=[3, 3, 3], block_type='basic', batchnorm=True, filters=16):
        """Initialize everything as in the ConvNet and then prepare the ResNet structure."""
        super().__init__(args)
        self.structure = structure
        self.block_type = block_type
        self.num_filters = filters

        self.batchnorm = batchnorm

    def construct_weights(self):
        """Construct all resnet weights."""
        weights, buffers = {}, {}
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=self.floattype)

        num_filters_in = self.channels
        num_filters = self.num_filters
        k = 3

        # 0) Input layer
        self.weight_layer(weights, buffers, 'input', num_filters_in, num_filters)
        num_filters_in = num_filters

        # 1) Deep layers
        for idx, stage in enumerate(self.structure):
            for block in range(stage):
                # print(f'This is stage idx {idx}, block {block}: {num_filters_in}:{num_filters}')
                self.weight_layer(weights, buffers, f'{idx}_{block}_1', num_filters_in, num_filters, size=k)
                self.weight_layer(weights, buffers, f'{idx}_{block}_2', num_filters, num_filters, size=k)
                if idx > 0 and block == 0:
                    self.weight_layer(weights, buffers, f'{idx}_{block}_r', num_filters_in, num_filters, size=1)
            num_filters_in = num_filters #todo no ref
            num_filters *= 2

        # 2) Output Dense
        weights['classifier_weights'] = tf.get_variable(
            'classifier_weights', [num_filters // 2, self.dim_output], initializer=fc_initializer, dtype=self.floattype)
        weights['classifier_bias'] = tf.Variable(
            tf.zeros([self.dim_output], dtype=self.floattype), name='classifier_bias')

        return weights, buffers

    def weight_layer(self, weights, buffers, id_string, num_filters_in, num_filters_out, size=3):
        """Create tf variables for a single res_layer."""
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=self.floattype)
        weights['conv_' + id_string] = tf.get_variable('conv_' +  id_string, [size, size, num_filters_in, num_filters_out],
                                                       initializer=conv_initializer, dtype=self.floattype)

        if self.batchnorm:
            weights['scale_' + id_string] = tf.Variable(tf.ones([num_filters_out],
                                                                dtype=self.floattype), name='scale_' + id_string)
            weights['offset_' + id_string] = tf.Variable(tf.zeros([num_filters_out],
                                                                  dtype=self.floattype), name='offset_' + id_string)
            # Explicit BatchNorm buffers
            buffers['mean_' + id_string] = tf.Variable(tf.zeros([num_filters_out]),
                                                       name='mean_' + id_string, trainable=False)
            buffers['var_' + id_string] = tf.Variable(tf.ones([num_filters_out]), name='var_' + id_string, trainable=False)

    def res_layer(self, inp, conv, scale, offset, buffermean, buffervar, training, stride=1, activation=tf.nn.relu):
        """Perform, conv, batch norm, nonlinearity."""
        inp = tf.nn.conv2d(inp, conv, [1, stride, stride, 1], 'SAME')
        if self.batchnorm:
            inp, buffermean, buffervar = self.batch_norm(inp, buffermean, buffervar, scale, offset, training)
        else:
            buffermean, buffervar = None, None
        inp = activation(inp)
        return inp, buffermean, buffervar

    def forward(self, input, weights, buffers, training=True):
        newbuffers = {}
        datamean, datastd = np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))
        y = tf_standardize(input, datamean, datastd)
        x, newbuffers[f'mean_input'], newbuffers[f'var_input'] = \
            self.res_layer(y, weights['conv_input'], weights.get('scale_input'), weights.get('offset_input'),
                           buffers.get('mean_input'), buffers.get('var_input'), training, stride=1)
        for idx, stage in enumerate(self.structure):
            for block in range(stage):
                stride = 1
                if idx > 0 and block == 0: stride = 2  # downsample
                y, newbuffers[f'mean_{idx}_{block}_1'], newbuffers[f'var_{idx}_{block}_1'] = \
                    self.res_layer(x, weights[f'conv_{idx}_{block}_1'], weights.get(f'scale_{idx}_{block}_1'),
                                   weights.get(f'offset_{idx}_{block}_1'),
                                   buffers.get(f'mean_{idx}_{block}_1'), buffers.get(f'var_{idx}_{block}_1'),
                                   training, stride=stride)

                y, newbuffers[f'mean_{idx}_{block}_2'], newbuffers[f'var_{idx}_{block}_2'] = \
                    self.res_layer(y, weights[f'conv_{idx}_{block}_2'], weights.get(f'scale_{idx}_{block}_2'),
                                   weights.get(f'offset_{idx}_{block}_2'),
                                   buffers.get(f'mean_{idx}_{block}_2'), buffers.get(f'var_{idx}_{block}_2'),
                                   training, stride=1, activation=tf.identity)
                if idx > 0 and block == 0:
                    x, newbuffers[f'mean_{idx}_{block}_r'], newbuffers[f'var_{idx}_{block}_r'] = \
                        self.res_layer(x, weights[f'conv_{idx}_{block}_r'], weights.get(f'scale_{idx}_{block}_r'),
                                       weights.get(f'offset_{idx}_{block}_r'),
                                       buffers.get(f'mean_{idx}_{block}_r'), buffers.get(f'var_{idx}_{block}_r'),
                                       training, stride=stride, activation=tf.identity)
                x = x + y
                x = tf.nn.relu(x)

        # Final pooling for head
        # print(x.shape)
        x = tf.nn.avg_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')
        # print(x.shape)
        features = tf.reshape(x, [-1, np.prod([int(dim) for dim in x.get_shape()[1:]])])
        logits = tf.matmul(features, weights['classifier_weights']) + weights['classifier_bias']
        if self.batchnorm:
            returnbuffers = newbuffers
        else:
            returnbuffers = {}

        return logits, features, returnbuffers
