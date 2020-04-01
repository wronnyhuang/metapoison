"""Implement vgg directly. Inspired by https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
import tensorflow as tf
# from tensorflow.contrib.layers.python import layers as tf_layers
import numpy as np
from utils import count_params_in_scope
from data import tf_standardize

from .convnet import ConvNet


class VGG(ConvNet):
    """General scriptable VGG class."""

    def __init__(self, args, blocks=[1, 1, 2, 2, 2], batchnorm=False):
        """Initialize with block config and batchnorm yes/no."""
        super().__init__(args)

        self.blocks = blocks
        self.batchnorm = batchnorm

    def construct_weights(self):
        """Construct all resnet weights."""
        weights, buffers = {}, {}
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=self.floattype)

        num_filter_per_stage = [self.channels, 64, 128, 256, 512, 512] # VGG defaults
        k = 3 # conv size

        # 1) Deep layers
        for idx, block in enumerate(self.blocks):
            for layer in range(block):
                filters_in = num_filter_per_stage[idx] if layer == 0 else num_filter_per_stage[idx + 1]
                filters_out = num_filter_per_stage[idx + 1]
                self.weight_layer(weights, buffers, f'{idx}_{layer}', filters_in, filters_out, size=k)

                print(f'This is stage idx {idx}, block {layer}: {filters_in}:{filters_out}')

        # 2) Output Dense
        weights['dense_weights_1'] = tf.get_variable(
            'dense_weights_1', [512, 512], initializer=fc_initializer, dtype=self.floattype)
        weights['dense_bias_1'] = tf.Variable(
            tf.zeros([512], dtype=self.floattype), name='dense_bias_1')
        weights['dense_weights_2'] = tf.get_variable(
            'dense_weights_2', [512, 512], initializer=fc_initializer, dtype=self.floattype)
        weights['dense_bias_2'] = tf.Variable(
            tf.zeros([512], dtype=self.floattype), name='dense_bias_2')

        # Classifier
        weights['classifier_weights'] = tf.get_variable(
            'classifier_weights', [512, self.dim_output], initializer=fc_initializer, dtype=self.floattype)
        weights['classifier_bias'] = tf.Variable(
            tf.zeros([self.dim_output], dtype=self.floattype), name='classifier_bias')

        return weights, buffers

    def weight_layer(self, weights, buffers, id_string, num_filters_in, num_filters_out, size=3, groups=1):
        """Create tf variables for a single res_layer."""
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=self.floattype)
        if groups == 1:
            weights['conv_' + id_string] = tf.get_variable('conv_' +  id_string, [size, size, num_filters_in, num_filters_out],
                                                           initializer=conv_initializer, dtype=self.floattype)
        else:
            weights['conv_' + id_string] = tf.get_variable('conv_' +  id_string, [size, size, num_filters_in, num_filters_out // num_filters_in],
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
        else:
            weights['bias_' + id_string] = tf.Variable(tf.zeros([num_filters_out],
                                                                dtype=self.floattype), name='bias_' + id_string)

        return weights, buffers

    def bn_layer(self, inp, conv, scale, offset, buffermean, buffervar, training, stride=1, activation=tf.nn.relu):
        """Perform, conv, batch norm, nonlinearity."""
        inp = tf.nn.conv2d(inp, conv, [1, stride, stride, 1], 'SAME')
        if self.batchnorm:
            inp, buffermean, buffervar = self.batch_norm(inp, buffermean, buffervar, scale, offset, training)
        else:
            buffermean, buffervar = None, None
        inp = activation(inp)
        return inp, buffermean, buffervar

    def conv_layer(self, inp, conv, bias, training, stride=1, activation=tf.nn.relu):
        """Perform, conv, batch norm, nonlinearity."""
        inp = tf.nn.conv2d(inp, conv, [1, stride, stride, 1], 'SAME') + bias
        buffermean, buffervar = None, None
        inp = activation(inp)
        return inp, buffermean, buffervar

    def forward(self, x, weights, buffers, training=True):

        stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
        newbuffers = {}
        datamean, datastd = np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))
        x = tf_standardize(x, datamean, datastd)

        for idx, block in enumerate(self.blocks):
            for layer in range(block):
                if self.batchnorm:
                    x, newbuffers[f'mean_{idx}_{layer}'], newbuffers[f'var_{idx}_{layer}'] = \
                        self.bn_layer(x, weights[f'conv_{idx}_{layer}'], weights.get(f'scale_{idx}_{layer}'),
                                      weights.get(f'offset_{idx}_{layer}'),
                                      buffers.get(f'mean_{idx}_{layer}'), buffers.get(f'var_{idx}_{layer}'),
                                      training, stride=1, activation=tf.nn.relu)
                else:
                    x, newbuffers[f'mean_{idx}_{layer}'], newbuffers[f'var_{idx}_{layer}'] = \
                        self.conv_layer(x, weights[f'conv_{idx}_{layer}'], weights[f'bias_{idx}_{layer}'],
                                        training, stride=1, activation=tf.nn.relu)

            x = tf.nn.max_pool(x, stride, stride, 'VALID')

        # Flatten:
        x = tf.reshape(x, [-1, np.prod([int(dim) for dim in x.get_shape()[1:]])])

        # Dense layers
        x = tf.nn.relu(tf.matmul(x, weights['dense_weights_1']) + weights['dense_bias_1'])
        features = tf.nn.relu(tf.matmul(x, weights['dense_weights_2']) + weights['dense_bias_2'])
        logits = tf.matmul(features, weights['classifier_weights']) + weights['classifier_bias']

        if self.batchnorm:
            returnbuffers = newbuffers
        else:
            returnbuffers = {}

        return logits, features, returnbuffers


def VGG11(args):
    """Return VGG-11."""
    return VGG(args, blocks=[1, 1, 2, 2, 2], batchnorm=False)

def VGG13(args):
    """Return VGG-13."""
    return VGG(args, blocks=[2, 2, 2, 2, 2], batchnorm=False)

def VGG16(args):
    """Return VGG-16."""
    return VGG(args, blocks=[2, 2, 3, 3, 3], batchnorm=False)

def VGG11BN(args):
    """Return VGG-11."""
    return VGG(args, blocks=[1, 1, 2, 2, 2], batchnorm=True)

def VGG13BN(args):
    """Return VGG-13."""
    return VGG(args, blocks=[2, 2, 2, 2, 2], batchnorm=True)

def VGG16BN(args):
    """Return VGG-16."""
    return VGG(args, blocks=[2, 2, 3, 3, 3], batchnorm=True)
