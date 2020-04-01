"""Implement a resnet directly. Inspired by https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
import tensorflow as tf
# from tensorflow.contrib.layers.python import layers as tf_layers
import numpy as np
from utils import count_params_in_scope
from data import tf_standardize

from .vgg import VGG

CFG = [(1, 16, 1, 1),
       (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6, 32, 3, 2),
       (6, 64, 4, 2),
       (6, 96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]


class MobileNet(VGG):
    """Implement default resnets in a functional representation.

    Weights are stored separately for easy meta-unrolling.
    """

    def __init__(self, args, cfg=CFG, batchnorm=True):
        """Initialize everything as in the ConvNet and then prepare the ResNet structure."""
        super().__init__(args)
        self.cfg = cfg
        self.batchnorm = batchnorm

    def construct_weights(self):
        """Construct all resnet weights."""
        weights, buffers = {}, {}
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=self.floattype)

        in_planes = 32
        k = 3 # conv size

        # Input layer:
        self.weight_layer(weights, buffers, 'input', 3, 32)

        # 1) Deep layers
        for idx, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for ids, stride in enumerate(strides):
                planes = expansion * in_planes
                self.weight_layer(weights, buffers, f'{idx}_{ids}_1', in_planes, planes, size=1)
                self.weight_layer(weights, buffers, f'{idx}_{ids}_2', planes, planes, size=k, groups=planes)
                self.weight_layer(weights, buffers, f'{idx}_{ids}_3', planes, out_planes, size=1)
                if stride == 1 and in_planes != out_planes:
                    self.weight_layer(weights, buffers, f'{idx}_{ids}_r', in_planes, out_planes, size=1)
                in_planes = out_planes

        # Output layer
        self.weight_layer(weights, buffers, 'output', 320, 1280, size=1)
        # Classifier
        weights['classifier_weights'] = tf.get_variable(
            'classifier_weights', [1280, self.dim_output], initializer=fc_initializer, dtype=self.floattype)
        weights['classifier_bias'] = tf.Variable(
            tf.zeros([self.dim_output], dtype=self.floattype), name='classifier_bias')

        return weights, buffers

    def forward_layer(self, x, weights, buffers, idx, training, stride=1, groups=1, activation=tf.nn.relu):
        if groups > 1:
            x = tf.nn.depthwise_conv2d(x, weights['conv_' + idx], [1, stride, stride, 1], 'SAME')
        else:
            x = tf.nn.conv2d(x, weights['conv_' + idx], [1, stride, stride, 1], 'SAME')
        if self.batchnorm:
            x, buffermean, buffervar = self.batch_norm(x, buffers['mean_' + idx], buffers['var_' + idx],
                                                       weights['scale_' + idx], weights['offset_' + idx], training)
        else:
            x, buffermean, buffervar = x + weights['bias_' + idx], None, None
        x = activation(x)
        return x, buffermean, buffervar

    def forward(self, x, weights, buffers, training=True):

        stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
        newbuffers = {}
        datamean, datastd = np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))
        x = tf_standardize(x, datamean, datastd)

        # Input layer
        in_planes = 32
        x, newbuffers[f'mean_input'], newbuffers[f'var_input'] = self.forward_layer(x, weights, buffers, 'input', training)

        # 1) Deep layers
        for idx, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for ids, stride in enumerate(strides):
                # print(f'{idx}:: This is stride {ids}, cfg: {expansion}:{out_planes}:{num_blocks}:{stride}')
                y, newbuffers[f'mean_{idx}_{ids}_1'], newbuffers[f'var_{idx}_{ids}_1'] = \
                    self.forward_layer(x, weights, buffers, f'{idx}_{ids}_1', training)
                y, newbuffers[f'mean_{idx}_{ids}_2'], newbuffers[f'var_{idx}_{ids}_2'] = \
                    self.forward_layer(y, weights, buffers, f'{idx}_{ids}_2', training, stride=stride, groups=out_planes)
                y, newbuffers[f'mean_{idx}_{ids}_3'], newbuffers[f'var_{idx}_{ids}_3'] = \
                    self.forward_layer(y, weights, buffers, f'{idx}_{ids}_3', training)
                if stride == 1 and in_planes != out_planes:
                    shortcut, newbuffers[f'mean_{idx}_{ids}_r'], newbuffers[f'var_{idx}_{ids}_r'] = \
                        self.forward_layer(x, weights, buffers, f'{idx}_{ids}_r', training)
                    x = y + shortcut
                else:
                    x = y
                in_planes = out_planes

        # Output layer:
        x, newbuffers[f'mean_output'], newbuffers[f'var_output'] = self.forward_layer(x, weights, buffers, 'output', training)
        # Dense layers
        x = tf.nn.avg_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')
        features = tf.reshape(x, [-1, np.prod([int(dim) for dim in x.get_shape()[1:]])])
        logits = tf.matmul(features, weights['classifier_weights']) + weights['classifier_bias']

        if self.batchnorm:
            returnbuffers = newbuffers
        else:
            returnbuffers = {}

        return logits, features, returnbuffers
