import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
import tensorflow as tf
# from tensorflow.contrib.layers.python import layers as tf_layers
import numpy as np
from utils import count_params_in_scope
from data import tf_standardize

# 5-layer convnet: from cbfinn's maml implementation. Batch norm capabilities added

class ConvNet():

    def __init__(self, args):

        self.args = args
        self.channels = 3
        self.dim_hidden = [16, 32, 32, 64, 64]
        # self.dim_hidden = [16, 16, 16, 32, 32]
        self.dim_output = 10
        self.img_size = 32
        self.batchnorm = False
        self.max_pool = True
        self.floattype = tf.float64 if self.args.bit64 else tf.float32
        self.inttype = tf.int64 if self.args.bit64 else tf.int32

    def construct_weights(self):
        weights, buffers = {}, {}
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=self.floattype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=self.floattype)
        k = 3

        for i in range(len(self.dim_hidden)):
            previous = self.channels if i == 0 else self.dim_hidden[i - 1]
            weights['conv' + str(i + 1)] = tf.get_variable('conv' + str(i + 1), [k, k, previous, self.dim_hidden[i]], initializer=conv_initializer, dtype=self.floattype)
            if not self.batchnorm:
                weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]], dtype=self.floattype), name='b' + str(i + 1))
            else:
                weights['scale' + str(i + 1)] = tf.Variable(tf.ones([self.dim_hidden[i]], dtype=self.floattype), name='scale' + str(i + 1))
                weights['offset' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]], dtype=self.floattype), name='offset' + str(i + 1))
                # Explicit BatchNorm buffers
                buffers['mean' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]), name='mean' + str(i + 1), trainable=False)
                buffers['var' + str(i + 1)] = tf.Variable(tf.ones([self.dim_hidden[i]]), name='var' + str(i + 1), trainable=False)
            
        # assumes max pooling
        weights['w6'] = tf.get_variable('w6', [self.dim_hidden[-1], self.dim_output], initializer=fc_initializer, dtype=self.floattype)
        weights['b6'] = tf.Variable(tf.zeros([self.dim_output], dtype=self.floattype), name='b6')

        return weights, buffers # buffers are the batch statistics at the various layers

    def forward(self, inp, weights, buffers, training=True):
        newbuffers = {}
        hiddens = []
        datamean, datastd = np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))
        hidden = tf_standardize(inp, datamean, datastd)
        for l in range(len(self.dim_hidden)):
            if not self.batchnorm:
                hidden = self.conv_block(hidden, weights[f'conv{l + 1}'], weights[f'b{l + 1}'])
            else:
                hidden, buffermean, buffervar = self.convbn_block(hidden, weights[f'conv{l + 1}'], weights[f'scale{l + 1}'],
                                                                  weights[f'offset{l + 1}'], buffers[f'mean{l + 1}'],
                                                                  buffers[f'var{l + 1}'], training)
                newbuffers[f'mean{l + 1}'] = buffermean
                newbuffers[f'var{l + 1}'] = buffervar
            hiddens.append(hidden)
        hidden = tf.reshape(hidden, [-1, np.prod([int(dim) for dim in hidden.get_shape()[1:]])])
        logits = tf.matmul(hidden, weights['w6']) + weights['b6']
        return logits, hiddens, newbuffers

    def conv_block(self, inp, conv, bias, activation=tf.nn.relu, max_pool_pad='VALID'):
        """Perform, conv, batch norm, nonlinearity, and max pool."""
        stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
        inp = tf.nn.conv2d(inp, conv, no_stride, 'SAME') + bias
        inp = activation(inp)
        inp = tf.nn.max_pool(inp, stride, stride, max_pool_pad)
        return inp

    def convbn_block(self, inp, conv, scale, offset, buffermean, buffervar, training, activation=tf.nn.relu, max_pool_pad='VALID'):
        """Perform, conv, batch norm, nonlinearity, and max pool."""
        stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
        inp = tf.nn.conv2d(inp, conv, no_stride, 'SAME')
        inp, buffermean, buffervar = self.batch_norm(inp, buffermean, buffervar, scale, offset, training)
        inp = activation(inp)
        inp = tf.nn.max_pool(inp, stride, stride, max_pool_pad)
        return inp, buffermean, buffervar
    
    def batch_norm(self, inp, buffermean, buffervar, scale, offset, training, decay=0.9, eps=1e-5):
        if training:
            curmean, curvar = tf.nn.moments(inp, axes=[0, 1, 2], keep_dims=False)
            newbuffermean = decay * buffermean + (1 - decay) * curmean
            newbuffervar = decay * buffervar + (1 - decay) * curvar
            self.curmean, self.curvar = curmean, curvar
        else:
            curmean, curvar = buffermean, buffervar
            newbuffermean = newbuffervar = None
        inp = tf.nn.batch_normalization(inp, curmean, curvar, offset, scale, eps)
        return inp, newbuffermean, newbuffervar

class ConvNetBN(ConvNet):
    def __init__(self, args):
        super().__init__(args)
        self.batchnorm = True
