"""Custom modules to call tf layers with external weights."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.keras.utils import conv_utils


class Conv2D(layers.Conv2D):
    """As Conv2D with overwritten call method."""

    __doc__ += layers.Conv2D.__doc__

    def call(self, inputs, params=None):

        if params[self.name + '/kernel:0'] is None:
            return super(layers.Conv2D, self).call(inputs)
        else:
            kernel = params.get(self.name + '/kernel:0')
            bias = params.get(self.name + '/bias:0')
        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class BatchNormalization(layers.BatchNormalization):
    """As Batchnorm (v2) with overwritten call method."""

    __doc__ += layers.BatchNormalization.__doc__

    def call(self, inputs, params=None, training=None):

        if params[self.name + '/gamma:0'] is None:
            return super(layers.BatchNormalization, self).call(inputs)
        else:
            gamma = params.get(self.name + '/gamma:0')
            beta = params.get(self.name + '/beta:0')

        training = self._get_training_value(training)
        if self.virtual_batch_size is not None:
            raise NotImplementedError()

        if not self.fused:
            raise NotImplementedError()
        else:
            outputs = self._fused_batch_norm(inputs, training=training, gamma=gamma, beta=beta)
        return outputs

    def _fused_batch_norm(self, inputs, training, beta=None, gamma=None):
        """Returns the output of fused batch norm."""
        if beta is None:
            beta = self.beta if self.center else self._beta_const
        if gamma is None:
            gamma = self.gamma if self.scale else self._gamma_const

        inputs_size = array_ops.size(inputs)

        def _fused_batch_norm_training():
            return nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                epsilon=self.epsilon,
                data_format=self._data_format)

        def _fused_batch_norm_inference():
            return nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format)

        output, mean, variance = tf_utils.smart_cond(training, _fused_batch_norm_training, _fused_batch_norm_inference)
        if not self._bessels_correction_test_only:
            # Remove Bessel's correction to be consistent with non-fused batch norm.
            # Note that the variance computed by fused batch norm is
            # with Bessel's correction.
            sample_size = math_ops.cast(
                array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
            factor = (sample_size - math_ops.cast(1.0, variance.dtype)) / sample_size
            variance *= factor

        training_value = tf_utils.constant_value(training)
        if training_value is None:
            momentum = tf_utils.smart_cond(training,
                                           lambda: self.momentum,
                                           lambda: 1.0)
        else:
            momentum = ops.convert_to_tensor(self.momentum)
        if training_value or training_value is None:
            if distribution_strategy_context.in_cross_replica_context():
                raise NotImplementedError()
            else:

                def mean_update():
                    return self._assign_moving_average(self.moving_mean, mean, momentum,
                                                       inputs_size)

                def variance_update():
                    return self._assign_moving_average(self.moving_variance, variance,
                                                       momentum, inputs_size)

            self.add_update(mean_update, inputs=True)
            self.add_update(variance_update, inputs=True)

        return output


class Dense(layers.Dense):
    """As normal Dense with overwritten call method."""

    __doc__ += layers.Dense.__doc__

    def call(self, inputs, params=None):

        if params[self.name + '/kernel:0'] is None:
            return super(layers.Dense, self).call(inputs)
        else:
            kernel = params.get(self.name + '/kernel:0')
            bias = params.get(self.name + '/bias:0')

        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            # Cast the inputs to self.dtype, which is the variable dtype. We do not
            # cast if `should_cast_variables` is True, as in that case the variable
            # will be automatically casted to inputs.dtype.
            if not self._mixed_precision_policy.should_cast_variables:
                inputs = math_ops.cast(inputs, self.dtype)
            outputs = gen_math_ops.mat_mul(inputs, kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class DepthwiseConv2D(layers.DepthwiseConv2D):
    """As normal DepthwiseConv2D but with overwritten call method."""

    __doc__ += layers.DepthwiseConv2D.__doc__

    def call(self, inputs, params=None):
        if params[self.name + '/depthwise_kernel:0'] is None:
            return super(layers.DepthwiseConv2D, self).call(inputs)
        else:
            depthwise_kernel = params.get(self.name + '/depthwise_kernel:0')
            bias = params.get(self.name + '/bias:0')

        outputs = backend.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.use_bias:
            outputs = backend.bias_add(
                outputs,
                bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs


class SeparableConv2D(layers.SeparableConv2D):
    """As separable Conv2D but with overwritten call to allow for external weights."""

    __doc__ += layers.SeparableConv2D.__doc__

    def call(self, inputs, params=None):
        if params[self.name + '/depthwise_kernel:0'] is None:
            return super(layers.SeparableConv2D, self).call(inputs)
        else:
            depthwise_kernel = params.get(self.name + '/depthwise_kernel:0')
            pointwise_kernel = params.get(self.name + '/pointwise_kernel:0')
            bias = params.get(self.name + '/bias:0')
        # Apply the actual ops.
        if self.data_format == 'channels_last':
            strides = (1,) + self.strides + (1,)
        else:
            strides = (1, 1) + self.strides
        outputs = nn.separable_conv2d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides=strides,
            padding=self.padding.upper(),
            rate=self.dilation_rate,
            data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
