"""Enables dynamic setting of underlying Keras module."""


from tensorflow.python.keras import backend
from tensorflow.python.keras import engine
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import utils
from tensorflow.python.util import tf_inspect

from learners.modules import Conv2D, BatchNormalization, Dense, DepthwiseConv2D, SeparableConv2D


_KERAS_BACKEND = backend
_KERAS_LAYERS = layers
_KERAS_MODELS = models
_KERAS_UTILS = utils


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    layers.Conv2D = Conv2D
    layers.BatchNormalization = BatchNormalization
    layers.Dense = Dense
    layers.DepthwiseConv2D = DepthwiseConv2D
    layers.SeparableConv2D = SeparableConv2D
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

__version__ = '1.0.8'

from . import vgg11
from . import vgg13
from . import vgg16
from . import vgg19
from . import resnet50
from . import inception_v3
from . import inception_resnet_v2
from . import xception
from . import mobilenet
from . import mobilenet_v2
from . import densenet
from . import nasnet
from . import resnet
from . import resnet_v2
from . import resnext
