import tensorflow as tf
import tensorflow_probability as tfp

def recolor(inputs, colorperts, name=None, eye=None):
    '''
    :param img: input images on which recolor is applied. Shape [B, H, W, 3]. Type is numpy array
    :param gridshape: Shape [nL, nU, nV]. Number of points along each axis in the colorspace grid
    :return: color-transformed image, color perturbation function trainable parameters
    '''
    gridshape = colorperts.shape.as_list()[1:]
    inputshape = inputs.shape.as_list()
    assert len(inputshape) == 4, 'input to recolor must be rank 4'
    assert len(gridshape) == 4, 'gridshape to recolor must be rank 4'

    # define identity color transform. Shape [ncolorres, ncolorres, ncolorres, 3]
    xrefmin = [0., -.5, -.5]  # minimum/maximum values for LUV channels
    xrefmax = [1., .5, .5]
    if eye is None:
        eye = tf.meshgrid(*[tf.linspace(start=start, stop=stop, num=ncolorres) for start, stop, ncolorres in zip(xrefmin, xrefmax, gridshape)], indexing='ij')
        eye = tf.stack(eye, axis=-1)

    # take a single image and a color perturbation grid and perform the color transformation
    def _recolor(arg):
        img, colorpert = arg  # img and colorpert shape [ncolorres, ncolorres, ncolorres, 3]
        img = tf.image.rgb_to_yuv(img) / 255.
        yref = eye + colorpert
        img = tfp.math.batch_interp_regular_nd_grid(img, xrefmin, xrefmax, yref, axis=-4)
        # img = tf.image.yuv_to_rgb(img) * 255.
        return img
    
    # apply _recolor to all images in batch
    outputs = tf.map_fn(_recolor, (inputs, colorperts), dtype=tf.float32, parallel_iterations=500, name=name)
    outputs = tf.image.yuv_to_rgb(outputs) * 255.
    return outputs, eye, xrefmin, xrefmax


def smoothloss(colorperts, gridshape):
    '''get the mean norm of the discrete gradients along each direction in colorspace'''
    
    # the gradients in each direction are calculated as the difference between the neighboring grid points in colorspace divided by their distance which is 1 / ncolorres
    dpert_y = colorperts[:, :-1, :, :, :] - colorperts[:, 1:, :, :, :]
    dpert_u = colorperts[:, :, :-1, :, :] - colorperts[:, :, 1:, :, :]
    dpert_v = colorperts[:, :, :, :-1, :] - colorperts[:, :, :, 1:, :]
    flattened = tf.concat([tf.reshape((d * ncolorres) ** 2, [-1]) for d, ncolorres in zip([dpert_y, dpert_u, dpert_v], gridshape)], axis=0)
    smoothloss = tf.reduce_mean(flattened)
    return smoothloss

