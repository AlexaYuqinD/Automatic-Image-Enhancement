"""
This file is used to load pre-trained VGG model
"""
# coding: utf-8

import numpy as np
import scipy.io
import tensorflow as tf
import utils

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
VGG_FILENAME = "imagenet-vgg-verydeep-19.mat"
EXPECTED_BYTES = 534904783  # size of file


class VGG(object):
    def __init__(self, input_img):
        # download the file
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME, EXPECTED_BYTES)
        # load the file
        self.vgg_layers = scipy.io.loadmat(VGG_FILENAME)["layers"]
        self.input_img = input_img
        # compute channel mean
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    def _weights(self, layer_idx, expected_layer_name):
        """
        get pre-trained weights of a specific layer
        
        :param layer_idx: layer id in VGG
        :param expected_layer_name: layer name
        :return: pre-trained W and b
        """
        W = self.vgg_layers[0][layer_idx][0][0][2][0][0]
        b = self.vgg_layers[0][layer_idx][0][0][2][0][1]
        # name of current layer
        layer_name = self.vgg_layers[0][layer_idx][0][0][0][0]
        assert layer_name == expected_layer_name, print("Layer name error!")

        return W, b.reshape(b.size)

    def conv2d_relu(self, prev_layer, layer_idx, layer_name):
        """
        Use ReLU as activation
        
        :param prev_layer: previous layer
        :param layer_idx: layer id in VGG
        :param layer_name: layer name
        """
        with tf.variable_scope(layer_name):
            # get current weights(numpy array)
            W, b = self._weights(layer_idx, layer_name)
            # convert to tensor
            W = tf.constant(W, name="weights")
            b = tf.constant(b, name="bias")
            conv2d = tf.nn.conv2d(input=prev_layer,
                                  filter=W,
                                  strides=[1, 1, 1, 1],
                                  padding="SAME")
            out = tf.nn.relu(conv2d + b)
        setattr(self, layer_name, out)

    def avgpool(self, prev_layer, layer_name):
        """
        average pooling layer (see reference paper)
        
        :param prev_layer: previous layer(conv layer)
        :param layer_name: layer name
        """
        with tf.variable_scope(layer_name):
            # average pooling
            out = tf.nn.avg_pool(value=prev_layer,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME")

        setattr(self, layer_name, out)

    def load(self):
        """
        load pre-trained data
        """
        self.conv2d_relu(self.input_img, 0, "conv1_1")
        self.conv2d_relu(self.conv1_1, 2, "conv1_2")
        self.avgpool(self.conv1_2, "avgpool1")
        self.conv2d_relu(self.avgpool1, 5, "conv2_1")
        self.conv2d_relu(self.conv2_1, 7, "conv2_2")
        self.avgpool(self.conv2_2, "avgpool2")
        self.conv2d_relu(self.avgpool2, 10, "conv3_1")
        self.conv2d_relu(self.conv3_1, 12, "conv3_2")
        self.conv2d_relu(self.conv3_2, 14, "conv3_3")
        self.conv2d_relu(self.conv3_3, 16, "conv3_4")
        self.avgpool(self.conv3_4, "avgpool3")
        self.conv2d_relu(self.avgpool3, 19, "conv4_1")
        self.conv2d_relu(self.conv4_1, 21, "conv4_2")
        self.conv2d_relu(self.conv4_2, 23, "conv4_3")
        self.conv2d_relu(self.conv4_3, 25, "conv4_4")
        self.avgpool(self.conv4_4, "avgpool4")
        self.conv2d_relu(self.avgpool4, 28, "conv5_1")
        self.conv2d_relu(self.conv5_1, 30, "conv5_2")
        self.conv2d_relu(self.conv5_2, 32, "conv5_3")
        self.conv2d_relu(self.conv5_3, 34, "conv5_4")
        self.avgpool(self.conv5_4, "avgpool5")