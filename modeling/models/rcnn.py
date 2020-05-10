from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
import tf_slim as slim

from protos import rcnn_pb2
import tensorflow_hub as hub


def RCNN(inputs, proposals, options, is_training=True):
  """Runs RCNN model on the `inputs`.

  Args:
    inputs: Input image, a [batch, height, width, 3] uint8 tensor. The pixel
      values are in the range of [0, 255].
    proposals: Boxes used to crop the image features, using normalized
      coordinates. It should be a [batch, max_num_proposals, 4] float tensor
      denoting [y1, x1, y2, x2].
    options: A fast_rcnn_pb2.FastRCNN proto.
    is_training: If true, the model shall be executed in training mode.

  Returns:
    A [batch, max_num_proposals, feature_dims] tensor.

  Raises:
    ValueError if options is invalid.
  """
  if inputs.dtype != tf.uint8:
    raise ValueError('The inputs has to be a tf.uint8 tensor.')

  module = hub.Module(options.hub_url, trainable=options.trainable)
  height, width = hub.get_expected_image_size(module)

  inputs = tf.image.convert_image_dtype(inputs, tf.float32)

  # Crop and resize images.
  batch = proposals.shape[0]
  max_num_proposals = tf.shape(proposals)[1]

  box_ind = tf.expand_dims(tf.range(batch), axis=-1)
  box_ind = tf.tile(box_ind, [1, max_num_proposals])

  cropped_inputs = tf.image.crop_and_resize(inputs,
                                            boxes=tf.reshape(
                                                proposals, [-1, 4]),
                                            box_ind=tf.reshape(box_ind, [-1]),
                                            crop_size=[height, width])

  outputs = module(cropped_inputs,
                   signature="image_feature_vector",
                   as_dict=True)['default']
  outputs = tf.reshape(outputs, [batch, -1, outputs.shape[-1]])
  return outputs, None
