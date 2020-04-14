from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
import tf_slim as slim
from slim.nets import nets_factory
from slim.preprocessing import preprocessing_factory

from protos import rcnn_pb2


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
  if not isinstance(options, rcnn_pb2.RCNN):
    raise ValueError('The options has to be a rcnn_pb2.RCNN proto!')
  if inputs.dtype != tf.uint8:
    raise ValueError('The inputs has to be a tf.uint8 tensor.')

  net_fn = nets_factory.get_network_fn(name=options.feature_extractor_name,
                                       num_classes=1001)
  default_image_size = getattr(net_fn, 'default_image_size', 224)

  # Preprocess image.
  preprocess_fn = preprocessing_factory.get_preprocessing(
      options.feature_extractor_name, is_training=False)
  inputs = preprocess_fn(inputs,
                         output_height=None,
                         output_width=None,
                         crop_image=False)

  # Crop and resize images.
  batch = proposals.shape[0]
  max_num_proposals = tf.shape(proposals)[1]

  box_ind = tf.expand_dims(tf.range(batch), axis=-1)
  box_ind = tf.tile(box_ind, [1, max_num_proposals])

  cropped_inputs = tf.image.crop_and_resize(
      inputs,
      boxes=tf.reshape(proposals, [-1, 4]),
      box_ind=tf.reshape(box_ind, [-1]),
      crop_size=[default_image_size, default_image_size])

  # Run CNN.
  _, end_points = net_fn(cropped_inputs)
  outputs = end_points[options.feature_extractor_endpoint]
  outputs = tf.reshape(outputs, [batch, max_num_proposals, -1])

  init_fn = slim.assign_from_checkpoint_fn(
      options.feature_extractor_checkpoint,
      slim.get_model_variables(options.feature_extractor_scope))

  def _init_from_ckpt_fn(_, sess):
    return init_fn(sess)

  return outputs, _init_from_ckpt_fn
