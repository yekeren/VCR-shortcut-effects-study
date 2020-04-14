from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
from modeling.models import fast_rcnn

from google.protobuf import text_format
from protos import fast_rcnn_pb2


class FastRCNNTest(tf.test.TestCase):

  def test_fast_rcnn_from_detection_checkpoint(self):
    options_str = r"""
      feature_extractor {
        type: 'faster_rcnn_inception_v2'
        first_stage_features_stride: 16
      }
      initial_crop_size: 14
      maxpool_kernel_size: 2
      maxpool_stride: 2
      dropout_keep_prob: 0.5
      dropout_on_feature_map: false
      checkpoint_path: 'data/detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt'
      from_classification_checkpoint: false
    """
    options = text_format.Merge(options_str, fast_rcnn_pb2.FastRCNN())

    inputs = tf.random.uniform(shape=[5, 320, 320, 3], maxval=255)
    proposals = tf.constant([[[0, 0, 1, 1]]] * 5, dtype=tf.float32)
    outputs, init_fn = fast_rcnn.FastRCNN(inputs,
                                          proposals,
                                          options,
                                          is_training=False)
    self.assertAllEqual(outputs.shape, [5, 1, 1024])

    with self.test_session() as sess:
      init_fn(None, sess)
      self.assertEmpty(sess.run(tf.compat.v1.report_uninitialized_variables()))

  def test_fast_rcnn_from_classification_checkpoint(self):
    options_str = r"""
      feature_extractor {
        type: 'faster_rcnn_inception_resnet_v2'
        first_stage_features_stride: 8
      }
      initial_crop_size: 17
      maxpool_kernel_size: 1
      maxpool_stride: 1
      dropout_keep_prob: 0.5
      dropout_on_feature_map: false
      checkpoint_path: 'data/classification/inception_resnet_v2_2016_08_30/inception_resnet_v2_2016_08_30.ckpt'
      from_classification_checkpoint: true
    """
    options = text_format.Merge(options_str, fast_rcnn_pb2.FastRCNN())

    inputs = tf.random.uniform(shape=[5, 320, 320, 3], maxval=255)
    proposals = tf.constant([[[0, 0, 1, 1]]] * 5, dtype=tf.float32)
    outputs, init_fn = fast_rcnn.FastRCNN(inputs,
                                          proposals,
                                          options,
                                          is_training=False)
    self.assertAllEqual(outputs.shape, [5, 1, 1088])

    with self.test_session() as sess:
      init_fn(None, sess)
      self.assertEmpty(sess.run(tf.compat.v1.report_uninitialized_variables()))

if __name__ == '__main__':
  tf.test.main()
