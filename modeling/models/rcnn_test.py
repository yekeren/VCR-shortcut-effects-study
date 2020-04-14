from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
from modeling.models import rcnn

from google.protobuf import text_format
from protos import rcnn_pb2


class RCNNTest(tf.test.TestCase):

  def test_rcnn(self):
    options_str = r"""
      feature_extractor_name: 'inception_v4'
      feature_extractor_scope: 'InceptionV4'
      feature_extractor_endpoint: 'PreLogitsFlatten'
      feature_extractor_checkpoint: 'data/classification/inception_v4_2016_09_09/inception_v4.ckpt'
    """
    options = text_format.Merge(options_str, rcnn_pb2.RCNN())

    inputs = tf.random.uniform(shape=[5, 320, 320, 3], maxval=255, dtype=tf.int32)
    inputs = tf.cast(inputs, dtype=tf.uint8)

    proposals = tf.constant([[[0, 0, 1, 1], [0.25, 0.25, 0.75, 0.75]]] * 5,
                            dtype=tf.float32)

    outputs, init_fn = rcnn.RCNN(inputs, proposals, options, is_training=False)
    self.assertAllEqual(outputs.shape, [5, 2, 1536])

    with self.test_session() as sess:
      init_fn(None, sess)
      self.assertEmpty(sess.run(tf.compat.v1.report_uninitialized_variables()))


if __name__ == '__main__':
  tf.test.main()
