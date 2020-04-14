from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format

from modeling.utils import optimization
from protos import optimizer_pb2


class OptimizationTest(tf.test.TestCase):

  def test_create_optimizer(self):
    options_str = "adagrad{}"
    options = text_format.Merge(options_str, optimizer_pb2.Optimizer())
    opt = optimization.create_optimizer(options)
    self.assertIsInstance(opt, tf.compat.v1.train.AdagradOptimizer)

    options_str = "rmsprop{}"
    options = text_format.Merge(options_str, optimizer_pb2.Optimizer())
    opt = optimization.create_optimizer(options)
    self.assertIsInstance(opt, tf.compat.v1.train.RMSPropOptimizer)

    options_str = "adam{}"
    options = text_format.Merge(options_str, optimizer_pb2.Optimizer())
    opt = optimization.create_optimizer(options)
    self.assertIsInstance(opt, tf.compat.v1.train.AdamOptimizer)


if __name__ == '__main__':
  tf.test.main()
