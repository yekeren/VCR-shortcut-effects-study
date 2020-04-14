from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modeling.utils import box_ops as ops

tf.compat.v1.enable_eager_execution()


class BoxOpsTest(tf.test.TestCase):

  def test_calc_area(self):
    self.assertAllClose(
        ops.calc_area([[0.0, 0.0, 1.0, 1.0], [0.25, 0.25, 0.75, 0.75]]),
        [1.0, 0.25])


if __name__ == '__main__':
  tf.test.main()
