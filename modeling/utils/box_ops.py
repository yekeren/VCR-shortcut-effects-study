from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def calc_area(box):
  """Calculates the area.

  Args:
    box: A [i1,...,iN,  4] float tensor, denoting the (ymin, xmin, ymax, xmax).

  Returns:
    area: Box areas, a [i1,...,iN] float tensor.
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  return (ymax - ymin) * (xmax - xmin)
