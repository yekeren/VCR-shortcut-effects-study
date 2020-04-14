from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_EPSILON = 1e-10
_INF = 1e10


def bilinear(x, y, inherit_from_slim_arg_scope=False):
  """Auxilary function for bilinear attention.

  Args:
    x: a [batch, seq_x_size, dims] float tensor.
    y: a [batch, seq_y_size, dims] float tensor.

  Returns:
    xwy: A [batch, seq_x_size, seq_y_size] float tensor.
  """
  if inherit_from_slim_arg_scope:
    xw = tf.contrib.layers.fully_connected(x,
                                           num_outputs=y.get_shape()[-1].value,
                                           activation_fn=None)
  else:
    xw = tf.contrib.layers.fully_connected(x,
                                           num_outputs=y.get_shape()[-1].value,
                                           normalizer_fn=lambda x: x,
                                           activation_fn=None)
  xwy = tf.linalg.matmul(xw, y, transpose_b=True)

  return xwy
