from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from protos import hyperparams_pb2
from protos import optimizer_pb2


def create_optimizer(options, learning_rate=0.1):
  """Builds optimizer from options.

  Args:
    options: An instance of optimizer_pb2.Optimizer.
    learning_rate: A scalar tensor denoting the learning rate.

  Returns:
    A tensorflow optimizer instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, optimizer_pb2.Optimizer):
    raise ValueError('The options has to be an instance of Optimizer.')

  optimizer = options.WhichOneof('optimizer')
  options = getattr(options, optimizer)

  if 'adagrad' == optimizer:
    return tf.compat.v1.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=options.initial_accumulator_value)

  if 'rmsprop' == optimizer:
    return tf.compat.v1.train.RMSPropOptimizer(learning_rate,
                                               decay=options.decay,
                                               momentum=options.momentum,
                                               epsilon=options.epsilon,
                                               centered=options.centered)

  if 'adam' == optimizer:
    return tf.compat.v1.train.AdamOptimizer(learning_rate,
                                            beta1=options.beta1,
                                            beta2=options.beta2,
                                            epsilon=options.epsilon)

  raise ValueError('Invalid optimizer: {}.'.format(optimizer))
