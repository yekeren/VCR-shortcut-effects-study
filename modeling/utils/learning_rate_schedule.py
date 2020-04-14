from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from protos import hyperparams_pb2
from protos import learning_rate_schedule_pb2


def create_learning_rate_schedule(options):
  """Builds learning_rate_schedule from options.

  Args:
    options: An instance of
      learning_rate_schedule_pb2.LearningRateSchedule.

  Returns:
    A tensorflow LearningRateSchedule instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, learning_rate_schedule_pb2.LearningRateSchedule):
    raise ValueError(
        'The options has to be an instance of LearningRateSchedule.')

  oneof = options.WhichOneof('learning_rate_schedule')

  if 'piecewise_constant_decay' == oneof:
    options = options.piecewise_constant_decay
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=options.boundaries[:], values=options.values[:])

  if 'exponential_decay' == oneof:
    options = options.exponential_decay
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=options.initial_learning_rate,
        decay_steps=options.decay_steps,
        decay_rate=options.decay_rate,
        staircase=options.staircase)

  raise ValueError('Invalid learning_rate_schedule: {}.'.format(oneof))
