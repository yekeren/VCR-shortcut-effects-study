from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format

from modeling.utils import learning_rate_schedule
from protos import learning_rate_schedule_pb2


class LearningRateScheduleTest(tf.test.TestCase):

  def test_learning_rate_schedule(self):
    options_str = r"""
    piecewise_constant_decay{
      values: 0.001
    }"""
    options = text_format.Merge(
        options_str, learning_rate_schedule_pb2.LearningRateSchedule())
    schedule = learning_rate_schedule.create_learning_rate_schedule(options)
    self.assertIsInstance(
        schedule, tf.keras.optimizers.schedules.PiecewiseConstantDecay)

    options_str = r"""
    exponential_decay {
      initial_learning_rate: 0.001
      decay_steps: 1000
      decay_rate: 1.0
    }"""
    options = text_format.Merge(
        options_str, learning_rate_schedule_pb2.LearningRateSchedule())
    schedule = learning_rate_schedule.create_learning_rate_schedule(options)
    self.assertIsInstance(
        schedule, tf.keras.optimizers.schedules.ExponentialDecay)

    options_str = r"""
    polynomial_decay {
      initial_learning_rate: 0.001
      decay_steps: 1000
      end_learning_rate: 0.0001
      power: 1.0
      cycle: true
    }"""
    options = text_format.Merge(
        options_str, learning_rate_schedule_pb2.LearningRateSchedule())
    schedule = learning_rate_schedule.create_learning_rate_schedule(options)
    self.assertIsInstance(
        schedule, tf.keras.optimizers.schedules.PolynomialDecay)


if __name__ == '__main__':
  tf.test.main()
