from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
from protos import rnn_pb2


def BiLSTM(sequence_feature, sequence_length, options, is_training=True):
  """Encodes sequence using BiLSTM model."""

  def _lstm_cell():
    """Returns the basic LSTM cell."""
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=options.hidden_units)
    if is_training:
      cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
          cell,
          input_keep_prob=options.input_keep_prob,
          output_keep_prob=options.output_keep_prob,
          state_keep_prob=options.state_keep_prob)
    return cell

  fw_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
      [_lstm_cell() for _ in range(options.number_of_layers)])
  bw_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
      [_lstm_cell() for _ in range(options.number_of_layers)])

  outputs, states = tf.compat.v1.nn.bidirectional_dynamic_rnn(
      cell_fw=fw_rnn_cell,
      cell_bw=bw_rnn_cell,
      inputs=sequence_feature,
      sequence_length=sequence_length,
      dtype=tf.float32)

  outputs = tf.concat(outputs, axis=-1)

  states_list = []
  for forward_or_backword_states in states:
    for layer_state in forward_or_backword_states:
      states_list.extend([layer_state.c, layer_state.h])
  states = tf.concat(states_list, axis=-1)
  return outputs, states


_RNNS = {'bilstm': BiLSTM}


def RNN(sequence_feature, sequence_length, options, is_training=True):
  """Runs RNN encoding on the sequence.

  Args:
    sequence_feature: A [batch, max_sequence_length, input_dims] float tensor.
    sequence_length: A [batch] int tensor.
    options: A rnn_pb2.RNN proto.
    is_training: If true, the model shall be executed in training mode.

  Returns:
    sequence_embedding: A [batch, max_sequence_length, output_dims] float tensor.
    sequence_feature: A [batch, output_dims] float tensor.
  """
  if not isinstance(options, rnn_pb2.RNN):
    raise ValueError('The options has to be a rnn_pb2.RNN proto!')

  rnn_oneof = options.WhichOneof('rnn_oneof')
  if not rnn_oneof in _RNNS:
    raise ValueError('Invalid rnn %s!' % rnn_oneof)

  return _RNNS[rnn_oneof](sequence_feature, sequence_length,
                          getattr(options, rnn_oneof), is_training)
