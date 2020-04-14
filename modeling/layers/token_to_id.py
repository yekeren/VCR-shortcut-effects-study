from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf


class TokenToIdLayer(tf.keras.layers.Layer):
  """TokenToId layer."""

  def __init__(self, vocab_file, unk_token_id, **kwargs):
    """Initializes the tf.lookup.StaticHashTable.

    Args:
      vocab_file: Path to the vocabulary file, in which each line is a token
        and the line number is the token id.
      unk_token_id: A number, id of the [UNK] token.
    """
    super(TokenToIdLayer, self).__init__(**kwargs)
    initializer = tf.lookup.TextFileInitializer(
        vocab_file, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
        tf.lookup.TextFileIndex.LINE_NUMBER)
    self.table = tf.lookup.StaticHashTable(initializer,
                                           default_value=unk_token_id,
                                           name='token_to_id')

  def call(self, inputs):
    """Converts the inputs to token_ids.

    Args:
      inputs: A tf.string tensor.

    Returns:
      A tf.int32 tensor which has the same shape as the inputs.
    """
    return tf.dtypes.cast(self.table.lookup(inputs), tf.int32)
