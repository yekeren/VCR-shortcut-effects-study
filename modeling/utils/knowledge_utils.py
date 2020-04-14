from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf


def create_tf_lookup_table(filename,
                           key_index=0,
                           value_index=1,
                           delimiter='\t',
                           default_value=''):
  """Helper function for creating a hash table.

  Args:
    filename: Path to the file containing key-value data.
    key_index: Column id of the key in the file.
    value_index: Column id of the value in the file.
    delimiter: Fields delimiter.
    default_value: Default value.
  
  Returns:
    A tf.contrib.lookup.HashTable instance.
  """
  initializer = tf.contrib.lookup.TextFileInitializer(filename,
                                                      key_dtype=tf.string,
                                                      key_index=key_index,
                                                      value_dtype=tf.string,
                                                      value_index=value_index,
                                                      delimiter=delimiter)
  return tf.contrib.lookup.HashTable(initializer, default_value=default_value)


class KnowledgeTable(object):

  def __init__(self,
               filename,
               key_index=0,
               value_index=1,
               delimiter='\t',
               default_value=''):

    self._default_value = default_value
    self._word_to_definition = create_tf_lookup_table(filename, key_index,
                                                      value_index, delimiter,
                                                      default_value)

  def query(self, keys):
    """Retrieves values based on keys.

    Args:
      keys: A string tensor of any shape, each token is a query.

    Returns:
      values: Retrieved contents that are tokenized.
      values_len: Lengths of the retrieved contents.
    """
    default_value = self._default_value

    # Query.
    values = self._word_to_definition.lookup(keys)
    values_shape = tf.shape(values)

    # Tokenization.
    values_flattened = tf.strings.split(tf.reshape(values, [-1]), sep=' ')
    values_flattened = tf.sparse_tensor_to_dense(values_flattened,
                                                 default_value=default_value)
    values_len = tf.count_nonzero(tf.not_equal(values_flattened, default_value),
                                  axis=-1,
                                  dtype=tf.int32)

    # Reshape back.
    max_def_len = tf.shape(values_flattened)[-1]
    values_reshaped = tf.reshape(values_flattened,
                                 tf.concat([values_shape, [max_def_len]], -1))
    values_len_reshaped = tf.reshape(values_len, values_shape)
    return values_reshaped, values_len_reshaped
