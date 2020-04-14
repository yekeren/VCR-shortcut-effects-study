from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from tempfile import NamedTemporaryFile

import tensorflow as tf
from modeling.utils import knowledge_utils

tf.compat.v1.enable_eager_execution()


class KnowledgeUtilsTest(tf.test.TestCase):

  def setUp(self):
    self._key1 = 'blizzard'
    self._key2 = 'masterpiece'
    self._value1 = 'a storm with widespread snowfall accompanied by strong winds'
    self._value2 = 'the most outstanding work of a creative artist or craftsman'

  def _create_file(self):
    """Creates a temporary file."""
    f = NamedTemporaryFile(delete=False)
    f.write('\n'.join([
        '%s\t%s' % (self._key1, self._value1),
        '%s\t%s' % (self._key2, self._value2)
    ]).encode('ascii'))
    temp_path = f.name
    f.close()
    return temp_path

  def test_create_tf_lookup_table(self):
    table = knowledge_utils.create_tf_lookup_table(self._create_file(),
                                                   default_value='')
    self.assertAllEqual(
        tf.convert_to_tensor([
            self._value1,
            self._value2,
            '',
        ]),
        table.lookup(
            tf.convert_to_tensor([self._key1, self._key2,
                                  'out-of-vocabulary'])).numpy())

  def test_query(self):
    table = knowledge_utils.KnowledgeTable(self._create_file(),
                                           default_value='')
    values, values_len = table.query(
        tf.convert_to_tensor([['blizzard', 'out-of-vocabulary'],
                              ['out-of-vocabulary', 'masterpiece']]))

    expected_value1 = self._value1.split(' ')
    expected_value2 = self._value2.split(' ')
    max_content_len = max(len(expected_value1), len(expected_value2))

    expected_padding = [''] * max_content_len
    expected_value1 += [''] * (max_content_len - len(expected_value1))
    expected_value2 += [''] * (max_content_len - len(expected_value2))

    self.assertAllEqual(values_len, [[9, 0], [0, 10]])
    self.assertAllEqual(
        values,
        tf.convert_to_tensor([[expected_value1, expected_padding],
                              [expected_padding, expected_value2]]))


if __name__ == '__main__':
  tf.test.main()
