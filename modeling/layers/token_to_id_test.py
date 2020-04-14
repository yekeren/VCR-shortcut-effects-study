from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tempfile import NamedTemporaryFile
from modeling.layers import token_to_id

tf.compat.v1.enable_eager_execution()


class TokenToIdLayerTest(tf.test.TestCase):

  def _create_temp_vocab_file(self):
    """Creates a temporary vocabulary file."""
    f = NamedTemporaryFile(delete=False)
    f.write('\n'.join(
        ['this', 'is', 'hello', 'world', '[UNK]', 'program', ',',
         '!']).encode('ascii'))
    temp_path = f.name
    f.close()
    return temp_path

  def test_token_to_id(self):
    vocab_file = self._create_temp_vocab_file()
    test_layer = token_to_id.TokenToIdLayer(vocab_file, unk_token_id=4)

    output = test_layer(tf.convert_to_tensor(['hello', ',', 'world', '!']))
    self.assertAllEqual(output, [2, 6, 3, 7])
    output = test_layer(tf.convert_to_tensor(['hell', ',', 'world', '!!']))
    self.assertAllEqual(output, [4, 6, 3, 4])

    os.unlink(vocab_file)
    self.assertFalse(os.path.exists(vocab_file))

  def test_token_to_id_2d(self):
    vocab_file = self._create_temp_vocab_file()
    test_layer = token_to_id.TokenToIdLayer(vocab_file, unk_token_id=4)

    output = test_layer(
        tf.convert_to_tensor([['hello', ',', 'world', '!'],
                              ['hell', ',', 'world', '!!']]))
    self.assertAllEqual(output, [[2, 6, 3, 7], [4, 6, 3, 4]])

    os.unlink(vocab_file)
    self.assertFalse(os.path.exists(vocab_file))

  def test_token_to_id_bert(self):
    test_layer = token_to_id.TokenToIdLayer(
        'data/bert/keras/cased_L-12_H-768_A-12/vocab.txt', unk_token_id=100)

    output = test_layer(tf.convert_to_tensor(['hello', ',', 'world', '!']))
    self.assertAllEqual(output, [19082, 117, 1362, 106])

    output = test_layer(tf.convert_to_tensor(['hello', ',', 'world', '!!']))
    self.assertAllEqual(output, [19082, 117, 1362, 100])


if __name__ == '__main__':
  tf.test.main()
