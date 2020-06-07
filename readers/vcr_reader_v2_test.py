from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf

import reader
from vcr_fields import InputFields

from google.protobuf import text_format
from protos import reader_pb2

tf.compat.v1.enable_eager_execution()


class VCRReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    options_str = r"""
      vcr_reader_v2 {
        input_pattern: "output/uncased/VCR-RAW/val_v2.record-00000-of-00005"
        shuffle_buffer_size: 10
        interleave_cycle_length: 1
        batch_size: 8
        prefetch_buffer_size: 8000
        detection_vocab_file: "data/detection.vocab"
        vocab_file: "data/bert/tf1.x/BERT-Base/vocab.txt"
        out_of_vocabulary_token_id: 100
      }
    """
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=False)()
    for elem in dataset.take(1):
      import pdb
      pdb.set_trace()
      for key, value in elem.items():
        logging.info('%s: %s', key, value.shape)
      logging.info('Examples:')
      logging.info('answer_choices: %s', elem['answer_choices'][0])
      logging.info('answer_choices_tag: %s', elem['answer_choices_tag'][0])
      logging.info('answer_choices_len: %s', elem['answer_choices_len'][0])
      logging.info('rationale_choices: %s', elem['rationale_choices'][0])
      logging.info('rationale_choices_tag: %s', elem['rationale_choices_tag'][0])
      logging.info('rationale_choices_len: %s', elem['rationale_choices_len'][0])


if __name__ == '__main__':
  tf.test.main()
