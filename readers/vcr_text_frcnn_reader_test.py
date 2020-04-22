from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf

import reader
from vcr_text_frcnn_reader import InputFields

from google.protobuf import text_format
from protos import reader_pb2

tf.compat.v1.enable_eager_execution()


class VCRTextImageReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    options_str = r"""
      vcr_text_frcnn_reader {
        input_pattern: "output/uncased/VCR-text_and_frcnn/val.record-*-of-00005"
        shuffle_buffer_size: 10
        interleave_cycle_length: 1
        batch_size: 60
        prefetch_buffer_size: 8000
        frcnn_feature_dims: 1536
        num_parallel_calls: 10
      }
    """
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=False)()
    for elem in dataset.take(10000):
      for key, value in elem.items():
        import pdb
        pdb.set_trace()
        j = 1
        # logging.info('=' * 64)
        # logging.info(key)
        # logging.info(value)


if __name__ == '__main__':
  tf.test.main()
