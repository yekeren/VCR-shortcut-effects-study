from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from protos import pipeline_pb2
from modeling import trainer
from readers import reader
from readers.vcr_text_only_reader import InputFields
from readers.vcr_text_only_reader import NUM_CHOICES
from vcr import builder
from protos import pipeline_pb2
import json

flags.DEFINE_string('model_dir', None,
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('pipeline_proto', None, 'Path to the pipeline proto file.')

FLAGS = flags.FLAGS

FIELD_ANSWER_PREDICTION = 'answer_prediction'


def _load_pipeline_proto(filename):
  """Loads pipeline proto from file.

  Args:
    filename: Path to the pipeline config file.

  Returns:
    An instance of pipeline_pb2.Pipeline.
  """
  with tf.io.gfile.GFile(filename, 'r') as fp:
    return text_format.Merge(fp.read(), pipeline_pb2.Pipeline())


def main(_):
  logging.set_verbosity(logging.DEBUG)

  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  for example in trainer.predict(pipeline_proto, FLAGS.model_dir):
    batch_size = len(example['question'])
    for i in range(batch_size):
      print('#' * 128)
      print(example['question'][i])
      print(example['answer_label'][i])
      import pdb
      pdb.set_trace()
      for j in range(4):
        sentence = []
        for token, indicator in zip(example['answer_choices'][i, j],
                                    example['shortcut_mask'][i, j]):
          if not indicator:
            sentence.append(token.decode('utf8') + '[REMOVE]')
          else:
            sentence.append(token.decode('utf8'))
        print(' '.join(sentence))
        print(example['answer_logits'][i][j].tolist())
        print(example['a_soft_sample'][i][j].tolist())
      print()


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_proto')
  app.run(main)
