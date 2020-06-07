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

flags.DEFINE_string('model_dir', None,
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('pipeline_proto', None, 'Path to the pipeline proto file.')

flags.DEFINE_bool('rationale', False, 'If true, evaluate rationale results.')

flags.DEFINE_string('input_pattern', None,
                    'If specified, replace the input files.')

FLAGS = flags.FLAGS


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
  pipeline_proto.eval_reader.vcr_reader_v2.batch_size = 9

  eval_name = None
  if FLAGS.input_pattern is not None:
    eval_name = os.path.basename(FLAGS.input_pattern).split('.')[0]
    del pipeline_proto.eval_reader.vcr_reader_v2.input_pattern[:]
    pipeline_proto.eval_reader.vcr_reader_v2.input_pattern.append(
        FLAGS.input_pattern)

  count = 0
  annot_ids, predictions = [], []
  for example_id, example in enumerate(
      trainer.predict(pipeline_proto, FLAGS.model_dir)):
    count += len(example['annot_id'])
    annot_ids.append(example['annot_id'])
    predictions.append(example['answer_prediction'])
    logging.info('Predicted %s', count)

  annot_ids = np.concatenate(annot_ids, 0)
  predictions = np.concatenate(predictions, 0)

  if eval_name is not None:
    eval_npy = os.path.join(FLAGS.model_dir, '%s.npy' % eval_name)
    with open(eval_npy, 'wb') as f:
      np.save(f, annot_ids)
      np.save(f, predictions)
    logging.info('Results are written to %s.', eval_npy)

  logging.info('Evaluated %i examples.', count)


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_proto')
  app.run(main)
