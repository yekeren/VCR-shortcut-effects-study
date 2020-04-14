from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os
import tensorflow as tf
from google.protobuf import text_format
from protos import pipeline_pb2
from modeling import trainer

flags.DEFINE_string('type', None, 'Reserved.')

flags.DEFINE_string('model_dir', None,
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('pipeline_proto', None, 'Path to the pipeline proto file.')

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

  tf.io.gfile.makedirs(FLAGS.model_dir)
  tf.io.gfile.copy(FLAGS.pipeline_proto,
                   os.path.join(FLAGS.model_dir, 'pipeline.pbtxt'),
                   overwrite=True)

  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)
  trainer.train_and_evaluate(pipeline_proto=pipeline_proto,
                             model_dir=FLAGS.model_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_proto')
  app.run(main)
