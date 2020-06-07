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

from bertviz import head_view


flags.DEFINE_string('model_dir',
                    'logs.final.eval/b2t2_entropy0.00001_res101_f00001/ckpts',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('pipeline_proto',
                    'logs.final.eval/b2t2_entropy0.00001_res101_f00001/pipeline.pbtxt',
                    'Path to the pipeline proto file.')

flags.DEFINE_integer('num_bert_layers', 12, 'Number of BERT layers.')

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


def _load_vocabulary(filename):
  """Loads vocabulary file.
        
  Args:
    filename: Path to the vocabulary file.
                    
  Returns:
    A list of python strings.
  """
  with tf.io.gfile.GFile(filename, 'r') as fp:
    return [x.strip('\n') for x in fp]


def main(_):
  logging.set_verbosity(logging.DEBUG)

  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  pipeline_proto.eval_reader.vcr_reader.batch_size = 1

  num_layers = FLAGS.num_bert_layers

  def attn_processor_fn(tf_graph):
    """Gets attention tensors by name.
    Args:
      tf_graph: tf.Graph instance.
      num_layers: Number of attention layers in BERT.
    """
    # B = batch size (number of sequences)
    # F = `from_tensor` sequence length
    # T = `to_tensor` sequence length
    # N = `num_attention_heads`
    # H = `size_per_head`

    # `attention_probs` = [B, N, F, T]
    predictions = {}
    for i in range(num_layers):
      predictions['bert_attn_%i' % i] = tf_graph.get_tensor_by_name(
          'bert/encoder/layer_%s/attention/self/Softmax:0' % i)
    return predictions

  vocab = _load_vocabulary(pipeline_proto.eval_reader.vcr_reader.vocab_file)

  count = 0
  params = {'create_additional_predictions': attn_processor_fn}

  entropy_list = []
  for example_id, example in enumerate(
          trainer.predict(pipeline_proto, FLAGS.model_dir, params=params)):

    detection_classes = []
    for i, x in enumerate(example['detection_classes'][0]):
      name = vocab[x]
      if name == '[unused400]':
        name = '[IMAGE]'
      detection_classes.append(name)

    answer_choices = [vocab[x] for x in example['mixed_answer_choices'][0][0]]
    tokens = ['[CLS]'] + detection_classes + ['[SEP]'] + answer_choices + ['[SEP]']

    # N = Number of layers.
    # H = Number of heads.
    # F = From tensor.
    # T = To tensor.

    # `attns` = [N, H, F, T]
    attns = np.concatenate([example['bert_attn_%i' % i]
                            for i in range(num_layers)], 0)
    log_attns = np.log(attns + 1e-8)

    # `entropy` = [N, H]
    entropy = - (attns * log_attns).sum(-1).mean(-1)
    entropy_list.append(entropy)

    # Counting.
    count += len(example['annot_id'])
    if count % 100 == 0:
      print(count)

  entropy = np.stack(entropy_list, -1).mean(-1)
  for i in range(len(entropy)):
    line = ','.join(['%.3lf' % x for x in entropy[i]])
    print(line)

  logging.info('Done')


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_proto')
  app.run(main)
