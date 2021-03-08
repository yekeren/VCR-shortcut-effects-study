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

from modeling import trainer
from readers import reader
from readers.vcr_fields import InputFields
from readers.vcr_fields import NUM_CHOICES

from models import builder
from protos import pipeline_pb2
import json

flags.DEFINE_string('model_dir', 'logs.adv_gen/gen_adv_answer',
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('pipeline_proto',
                    'logs.adv_gen/gen_adv_answer/pipeline.pbtxt',
                    'Path to the pipeline proto file.')

# flags.DEFINE_string(
#     'output_jsonl_file',
#     'data/modified_annots/val_answer_shortcut/val_answer_shortcut.jsonl.new',
#     'Path to the output jsonl file.')
flags.DEFINE_string(
    'output_jsonl_file',
    'data/modified_annots/train_answer_shortcut/train_answer_shortcut.jsonl.new2',
    'Path to the output jsonl file.')

################################
# Rationale
################################
# flags.DEFINE_string('model_dir', 'logs.adv_gen/gen_adv_rationale',
#                     'Path to the directory which holds model checkpoints.')
# 
# flags.DEFINE_string('pipeline_proto',
#                     'logs.adv_gen/gen_adv_rationale/pipeline.pbtxt',
#                     'Path to the pipeline proto file.')
# 
# flags.DEFINE_string(
#     'output_jsonl_file',
#     'data/modified_annots_rationale/val_answer_shortcut/val_answer_shortcut_rationale.jsonl.new',
#     'Path to the output jsonl file.')
# 
flags.DEFINE_string('vocab_file', 'data/bert/tf1.x/BERT-Base/vocab.txt',
                    'Path to the vocabulary file.')

flags.DEFINE_bool('rationale', False, 'If true, generate rationale data.')

FLAGS = flags.FLAGS

PAD_ID = 0
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
IMG_ID = 405

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


def _load_vocab_file(filename):
  """Loads vocabulary from file.

  Args:
    filename: Path to the vocabulary file.

  Returns:
    A list of strings.
  """
  with tf.io.gfile.GFile(filename, 'r') as fp:
    return [x.strip('\n') for x in fp.readlines()]


def pack_tensor_values(choices, choices_len, vocab):
  """Prints tensor values."""
  results = []
  for choice, choice_len in zip(choices, choices_len):
    results.append([vocab[x] for x in choice[:choice_len]])
  return results


def main(_):
  logging.set_verbosity(logging.DEBUG)

  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)
  vocab = _load_vocab_file(FLAGS.vocab_file)

  # Get `next_examples_ts' tensor.
  if 'train' in FLAGS.output_jsonl_file:
    input_fn = reader.get_input_fn(pipeline_proto.train_reader,
                                   is_training=False)
  else:
    input_fn = reader.get_input_fn(pipeline_proto.eval_reader,
                                   is_training=False)

  iterator = input_fn().make_initializable_iterator()
  next_examples_ts = iterator.get_next()

  # Build model that takes placeholder as inputs, and predicts the logits.
  frcnn_dims = pipeline_proto.eval_reader.vcr_text_frcnn_reader.frcnn_feature_dims
  (label_pl, choices_pl, choices_tag_pl,
   choices_len_pl) = (tf.placeholder(tf.int32, [1]),
                      tf.placeholder(tf.int32, [1, NUM_CHOICES, None]),
                      tf.placeholder(tf.int32, [1, NUM_CHOICES, None]),
                      tf.placeholder(tf.int32, [1, NUM_CHOICES]))
  (num_detections_pl, detection_boxes_pl, detection_classes_pl,
   detection_scores_pl,
   detection_features_pl) = (tf.placeholder(tf.int32, [1]),
                             tf.placeholder(tf.float32, [1, None, 4]),
                             tf.placeholder(tf.int32, [1, None]),
                             tf.placeholder(tf.float32, [1, None]),
                             tf.placeholder(tf.float32, [1, None, frcnn_dims]))

  model = builder.build(pipeline_proto.model, is_training=False)
  logits_ts = model.predict({
      InputFields.num_detections: num_detections_pl,
      InputFields.detection_boxes: detection_boxes_pl,
      InputFields.detection_classes: detection_classes_pl,
      InputFields.detection_scores: detection_scores_pl,
      InputFields.detection_features: detection_features_pl,
      model._field_choices: choices_pl,
      model._field_choices_tag: choices_tag_pl,
      model._field_choices_len: choices_len_pl,
  })[FIELD_ANSWER_PREDICTION]

  losses_ts = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_ts,
                                                      labels=tf.one_hot(
                                                          label_pl,
                                                          depth=NUM_CHOICES))
  saver = tf.train.Saver()

  # Find the latest checkpoint file.
  ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  assert ckpt_path is not None

  def _calc_score_and_loss(choices, choice_tag, choices_len, label,
                           num_detections, detection_boxes, detection_clases,
                           detection_scores, detection_features):
    """Get the VCR matching scores and losses."""
    (scores, losses) = sess.run(
        [logits_ts, losses_ts],
        feed_dict={
            label_pl: np.expand_dims(label, 0),
            choices_pl: np.expand_dims(choices, 0),
            choices_tag_pl: np.expand_dims(choices_tag, 0),
            choices_len_pl: np.expand_dims(choices_len, 0),
            num_detections_pl: np.expand_dims(num_detections, 0),
            detection_boxes_pl: np.expand_dims(detection_boxes, 0),
            detection_classes_pl: np.expand_dims(detection_clases, 0),
            detection_scores_pl: np.expand_dims(detection_scores, 0),
            detection_features_pl: np.expand_dims(detection_features, 0),
        })
    return scores[0], losses[0]

  # Run inference using the pretrained Bert model.
  with tf.Session() as sess, open(FLAGS.output_jsonl_file, 'w') as output_fp:
    sess.run(iterator.initializer)
    sess.run(tf.tables_initializer())
    saver.restore(sess, ckpt_path)
    logging.info('Restore from %s.', ckpt_path)

    batch_id = 0
    while True:
      batch_id += 1
      try:
        inputs_batched = sess.run(next_examples_ts)
        batch_size = len(inputs_batched[InputFields.annot_id])

        masks = np.array([[MASK_ID], [MASK_ID], [MASK_ID], [MASK_ID]])
        ones = np.array([[1], [1], [1], [1]])

        for example_id in range(batch_size):

          (annot_id, choices, choices_tag, choices_len, label) = (
              inputs_batched[InputFields.annot_id][example_id].decode('utf8'),
              inputs_batched[model._field_choices][example_id],
              inputs_batched[model._field_choices_tag][example_id],
              inputs_batched[model._field_choices_len][example_id],
              inputs_batched[model._field_label][example_id])
          (num_detections, detection_boxes, detection_clases, detection_scores,
           detection_features) = (
               inputs_batched[InputFields.num_detections][example_id],
               inputs_batched[InputFields.detection_boxes][example_id],
               inputs_batched[InputFields.detection_classes][example_id],
               inputs_batched[InputFields.detection_scores][example_id],
               inputs_batched[InputFields.detection_features][example_id])

          # Scores of the original choices.
          orig_scores, orig_losses = _calc_score_and_loss(
              choices, choices_tag, choices_len, label, num_detections,
              detection_boxes, detection_clases, detection_scores,
              detection_features)

          # Adversarial atacking.
          max_losses = np.zeros(NUM_CHOICES)
          max_losses_choices = choices

          if FLAGS.rationale:
            sep_pos = np.where(choices == SEP_ID)[1].take([1, 3, 5, 7])
          else:
            sep_pos = np.where(choices == SEP_ID)[1]

          result_losses = [[] for _ in range(4)]
          result_tokens = [[] for _ in range(4)]

          for pos_id in range(sep_pos.min() + 1, choices_len.max()):
            # Compute the new losses.
            new_choices = np.concatenate(
                [choices[:, :pos_id], masks, choices[:, pos_id + 1:]], -1)
            new_choices_tag = np.concatenate(
                [choices_tag[:, :pos_id], -ones, choices_tag[:, pos_id + 1:]],
                -1)
            scores, losses = _calc_score_and_loss(
                new_choices, new_choices_tag, choices_len, label,
                num_detections, detection_boxes, detection_clases,
                detection_scores, detection_features)

            # Update the maximum values.
            token_id = choices[:, pos_id]
            is_valid = np.logical_not(
                np.logical_or(
                    token_id == PAD_ID,
                    np.logical_or(token_id == CLS_ID, token_id == SEP_ID)))

            for choice_id in range(4):
              if is_valid[choice_id]:
                result_losses[choice_id].append(
                    round(float(losses[choice_id]), 4))
                result_tokens[choice_id].append(
                    vocab[choices[choice_id][pos_id]])

            # Maximize loss.
            adversarial_select_cond = np.logical_and(losses > max_losses,
                                                     is_valid)
            max_losses_choices = np.where(
                np.expand_dims(adversarial_select_cond, -1), new_choices,
                max_losses_choices)
            max_losses = np.maximum(max_losses, losses)

          #END: for pos_id in range(sep_pos.min() + 1, choices_len.max()):

          choices = pack_tensor_values(choices, choices_len, vocab)
          adversarial_choices = pack_tensor_values(max_losses_choices,
                                                   choices_len, vocab)

          output_annot = {
              'annot_id': annot_id,
              'label': int(label),
              'choices': choices,
              'adversarial_choices': adversarial_choices,
              'result_losses': result_losses,
              'result_tokens': result_tokens,
          }
          # print(label)
          # for i in range(4):
          #   print(choices[i])
          #   print(adversarial_choices[i])
          output_fp.write(json.dumps(output_annot) + '\n')

        if batch_id % 10 == 0:
          logging.info('batch_id=%i', batch_id)

      except tf.errors.OutOfRangeError as ex:
        logging.info('Done!')
        break

  output_fp.close()


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_proto')
  app.run(main)
