from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl import app
from absl import flags
from absl import logging

import hashlib
import io
import zipfile
import numpy as np
import PIL.Image
import tensorflow as tf

flags.DEFINE_string('annotations_jsonl_file', 'data/vcr1annots/val.jsonl',
                    'Path to the annotations file in jsonl format.')

FLAGS = flags.FLAGS


def _load_annotations(filename):
  """Loads annotations from file.

  Args:
    filename: Path to the jsonl annotations file.

  Returns:
    A list of python dictionary, each is parsed from a json object.
  """
  with tf.io.gfile.GFile(filename, 'r') as f:
    return [json.loads(x.strip('\n')) for x in f]


def uniq_tags(mixed_caption):
  tags = []
  for ch in mixed_caption:
    if isinstance(ch, list):
      tags.extend(ch)
  return set(tags)


def main(_):
  logging.set_verbosity(logging.INFO)

  # Load annotations.
  annots = _load_annotations(FLAGS.annotations_jsonl_file)
  logging.info('Loaded %i annotations.', len(annots))

  bingo_answer = 0
  bingo_rationale = 0
  bingo_max_answer = 0
  bingo_max_rationale = 0

  for idx, annot in enumerate(annots):
    (question, answer_choices, answer_label, rationale_choices,
     rationale_label) = (annot['question'], annot['answer_choices'],
                         annot['answer_label'], annot['rationale_choices'],
                         annot['rationale_label'])
    answer = answer_choices[answer_label]

    question_tags = uniq_tags(question)
    answer_tags = uniq_tags(answer)

    answer_scores = np.array(
        [len(uniq_tags(x) & question_tags) for x in answer_choices])
    rationale_scores = np.array([
        len(uniq_tags(x) & (question_tags | answer_tags))
        for x in rationale_choices
    ])
    if answer_scores.argmax() == answer_label:
      bingo_answer += 1
    if rationale_scores.argmax() == rationale_label:
      bingo_rationale += 1

    if answer_scores[answer_label] == answer_scores.max():
      bingo_max_answer += 1
    if rationale_scores[rationale_label] == rationale_scores.max():
      bingo_max_rationale += 1

  print('Accuracy (answer): %.3lf' % (bingo_answer * 1.0 / len(annots)))
  print('Accuracy (rationale): %.3lf' % (bingo_rationale * 1.0 / len(annots)))
  print('Ratio (answer_max): %.3lf' % (bingo_max_answer * 1.0 / len(annots)))
  print('Ratio (rationale_max): %.3lf' %
        (bingo_max_rationale * 1.0 / len(annots)))

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
