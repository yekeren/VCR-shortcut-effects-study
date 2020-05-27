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

from bert import tokenization

flags.DEFINE_string('annotations_jsonl_file', 'data/vcr1annots/val.jsonl',
                    'Path to the annotations file in jsonl format.')

flags.DEFINE_string(
    'adversarial_annotations_jsonl_file',
    'data/adversarial_annotations/adversarial_annotations.jsonl',
    'Path to the adversarial annotations file in jsonl format.')

flags.DEFINE_string('output_jsonl_file', 'data/val_adv_annots.jsonl',
                    'Path to the annotations file in jsonl format.')

flags.DEFINE_enum('name', 'remove_shortcut', [
    'remove_shortcut', 'keep_shortcut_1', 'keep_shortcut_3', 'keep_shortcut_5'
], 'Method name.')

flags.DEFINE_bool('only_modify_distracting_options', False,
                  'If true, only modify distracting options.')

FLAGS = flags.FLAGS

MASKING_OFFSET = 10000


def _load_annotations(filename):
  """Loads annotations from file.

  Args:
    filename: Path to the jsonl annotations file.

  Returns:
    A list of python dictionary, each is parsed from a json object.
  """
  with tf.io.gfile.GFile(filename, 'r') as f:
    return [json.loads(x.strip('\n')) for x in f]


def _modify_annotation(choice_to_match, tokens, losses):
  """Modify annotations."""

  assert len(tokens) == len(losses) >= len(choice_to_match)

  losses = np.array(losses)
  if FLAGS.name == 'remove_shortcut':
    masks = losses.argmax() == range(len(losses))
  elif FLAGS.name == 'keep_shortcut_1':
    masks = np.ones(len(losses), dtype=np.bool)
    masks[losses.argsort()[-1:]] = False
  elif FLAGS.name == 'keep_shortcut_3':
    masks = np.ones(len(losses), dtype=np.bool)
    masks[losses.argsort()[-3:]] = False
  elif FLAGS.name == 'keep_shortcut_5':
    masks = np.ones(len(losses), dtype=np.bool)
    masks[losses.argsort()[-5:]] = False
  else:
    raise ValueError('Invalid method name %s.' % FLAGS.name)

  a = b = 0
  new_choice = []
  while a < len(choice_to_match) and b < len(masks):
    if isinstance(choice_to_match[a], list):
      tags = []
      for tag in choice_to_match[a]:
        if masks[b]:
          tags.append(MASKING_OFFSET + tag)
        else:
          tags.append(tag)
        b += 1
      new_choice.append(tags)
    else:
      if masks[b]:
        new_choice.append('[MASK]')
      else:
        new_choice.append(choice_to_match[a])
      b += 1
    a += 1
  assert a == len(choice_to_match) and b == len(tokens)
  return new_choice


def main(_):
  logging.set_verbosity(logging.INFO)

  annots = _load_annotations(FLAGS.annotations_jsonl_file)
  adv_annots = _load_annotations(FLAGS.adversarial_annotations_jsonl_file)
  adv_annots = dict([(x['annot_id'], x) for x in adv_annots])

  logging.info('Loaded %i annotations.', len(annots))
  logging.info('Loaded %i adversarial annotations.', len(adv_annots))

  with tf.io.gfile.GFile(FLAGS.output_jsonl_file, 'w') as f:
    for annot in annots:
      if annot['annot_id'] in adv_annots:
        adv_annot = adv_annots[annot['annot_id']]

        new_answer_choices = []
        new_rationale_choices = []
        for i in range(4):
          if annot[
              'answer_label'] == i and FLAGS.only_modify_distracting_options:
            new_answer_choices.append(annot['answer_choices'][i])
          else:
            new_answer_choices.append(
                _modify_annotation(annot['answer_choices'][i],
                                   adv_annot['answer_tokens'][i],
                                   adv_annot['answer_losses'][i]))

        for i in range(4):
          if annot[
              'rationale_label'] == i and FLAGS.only_modify_distracting_options:
            new_rationale_choices.append(annot['rationale_choices'][i])
          else:
            new_rationale_choices.append(
                _modify_annotation(annot['rationale_choices'][i],
                                   adv_annot['rationale_tokens'][i],
                                   adv_annot['rationale_losses'][i]))

        # for i in range(4):
        #   print(new_answer_choices[i])
        # for i in range(4):
        #   print(new_rationale_choices[i])
        annot['answer_choices'] = new_answer_choices
        annot['rationale_choices'] = new_rationale_choices

      f.write(json.dumps(annot) + '\n')

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
