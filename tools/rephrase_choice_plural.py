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

flags.DEFINE_string('output_jsonl_file',
                    'data/rule_based_annotations/val_replaced.jsonl',
                    'Path to the annotations file in jsonl format.')

flags.DEFINE_bool('modify_positives', False, 'If true, modify correct choices.')

flags.DEFINE_bool('modify_negatives', False,
                  'If true, modify distracting choices.')

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


def get_the_only_person_group(mixed_caption, classes):
  """Gets the person group mentioned in the sentence.

  Args:
    mixed_caption: A list mixed with string tokens and tag tokens.
    classes: A list of detected classes.

  Returns:
    tags: A list of person tags.
  """
  tag_groups = [x for x in mixed_caption if isinstance(x, list)]
  if len(tag_groups) != 1:
    return None

  if any([classes[tag] != 'person' for tag in tag_groups[0]]):
    return None

  if len(tag_groups[0]) <= 1:
    return None

  return sorted(tag_groups[0])


def replace_pronoun_with_tag(mixed_caption, classes, from_tags, to_tags):
  """Replaces to get a new confusing caption.  """
  new_mixed_caption = []
  for token_or_tags in mixed_caption:
    if token_or_tags == from_tags:
      new_mixed_caption.append(to_tags)
    elif token_or_tags in ['they', 'They']:
      new_mixed_caption.append(to_tags)
    else:
      new_mixed_caption.append(token_or_tags)
  return new_mixed_caption


def replace_tag_with_pronoun(mixed_caption,
                             classes,
                             from_tags,
                             to_tags,
                             default_pronoun='they'):
  """Replaces to get a new confusing caption.  """
  new_mixed_caption = []
  for i, token_or_tags in enumerate(mixed_caption):
    if token_or_tags == from_tags:
      if i == 0:
        new_mixed_caption.append(default_pronoun[0].upper() +
                                 default_pronoun[1:].lower())
      else:
        new_mixed_caption.append(default_pronoun.lower())
    else:
      new_mixed_caption.append(token_or_tags)
  return new_mixed_caption


def modify_choices(detection_classes, question, choices, label):
  """Modify choices."""
  question_person_group = get_the_only_person_group(question, detection_classes)

  new_choices = []
  modified = 0
  for i, choice in enumerate(choices):

    new_choice = choice

    person_group = get_the_only_person_group(choice, detection_classes)

    if person_group is not None or 'they' in choice or 'They' in choice:

      if i != label:
        if FLAGS.modify_negatives:
          modified += 1
          new_choice = replace_pronoun_with_tag(choice, detection_classes,
                                                person_group,
                                                question_person_group)
      else:
        if FLAGS.modify_positives:
          modified += 1
          new_choice = replace_tag_with_pronoun(choice, detection_classes,
                                                person_group,
                                                question_person_group)
    new_choices.append(new_choice)
  return new_choices, modified


def main(_):
  logging.set_verbosity(logging.INFO)

  annots = _load_annotations(FLAGS.annotations_jsonl_file)
  logging.info('Loaded %i annotations.', len(annots))

  count = 0
  count_replaced_answer_choices = 0
  count_replaced_rationale_choices = 0
  with tf.io.gfile.GFile(FLAGS.output_jsonl_file, 'w') as f:
    for idx, annot in enumerate(annots):
      (question, answer_choices, answer_label, rationale_choices,
       rationale_label,
       detection_classes) = (annot['question'], annot['answer_choices'],
                             annot['answer_label'], annot['rationale_choices'],
                             annot['rationale_label'], annot['objects'])

      # Check if the question is regarding a group of persons.
      question_person_group = get_the_only_person_group(question,
                                                        detection_classes)
      if question_person_group is None:
        continue

      # Modify annotations.
      (new_answer_choices,
       modified_answer_choices) = modify_choices(detection_classes,
                                                 question=question,
                                                 choices=answer_choices,
                                                 label=answer_label)

      (new_rationale_choices,
       modified_rationale_choices) = modify_choices(detection_classes,
                                                    question=question,
                                                    choices=rationale_choices,
                                                    label=rationale_label)

      count += 1
      count_replaced_answer_choices += modified_answer_choices
      count_replaced_rationale_choices += modified_rationale_choices

      # Write output.
      assert len(new_answer_choices) == len(new_rationale_choices) == 4

      annot['answer_choices'] = new_answer_choices
      annot['rationale_choices'] = new_rationale_choices
      f.write(json.dumps(annot) + '\n')

  logging.info('Found %s / %s.', count, len(annots))
  logging.info('Replaced %s / %s answer choices.',
               count_replaced_answer_choices, 4 * count)
  logging.info('Replaced %s / %s rationale choices.',
               count_replaced_rationale_choices, 4 * count)
  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
