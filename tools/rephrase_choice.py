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

flags.DEFINE_string('default_gender_pronoun', 'he',
                    'Default gender pronoun to use.')

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


def get_uniq_person_tags(mixed_caption, classes):
  """Returns the uniq tags in the caption.

  Args:
    mixed_caption: A list mixed with string tokens and tag tokens.
    classes: A list of detected classes.

  Returns:
    tags: A list of tag tokens (integers).
  """
  tags = set()
  for token_or_tags in mixed_caption:
    if isinstance(token_or_tags, list):
      for tag in token_or_tags:
        if classes[tag] == 'person':
          tags.add(tag)
  return list(tags)


def get_person_pronouns(mixed_caption):
  """Returns the number of pronouns in the caption.
  Args:
    mixed_caption: A list mixed with string tokens and tag tokens.

  Returns:
    pronouns: A list of lower case strings.
  """
  pronouns = set()
  for token_or_tags in mixed_caption:
    if isinstance(token_or_tags, str):
      if token_or_tags.lower() in ['he', 'she']:
        pronouns.add(token_or_tags.lower())
  return list(pronouns)


def get_most_probable_pronoun(choice, question, default='he'):

  def _update_count(count, caption):
    for token in caption:
      if token in ['he', 'his', 'him', 'He', 'His', 'Him']:
        count['he'] += 1
      elif token in ['she', 'her', 'hers', 'She', 'Her', 'Hers']:
        count['she'] += 1
    return count

  count = {'he': 0, 'she': 0}
  _update_count(count, choice)
  if count['he'] == count['she']:
    _update_count(count, question)

  if count['he'] > count['she']:
    return 'he'
  elif count['he'] < count['she']:
    return 'she'
  else:
    return default


def replace_gender_pronoun_with_tag(mixed_caption, classes, from_tags, to_tags):
  """Replaces to get a new confusing caption.  """
  new_mixed_caption = []
  for token_or_tags in mixed_caption:
    if token_or_tags == from_tags:
      new_mixed_caption.append(to_tags)
    elif token_or_tags in ['he', 'she', 'He', 'She']:
      new_mixed_caption.append(to_tags)
    else:
      new_mixed_caption.append(token_or_tags)
  return new_mixed_caption


def replace_tag_with_gender_pronoun(mixed_caption, classes, from_tags, to_tags,
                                    default_pronoun):
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
  question_person_tags = get_uniq_person_tags(question, detection_classes)

  new_choices = []
  modified = 0
  for i, choice in enumerate(choices):
    person_tags = get_uniq_person_tags(choice, detection_classes)
    person_pronouns = get_person_pronouns(choice)

    new_choice = choice
    if len(person_tags) + len(person_pronouns) == 1:
      if i != label:
        if FLAGS.modify_negatives:
          modified += 1
          new_choice = replace_gender_pronoun_with_tag(choice, detection_classes,
                                                       person_tags,
                                                       question_person_tags)
      else:
        if FLAGS.modify_positives:
          modified += 1
          default = FLAGS.default_gender_pronoun
          default_pronoun = get_most_probable_pronoun(choice,
                                                      question,
                                                      default=default)
          new_choice = replace_tag_with_gender_pronoun(choice, detection_classes,
                                                       person_tags,
                                                       question_person_tags,
                                                       default_pronoun)
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

      # Check if the question is regarding a single person.
      question_person_tags = get_uniq_person_tags(question, detection_classes)
      if len(question_person_tags) != 1:
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
