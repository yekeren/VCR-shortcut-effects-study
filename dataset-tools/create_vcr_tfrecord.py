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

flags.DEFINE_string('bert_vocab_file',
                    'data/bert/tf1.x/cased_L-12_H-768_A-12/vocab.txt',
                    'Path to the Bert vocabulary file.')

flags.DEFINE_boolean('do_lower_case', False,
                     'To be passed to the bert tokenizer.')

flags.DEFINE_string('annotations_jsonl_file', 'data/vcr1annots/val.jsonl',
                    'Path to the annotations file in jsonl format.')

flags.DEFINE_integer('num_shards', 10,
                     'Number of shards of the output tfrecord files.')

flags.DEFINE_integer('shard_id', 0, 'Shard id of the current process.')

flags.DEFINE_string('output_tfrecord_path', 'output/val.record',
                    'Path to the output tfrecord files.')

flags.DEFINE_string('image_zip_file', 'data/vcr1images.zip',
                    'Path to the zip file of images.')

flags.DEFINE_boolean('only_use_relevant_dets', False,
                     'If true, only use relevant detections.')

FLAGS = flags.FLAGS

# GENDER_NEUTRAL_NAMES = [
#     'Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry',
#     'Jody', 'Kendall', 'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn'
# ]
NUM_CHOICES = 4
GENDER_NEUTRAL_NAMES = [
    'Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry',
    'Kendall', 'Frankie', 'Pat', 'Quinn'
]
MASKING_OFFSET = 10000

_NUM_PARTITIONS = 100


def get_partition_id(annot_id, num_partitions=_NUM_PARTITIONS):
  split, number = annot_id.split('-')
  return int(number) % num_partitions


def _load_annotations(filename):
  """Loads annotations from file.

  Args:
    filename: Path to the jsonl annotations file.

  Returns:
    A list of python dictionary, each is parsed from a json object.
  """
  with tf.io.gfile.GFile(filename, 'r') as f:
    return [json.loads(x.strip('\n')) for x in f]


def _fix_tokenization(tokenized_sent, obj_to_type, old_det_to_new_ind,
                      bert_tokenizer, do_lower_case):
  """Converts tokenized annotations into tokenized sentence.

  Args:
    tokenized_sent: Tokenized sentence with detections collapsed to a list.
    obj_to_type: [person, person, pottedplant] indexed by the labels.
    old_det_to_new_ind: A mapping from the old indices to the new indices.

  Returns:
    tokenized_sent: A list of string tokens.
  """
  case_fn = lambda x: x.lower() if do_lower_case else x

  new_tokenization_with_tags = []
  for tok in tokenized_sent:
    if isinstance(tok, list):
      for idx in tok:
        mask_flag = False
        if idx >= MASKING_OFFSET:
          mask_flag = True
          idx -= MASKING_OFFSET
        if old_det_to_new_ind is not None:
          idx = old_det_to_new_ind[idx]

        obj_type = obj_to_type[idx]
        text_to_use = GENDER_NEUTRAL_NAMES[
            idx %
            len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
        if mask_flag:
          new_tokenization_with_tags.append(('[MASK]', idx))
        else:
          new_tokenization_with_tags.append((case_fn(text_to_use), idx))
    else:
      # new_tokenization_with_tags.append((case_fn(tok), -1))
      # yek@: disable wordpiece tokenizer
      if tok == '[MASK]':
        new_tokenization_with_tags.append((tok, -1))
      else:
        for sub_tok in bert_tokenizer.wordpiece_tokenizer.tokenize(case_fn(tok)):
          new_tokenization_with_tags.append((sub_tok, -1))

  tokenized_sent, tags = zip(*new_tokenization_with_tags)
  return list(tokenized_sent), list(tags)


def get_detections_to_use(obj_to_type, tokens_mixed_with_tags):
  """Gets the detections to use, filtering out objects that are not mentioned.

  Args:
    obj_to_type: A list of object names.
    tokens_mixed_with_tags: Sentences that refer to the object list.

  Returns:
    indices: Indices in the obj_to_type, denoting the mentioned objects.
  """
  detections_to_use = np.zeros(len(obj_to_type), dtype=bool)
  people = np.array([x == 'person' for x in obj_to_type], dtype=bool)

  for sentence in tokens_mixed_with_tags:
    for possibly_det_list in sentence:
      if isinstance(possibly_det_list, list):
        for tag in possibly_det_list:
          assert 0 <= tag < len(obj_to_type)
          detections_to_use[tag] = True
      elif possibly_det_list.lower() in ['everyone', 'everyones']:
        detections_to_use |= people
  if not detections_to_use.any():
    detections_to_use |= people
  detections_to_use = np.where(detections_to_use)[0]

  old_det_to_new_ind = np.zeros(len(obj_to_type), dtype=np.int32) - 1
  old_det_to_new_ind[detections_to_use] = np.arange(detections_to_use.shape[0],
                                                    dtype=np.int32)
  return detections_to_use, old_det_to_new_ind


def _create_tf_example(encoded_jpeg, annot, meta, bert_tokenizer, do_lower_case,
                       only_use_relevant_dets, desired_size=400):
  """Creates an example from the annotation.

  Args:
    encoded_jpeg: A python string, the encoded jpeg data.
    annot: A python dictionary parsed from the json object.
    meta: A python dictionary containing object information.
    bert_tokenizer: A tokenization.FullTokenizer object.
    do_lower_case: If true, convert text to lower case.

  Returns:
    tf_example: A tf.train.Example proto.
  """

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value.encode('utf8')]))

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def _bytes_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[x.encode('utf8') for x in value]))

  def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  # Encode objects and boxes.
  image_height, image_width = meta['height'], meta['width']
  assert (meta['names'] == annot['objects']
         ), 'Meta data does not match the annotation.'

  obj_to_type = annot['objects']
  question = annot['question']
  answer_choices = annot['answer_choices']
  rationale_choices = annot['rationale_choices']
  assert NUM_CHOICES == len(answer_choices) == len(rationale_choices)

  obj_to_type = np.array(obj_to_type)
  boxes = np.array(meta['boxes'])

  old_det_to_new_ind = None
  if only_use_relevant_dets:
    detections_to_use, old_det_to_new_ind = get_detections_to_use(
        obj_to_type, [question] + answer_choices + rationale_choices)
    old_det_to_new_ind = [-1 if x < 0 else x + 1 for x in old_det_to_new_ind]

    # Gather elements using the new indices.
    obj_to_type = obj_to_type[detections_to_use]
    boxes = boxes[detections_to_use]
  else:
    old_det_to_new_ind = np.arange(len(obj_to_type), dtype=np.int32) + 1

  # Add [0, 0, height, width] as for the full image.
  obj_to_type = np.concatenate([['[unused400]'], obj_to_type], 0)
  boxes = np.concatenate([[[0, 0, image_width, image_height, 1]], boxes], 0)

  xmin, ymin, xmax, ymax, score = [boxes[:, i] for i in range(5)]
  xmin /= image_width
  ymin /= image_height
  xmax /= image_width
  ymax /= image_height

  feature = {}
  for key, value in annot.items():
    if isinstance(value, int):
      feature[key] = _int64_feature(value)
    elif isinstance(value, str):
      feature[key] = _bytes_feature(value)
  feature['image/object/bbox/xmin'] = _float_feature_list(xmin.tolist())
  feature['image/object/bbox/ymin'] = _float_feature_list(ymin.tolist())
  feature['image/object/bbox/xmax'] = _float_feature_list(xmax.tolist())
  feature['image/object/bbox/ymax'] = _float_feature_list(ymax.tolist())
  feature['image/object/bbox/score'] = _float_feature_list(score.tolist())
  feature['image/object/bbox/label'] = _bytes_feature_list(obj_to_type.tolist())

  # Encode jpeg data, resize image if specified.
  image = PIL.Image.open(io.BytesIO(encoded_jpeg))
  assert image.format == 'JPEG'

  min_size = min(image.width, image.height)
  scale = 1.0 * desired_size / min_size
  if scale < 1.0:
    new_height, new_width = (int(image.height * scale),
                             int(image.width * scale))
    image = image.resize((new_width, new_height))
  with io.BytesIO() as f:
    image.save(f, format="JPEG")
    encoded_jpeg = f.getvalue()

  feature['image/format'] = _bytes_feature('jpeg')
  feature['image/encoded'] = tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[encoded_jpeg]))

  # Encode question.
  question_tokens, question_tags = _fix_tokenization(question, obj_to_type,
                                                     old_det_to_new_ind,
                                                     bert_tokenizer,
                                                     do_lower_case)
  feature['question'] = _bytes_feature_list(question_tokens)
  feature['question_tag'] = _int64_feature_list(question_tags)

  # Encode answer choices.
  for idx, tokenized_sent in enumerate(answer_choices):
    tokens, tags = _fix_tokenization(tokenized_sent, obj_to_type,
                                     old_det_to_new_ind, bert_tokenizer,
                                     do_lower_case)
    feature['answer_choice_%i' % (idx + 1)] = _bytes_feature_list(tokens)
    feature['answer_choice_tag_%i' % (idx + 1)] = _int64_feature_list(tags)

  # Encode ratinale choices.
  for idx, tokenized_sent in enumerate(rationale_choices):
    tokens, tags = _fix_tokenization(tokenized_sent, obj_to_type,
                                     old_det_to_new_ind, bert_tokenizer,
                                     do_lower_case)
    feature['rationale_choice_%i' % (idx + 1)] = _bytes_feature_list(tokens)
    feature['rationale_choice_tag_%i' % (idx + 1)] = _int64_feature_list(tags)

  tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
  return tf_example


def main(_):
  logging.set_verbosity(logging.INFO)

  # Create Bert model.
  bert_tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.bert_vocab_file,
                                              do_lower_case=FLAGS.do_lower_case)

  # Load annotations.
  annots = _load_annotations(FLAGS.annotations_jsonl_file)
  logging.info('Loaded %i annotations.', len(annots))

  shard_id, num_shards = FLAGS.shard_id, FLAGS.num_shards
  assert 0 <= shard_id < num_shards

  writer = tf.io.TFRecordWriter(FLAGS.output_tfrecord_path + '-%05d-of-%05d' %
                                (shard_id, num_shards))

  with zipfile.ZipFile(FLAGS.image_zip_file) as image_zip:
    for idx, annot in enumerate(annots):
      if (idx + 1) % 1000 == 0:
        logging.info('On example %i/%i.', idx + 1, len(annots))

      annot_id = int(annot['annot_id'].split('-')[-1])
      if annot_id % num_shards != shard_id:
        continue

      # Read meta data.
      meta_fn = os.path.join('vcr1images', annot['metadata_fn'])
      try:
        with image_zip.open(meta_fn, 'r') as f:
          meta = json.load(f)
      except Exception as ex:
        logging.warn('Skip %s.', meta_fn)
        continue

      # Read image data.
      img_fn = os.path.join('vcr1images', annot['img_fn'])
      try:
        with image_zip.open(img_fn, 'r') as f:
          encoded_jpeg = f.read()
      except Exception as ex:
        logging.warn('Skip %s.', img_fn)
        continue
      # Create TF example.
      tf_example = _create_tf_example(encoded_jpeg, annot, meta, bert_tokenizer,
                                      FLAGS.do_lower_case,
                                      FLAGS.only_use_relevant_dets)
      writer.write(tf_example.SerializeToString())

  writer.close()

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
