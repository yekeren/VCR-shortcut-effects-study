from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modeling.layers import token_to_id
from tf_slim import tfexample_decoder

from protos import reader_pb2

PAD = '[PAD]'
PAD_ID = 0
NUM_CHOICES = 4
DETECTION_ID_OFFSET = 399


class TFExampleFields(object):
  """Fields in the tf.train.Example."""
  annot_id = 'annot_id'
  img_id = 'img_id'
  img_encoded = 'image/encoded'
  img_format = 'image/format'
  answer_label = 'answer_label'
  rationale_label = 'rationale_label'

  detection_classes = "image/object/bbox/label"
  detection_scores = "image/object/bbox/score"
  detection_boxes_scope = "image/object/bbox/"
  detection_boxes_keys = ['ymin', 'xmin', 'ymax', 'xmax']
  detection_boxes_ymin = 'image/object/bbox/ymin'
  detection_boxes_ymax = 'image/object/bbox/ymax'
  detection_boxes_xmin = 'image/object/bbox/xmin'
  detection_boxes_xmax = 'image/object/bbox/xmax'

  question = 'question'
  question_tag = 'question_tag'
  answer_choice = 'answer_choice'
  answer_choice_tag = 'answer_choice_tag'
  rationale_choice = 'rationale_choice'
  rationale_choice_tag = 'rationale_choice_tag'


class Detections(object):
  """Rrepresents the detection."""

  def __init__(self, detection_boxes, detection_classes, detection_scores):
    self._detection_boxes = detection_boxes
    self._detection_classes = DETECTION_ID_OFFSET + detection_classes
    self._detection_scores = detection_scores
    self._num_detections = tf.shape(detection_boxes)[0]

  def to_dict(self):
    return {
        'num_detections': self._num_detections,
        'detection_boxes': self._detection_boxes,
        'detection_classes': self._detection_classes,
        'detection_scores': self._detection_scores
    }

  @classmethod
  def get_padded_shapes(self):
    return {
        'num_detections': [],
        'detection_boxes': [None, 4],
        'detection_classes': [None],
        'detection_scores': [None]
    }

  @classmethod
  def get_padding_values(self):
    return {
        'num_detections': 0,
        'detection_boxes': 0.0,
        'detection_classes': PAD_ID,
        'detection_scores': 0.0
    }


class MixedSequence(object):
  """Represents the sequence of mixed tokens and tags."""

  def __init__(self, tokens, tags):
    self._tags = tags
    self._tokens = tokens
    self._length = tf.shape(tokens)[0]

  def to_dict(self):
    return {'length': self._length, 'tokens': self._tokens, 'tags': self._tags}

  @classmethod
  def get_padded_shapes(self):
    return {'length': [], 'tokens': [None], 'tags': [None]}

  @classmethod
  def get_padding_values(self):
    return {'length': 0, 'tokens': PAD_ID, 'tags': -1}


def _update_decoded_example(decoded_example, options):
  """Updates the decoded example, add size to the varlen feature.

  Args:
    decoded_example: A tensor dictionary keyed by name.
    options: An instance of reader_pb2.Reader.

  Returns:
    decoded_example: The same instance with content modified.
  """
  token_to_id_fn = token_to_id.TokenToIdLayer(
      options.vocab_file, options.out_of_vocabulary_token_id)
  detection_to_id_fn = token_to_id.TokenToIdLayer(options.detection_vocab_file,
                                                  0)

  # Image and bounding boxes.
  image = decoded_example['img_data']
  image_shape = tf.shape(image)

  detections = Detections(
      decoded_example.pop('detection_boxes'),
      detection_to_id_fn(decoded_example.pop('detection_classes')),
      decoded_example.pop('detection_scores'))

  decoded_example.update({
      'img_height': image_shape[0],
      'img_width': image_shape[1],
      'detections': detections.to_dict(),
  })

  # Answer and rationale choices.
  for i in range(NUM_CHOICES):
    answer_choice = MixedSequence(
        token_to_id_fn(decoded_example.pop('answer_choice_%i' % i)),
        decoded_example.pop('answer_choice_tag_%i' % i))
    rationale_choice = MixedSequence(
        token_to_id_fn(decoded_example.pop('rationale_choice_%i' % i)),
        decoded_example.pop('rationale_choice_tag_%i' % i))

    decoded_example.update({
        'answer_choice_%i' % i: answer_choice.to_dict(),
        'rationale_choice_%i' % i: rationale_choice.to_dict(),
    })

  # Question and answer.
  question = MixedSequence(token_to_id_fn(decoded_example.pop('question')),
                           decoded_example.pop('question_tag'))

  decoded_example.update({'question': question.to_dict()})

  return decoded_example


def _parse_single_example(example, options):
  """Parses a single tf.Example proto.

  Args:
    example: An Example proto.
    options: An instance of reader_pb2.Reader.

  Returns:
    A dictionary indexed by tensor name.
  """
  ###################################
  # Initialize `keys_to_features`.
  ###################################
  keys_to_features = {
      TFExampleFields.annot_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.img_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.img_encoded: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.img_format: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.answer_label: tf.io.FixedLenFeature([], tf.int64),
      TFExampleFields.rationale_label: tf.io.FixedLenFeature([], tf.int64),
      TFExampleFields.detection_classes: tf.io.VarLenFeature(tf.string),
      TFExampleFields.detection_scores: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.detection_boxes_ymin: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.detection_boxes_ymax: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.detection_boxes_xmin: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.detection_boxes_xmax: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.question: tf.io.VarLenFeature(tf.string),
      TFExampleFields.question_tag: tf.io.VarLenFeature(tf.int64),
  }

  # Answer and rationale choices.
  for i in range(NUM_CHOICES):
    keys_to_features.update({
        TFExampleFields.answer_choice + '_%i' % i:
            tf.io.VarLenFeature(tf.string),
        TFExampleFields.answer_choice_tag + '_%i' % i:
            tf.io.VarLenFeature(tf.int64),
        TFExampleFields.rationale_choice + '_%i' % i:
            tf.io.VarLenFeature(tf.string),
        TFExampleFields.rationale_choice_tag + '_%i' % i:
            tf.io.VarLenFeature(tf.int64),
    })

  ###################################
  # Initialize `items_to_handlers`.
  ###################################
  items_to_handlers = {
      'annot_id':
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.annot_id,
                                   default_value=''),
      'img_id':
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_id,
                                   default_value=''),
      'img_data':
          tfexample_decoder.Image(image_key=TFExampleFields.img_encoded,
                                  format_key=TFExampleFields.img_format,
                                  shape=None),
      'answer_label':
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.answer_label,
                                   default_value=-1),
      'rationale_label':
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.rationale_label,
                                   default_value=-1),
      'detection_boxes':
          tfexample_decoder.BoundingBox(
              keys=TFExampleFields.detection_boxes_keys,
              prefix=TFExampleFields.detection_boxes_scope),
      'detection_classes':
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.detection_classes,
                                   default_value=PAD),
      'detection_scores':
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.detection_scores,
                                   default_value=0),
      'question':
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.question,
                                   default_value=PAD),
      'question_tag':
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.question_tag,
                                   default_value=-1),
  }

  # Answer and rationale choices.
  for i in range(NUM_CHOICES):
    items_to_handlers['answer_choice_%i' % i] = tfexample_decoder.Tensor(
        tensor_key='answer_choice_%i' % i, default_value=PAD)
    items_to_handlers['answer_choice_tag_%i' % i] = tfexample_decoder.Tensor(
        tensor_key='answer_choice_tag_%i' % i, default_value=-1)

    items_to_handlers['rationale_choice_%i' % i] = tfexample_decoder.Tensor(
        tensor_key='rationale_choice_%i' % i, default_value=PAD)
    items_to_handlers['rationale_choice_tag_%i' % i] = tfexample_decoder.Tensor(
        tensor_key='rationale_choice_tag_%i' % i, default_value=-1)

  # Decode example.
  example_decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                       items_to_handlers)

  output_keys = example_decoder.list_items()
  output_tensors = example_decoder.decode(example)
  output_tensors = [
      x if x.dtype != tf.int64 else tf.cast(x, tf.int32) for x in output_tensors
  ]
  decoded_example = dict(zip(output_keys, output_tensors))
  return _update_decoded_example(decoded_example, options)


def _create_dataset(options, is_training, input_pipeline_context=None):
  """Creates dataset object based on options.

  Args:
    options: An instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.
    input_pipeline_context: A tf.distribute.InputContext instance.

  Returns:
    A tf.data.Dataset object.
  """
  dataset = tf.data.Dataset.list_files(options.input_pattern[:],
                                       shuffle=is_training)

  batch_size = options.batch_size
  if input_pipeline_context:
    if input_pipeline_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                              input_pipeline_context.input_pipeline_id)
    batch_size = input_pipeline_context.get_per_replica_batch_size(
        options.batch_size)

  if is_training:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(options.shuffle_buffer_size)
  dataset = dataset.interleave(tf.data.TFRecordDataset,
                               cycle_length=options.interleave_cycle_length)

  def parse_fn(x):
    return _parse_single_example(x, options)

  dataset = dataset.map(map_func=parse_fn,
                        num_parallel_calls=options.num_parallel_calls)

  padded_shapes = {
      'annot_id': [],
      'img_id': [],
      'img_data': [None, None, 3],
      'img_height': [],
      'img_width': [],
      'answer_label': [],
      'rationale_label': [],
      'detections': Detections.get_padded_shapes(),
      'question': MixedSequence.get_padded_shapes(),
      'answer_choice_0': MixedSequence.get_padded_shapes(),
      'answer_choice_1': MixedSequence.get_padded_shapes(),
      'answer_choice_2': MixedSequence.get_padded_shapes(),
      'answer_choice_3': MixedSequence.get_padded_shapes(),
      'rationale_choice_0': MixedSequence.get_padded_shapes(),
      'rationale_choice_1': MixedSequence.get_padded_shapes(),
      'rationale_choice_2': MixedSequence.get_padded_shapes(),
      'rationale_choice_3': MixedSequence.get_padded_shapes(),
  }
  padding_values = {
      'annot_id': '',
      'img_id': '',
      'img_data': tf.constant(0, dtype=tf.uint8),
      'img_height': 0,
      'img_width': 0,
      'answer_label': -1,
      'rationale_label': -1,
      'detections': Detections.get_padding_values(),
      'question': MixedSequence.get_padding_values(),
      'answer_choice_0': MixedSequence.get_padding_values(),
      'answer_choice_1': MixedSequence.get_padding_values(),
      'answer_choice_2': MixedSequence.get_padding_values(),
      'answer_choice_3': MixedSequence.get_padding_values(),
      'rationale_choice_0': MixedSequence.get_padding_values(),
      'rationale_choice_1': MixedSequence.get_padding_values(),
      'rationale_choice_2': MixedSequence.get_padding_values(),
      'rationale_choice_3': MixedSequence.get_padding_values(),
  }
  dataset = dataset.padded_batch(batch_size,
                                 padded_shapes=padded_shapes,
                                 padding_values=padding_values,
                                 drop_remainder=True)
  dataset = dataset.prefetch(options.prefetch_buffer_size)
  return dataset


def get_input_fn(options, is_training):
  """Returns a function that generate input examples.

  Args:
    options: An instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.VCRReaderV2):
    raise ValueError('options has to be an instance of VCRReaderV2.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
