from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf

from tf_slim import tfexample_decoder
from protos import reader_pb2
from readers.vcr_fields import *


def _pad_sequences(sequences, pad=PAD):
  """Pads sequences to the max-length.

  Args:
    sequences: A list of 1-D tensor of size num_sequences, each elem in
      the 1-D tensor denotes a sequence.

  Returns:
    padded_sequences: A [num_sequences, max_sequence_len] tensor.
    lengths: A [num_sequences] int tensor.
  """
  lengths = [tf.shape(x)[0] for x in sequences]
  padded_size = tf.reduce_max(lengths)
  padded_sequences = tf.stack([
      tf.pad(x,
             paddings=[[0, padded_size - lengths[i]]],
             mode='CONSTANT',
             constant_values=pad) for i, x in enumerate(sequences)
  ])
  return padded_sequences, tf.stack(lengths)


def _update_decoded_example(decoded_example, options):
  """Updates the decoded example, add size to the varlen feature.

  Args:
    decoded_example: A tensor dictionary keyed by name.
    options: An instance of reader_pb2.Reader.

  Returns:
    decoded_example: The same instance with content modified.
  """
  # Number of objects.
  object_bboxes = decoded_example[InputFields.object_bboxes]
  num_objects = tf.shape(object_bboxes)[0]

  # Question length.
  question = decoded_example[InputFields.question]
  question_len = tf.shape(question)[0]
  question_tag = decoded_example[InputFields.question_tag]

  # Answer choices and lengths.
  answer_choices_list = [
      decoded_example.pop(TFExampleFields.answer_choice + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  answer_choices_with_question_list = [
      tf.concat([question, ['[SEP]'], x], 0)
      for x in answer_choices_list
  ]
  (answer_choices, answer_choices_len) = _pad_sequences(answer_choices_list)
  (answer_choices_with_question, answer_choices_with_question_len
  ) = _pad_sequences(answer_choices_with_question_list)

  # Answer tags.
  answer_choices_tag_list = [
      decoded_example.pop(TFExampleFields.answer_choice_tag + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  answer_choices_with_question_tag_list = [
      tf.concat([question_tag, [-1], x], 0)
      for x in answer_choices_tag_list
  ]
  answer_choices_tag, _ = _pad_sequences(answer_choices_tag_list, -1)
  answer_choices_with_question_tag, _ = _pad_sequences(
      answer_choices_with_question_tag_list, -1)

  answer_len = answer_choices_len[decoded_example[InputFields.answer_label]]
  answer = answer_choices[decoded_example[
      InputFields.answer_label]][:answer_len]
  answer_tag = answer_choices_tag[decoded_example[
      InputFields.answer_label]][:answer_len]

  # Rationale choices and lengths.
  rationale_choices_list = [
      decoded_example.pop(TFExampleFields.rationale_choice + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  rationale_choices_with_question_list = [
      tf.concat(
          [question, ['[SEP]'], answer, ['[SEP]'], x], 0)
      for x in rationale_choices_list
  ]
  (rationale_choices,
   rationale_choices_len) = _pad_sequences(rationale_choices_list)
  (rationale_choices_with_question, rationale_choices_with_question_len
  ) = _pad_sequences(rationale_choices_with_question_list)

  # Rationale tags.
  rationale_choices_tag_list = [
      decoded_example.pop(TFExampleFields.rationale_choice_tag + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  rationale_choices_with_question_tag_list = [
      tf.concat([question_tag, [-1], answer_tag, [-1], x], 0)
      for x in rationale_choices_tag_list
  ]
  rationale_choices_tag, _ = _pad_sequences(rationale_choices_tag_list, -1)
  rationale_choices_with_question_tag, _ = _pad_sequences(
      rationale_choices_with_question_tag_list, -1)

  # Image shape.
  image = decoded_example[InputFields.img_data]
  image_shape = tf.shape(image)

  decoded_example.update({
      InputFields.img_data:
          image,
      InputFields.img_height:
          image_shape[0],
      InputFields.img_width:
          image_shape[1],
      InputFields.num_objects:
          num_objects,
      InputFields.question_tag:
          question_tag,
      InputFields.question_len:
          question_len,
      InputFields.answer_choices:
          answer_choices,
      InputFields.answer_choices_tag:
          answer_choices_tag,
      InputFields.answer_choices_len:
          answer_choices_len,
      InputFields.answer_choices_with_question:
          answer_choices_with_question,
      InputFields.answer_choices_with_question_tag:
          answer_choices_with_question_tag,
      InputFields.answer_choices_with_question_len:
          answer_choices_with_question_len,
      InputFields.rationale_choices:
          rationale_choices,
      InputFields.rationale_choices_tag:
          rationale_choices_tag,
      InputFields.rationale_choices_len:
          rationale_choices_len,
      InputFields.rationale_choices_with_question:
          rationale_choices_with_question,
      InputFields.rationale_choices_with_question_tag:
          rationale_choices_with_question_tag,
      InputFields.rationale_choices_with_question_len:
          rationale_choices_with_question_len,
  })

  return decoded_example


def _parse_single_example(example, options):
  """Parses a single tf.Example proto.

  Args:
    example: An Example proto.
    options: An instance of reader_pb2.Reader.

  Returns:
    A dictionary indexed by tensor name.
  """
  # Initialize `keys_to_features`.
  keys_to_features = {
      TFExampleFields.img_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.img_encoded: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.img_format: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.annot_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.answer_label: tf.io.FixedLenFeature([], tf.int64),
      TFExampleFields.rationale_label: tf.io.FixedLenFeature([], tf.int64),
      TFExampleFields.img_bbox_label: tf.io.VarLenFeature(tf.string),
      TFExampleFields.img_bbox_score: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.question: tf.io.VarLenFeature(tf.string),
      TFExampleFields.question_tag: tf.io.VarLenFeature(tf.int64),
  }
  for bbox_key in TFExampleFields.img_bbox_field_keys:
    bbox_field = os.path.join(TFExampleFields.img_bbox_scope, bbox_key)
    keys_to_features[bbox_field] = tf.io.VarLenFeature(tf.float32)
  for i in range(1, 1 + NUM_CHOICES):
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

  # Initialize `items_to_handlers`.
  items_to_handlers = {
      InputFields.img_id:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_id,
                                   default_value=''),
      InputFields.annot_id:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.annot_id,
                                   default_value=''),
      InputFields.answer_label:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.answer_label,
                                   default_value=-1),
      InputFields.rationale_label:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.rationale_label,
                                   default_value=-1),
      InputFields.img_data:
          tfexample_decoder.Image(image_key=TFExampleFields.img_encoded,
                                  format_key=TFExampleFields.img_format,
                                  shape=None),
      InputFields.object_bboxes:
          tfexample_decoder.BoundingBox(
              keys=TFExampleFields.img_bbox_field_keys,
              prefix=TFExampleFields.img_bbox_scope),
      InputFields.object_labels:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_bbox_label,
                                   default_value=''),
      InputFields.object_scores:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_bbox_score,
                                   default_value=0),
      InputFields.question:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.question,
                                   default_value=PAD),
      InputFields.question_tag:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.question_tag,
                                   default_value=-1),
  }

  for i in range(1, 1 + NUM_CHOICES):
    tensor_key = TFExampleFields.answer_choice + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=PAD)

    tensor_key = TFExampleFields.answer_choice_tag + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=-1)

    tensor_key = TFExampleFields.rationale_choice + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=PAD)

    tensor_key = TFExampleFields.rationale_choice_tag + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=-1)

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

  parse_fn = lambda x: _parse_single_example(x, options)
  dataset = dataset.map(map_func=parse_fn,
                        num_parallel_calls=options.num_parallel_calls)

  padded_shapes = {
      InputFields.img_id: [],
      InputFields.annot_id: [],
      InputFields.answer_label: [],
      InputFields.rationale_label: [],
      InputFields.img_data: [None, None, 3],
      InputFields.img_height: [],
      InputFields.img_width: [],
      InputFields.num_objects: [],
      InputFields.object_bboxes: [None, 4],
      InputFields.object_labels: [None],
      InputFields.object_scores: [None],
      InputFields.question: [None],
      InputFields.question_tag: [None],
      InputFields.question_len: [],
      InputFields.answer_choices: [NUM_CHOICES, None],
      InputFields.answer_choices_tag: [NUM_CHOICES, None],
      InputFields.answer_choices_len: [NUM_CHOICES],
      InputFields.answer_choices_with_question: [NUM_CHOICES, None],
      InputFields.answer_choices_with_question_tag: [NUM_CHOICES, None],
      InputFields.answer_choices_with_question_len: [NUM_CHOICES],
      InputFields.rationale_choices: [NUM_CHOICES, None],
      InputFields.rationale_choices_tag: [NUM_CHOICES, None],
      InputFields.rationale_choices_len: [NUM_CHOICES],
      InputFields.rationale_choices_with_question: [NUM_CHOICES, None],
      InputFields.rationale_choices_with_question_tag: [NUM_CHOICES, None],
      InputFields.rationale_choices_with_question_len: [NUM_CHOICES],
  }
  padding_values = {
      InputFields.img_id: '',
      InputFields.annot_id: '',
      InputFields.answer_label: -1,
      InputFields.rationale_label: -1,
      InputFields.img_data: tf.constant(0, dtype=tf.uint8),
      InputFields.img_height: 0,
      InputFields.img_width: 0,
      InputFields.num_objects: 0,
      InputFields.object_bboxes: 0.0,
      InputFields.object_labels: '',
      InputFields.object_scores: 0.0,
      InputFields.question: PAD,
      InputFields.question_tag: -1,
      InputFields.question_len: 0,
      InputFields.answer_choices: PAD,
      InputFields.answer_choices_tag: -1,
      InputFields.answer_choices_len: 0,
      InputFields.answer_choices_with_question: PAD,
      InputFields.answer_choices_with_question_tag: -1,
      InputFields.answer_choices_with_question_len: 0,
      InputFields.rationale_choices: PAD,
      InputFields.rationale_choices_tag: -1,
      InputFields.rationale_choices_len: 0,
      InputFields.rationale_choices_with_question: PAD,
      InputFields.rationale_choices_with_question_tag: -1,
      InputFields.rationale_choices_with_question_len: 0,
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
  if not isinstance(options, reader_pb2.VCRReader):
    raise ValueError('options has to be an instance of VCRReader.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
