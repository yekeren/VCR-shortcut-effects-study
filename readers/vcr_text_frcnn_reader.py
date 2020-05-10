from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf

from tf_slim import tfexample_decoder
from protos import reader_pb2
from readers.vcr_fields import *
from modeling.layers import token_to_id


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
  token_to_id_func = token_to_id.TokenToIdLayer(
      options.vocab_file, options.out_of_vocabulary_token_id)

  # Number of objects.
  detection_boxes = decoded_example[InputFields.detection_boxes]
  detection_classes = decoded_example[InputFields.detection_classes]
  num_detections = tf.shape(detection_boxes)[0]

  # Object Fast-RCNN features.
  detection_features = decoded_example.pop(TFExampleFields.detection_features)
  detection_features = tf.reshape(detection_features,
                                  [-1, options.frcnn_feature_dims])

  # Question length.
  question = decoded_example[InputFields.question]
  question_tag = decoded_example[InputFields.question_tag]
  question_len = tf.shape(question)[0]

  # Answer and rationale choices.
  answer_choices_list = [
      decoded_example.pop(TFExampleFields.answer_choice + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  answer_choices_tag_list = [
      decoded_example.pop(TFExampleFields.answer_choice_tag + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  (answer_choices, answer_choices_len) = _pad_sequences(answer_choices_list)
  (answer_choices_tag, _) = _pad_sequences(answer_choices_tag_list, -1)

  rationale_choices_list = [
      decoded_example.pop(TFExampleFields.rationale_choice + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  rationale_choices_tag_list = [
      decoded_example.pop(TFExampleFields.rationale_choice_tag + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  (rationale_choices,
   rationale_choices_len) = _pad_sequences(rationale_choices_list)
  (rationale_choices_tag, _) = _pad_sequences(rationale_choices_tag_list, -1)

  # Mixed question -> answer, question-answer -> rationale.
  answer_len = answer_choices_len[decoded_example[InputFields.answer_label]]
  answer = answer_choices[decoded_example[
      InputFields.answer_label]][:answer_len]
  answer_tag = answer_choices_tag[decoded_example[
      InputFields.answer_label]][:answer_len]

  mixed_answer_choices_list = [
      tf.concat([question, ['[SEP]'], x], 0) for x in answer_choices_list
  ]
  mixed_answer_choices_tag_list = [
      tf.concat([question_tag, [-1], x], 0) for x in answer_choices_tag_list
  ]
  (mixed_answer_choices,
   mixed_answer_choices_len) = _pad_sequences(mixed_answer_choices_list)
  (mixed_answer_choices_tag, _) = _pad_sequences(mixed_answer_choices_tag_list,
                                                 pad=-1)

  mixed_rationale_choices_list = [
      tf.concat([question, ['[SEP]'], answer, ['[SEP]'], x], 0)
      for x in rationale_choices_list
  ]
  mixed_rationale_choices_tag_list = [
      tf.concat([question_tag, [-1], answer_tag, [-1], x], 0)
      for x in rationale_choices_tag_list
  ]
  (mixed_rationale_choices,
   mixed_rationale_choices_len) = _pad_sequences(mixed_rationale_choices_list)
  (mixed_rationale_choices_tag,
   _) = _pad_sequences(mixed_rationale_choices_tag_list, pad=-1)

  decoded_example.update({
      InputFields.num_detections:
          num_detections,
      InputFields.detection_classes:
          token_to_id_func(detection_classes),
      InputFields.detection_features:
          detection_features,
      InputFields.question:
          tf.tile(tf.expand_dims(token_to_id_func(question), 0),
                  [NUM_CHOICES, 1]),
      InputFields.question_tag:
          tf.tile(tf.expand_dims(question_tag, 0), [NUM_CHOICES, 1]),
      InputFields.question_len:
          tf.tile(tf.expand_dims(question_len, 0), [NUM_CHOICES]),
      InputFields.answer_choices:
          token_to_id_func(answer_choices),
      InputFields.answer_choices_tag:
          answer_choices_tag,
      InputFields.answer_choices_len:
          answer_choices_len,
      InputFields.rationale_choices:
          token_to_id_func(rationale_choices),
      InputFields.rationale_choices_tag:
          rationale_choices_tag,
      InputFields.rationale_choices_len:
          rationale_choices_len,
      InputFields.mixed_answer_choices:
          token_to_id_func(mixed_answer_choices),
      InputFields.mixed_answer_choices_tag:
          mixed_answer_choices_tag,
      InputFields.mixed_answer_choices_len:
          mixed_answer_choices_len,
      InputFields.mixed_rationale_choices:
          token_to_id_func(mixed_rationale_choices),
      InputFields.mixed_rationale_choices_tag:
          mixed_rationale_choices_tag,
      InputFields.mixed_rationale_choices_len:
          mixed_rationale_choices_len,
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
      TFExampleFields.annot_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.answer_label: tf.io.FixedLenFeature([], tf.int64),
      TFExampleFields.rationale_label: tf.io.FixedLenFeature([], tf.int64),
      TFExampleFields.detection_classes: tf.io.VarLenFeature(tf.string),
      TFExampleFields.detection_scores: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.detection_features: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.question: tf.io.VarLenFeature(tf.string),
      TFExampleFields.question_tag: tf.io.VarLenFeature(tf.int64),
  }
  for bbox_key in TFExampleFields.detection_boxes_keys:
    bbox_field = os.path.join(TFExampleFields.detection_boxes_scope, bbox_key)
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
      InputFields.detection_boxes:
          tfexample_decoder.BoundingBox(
              keys=TFExampleFields.detection_boxes_keys,
              prefix=TFExampleFields.detection_boxes_scope),
      InputFields.detection_classes:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.detection_classes,
                                   default_value=''),
      InputFields.detection_scores:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.detection_scores,
                                   default_value=0),
      InputFields.question:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.question,
                                   default_value=PAD),
      InputFields.question_tag:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.question_tag,
                                   default_value=-1),
      TFExampleFields.detection_features:
          tfexample_decoder.Tensor(
              tensor_key=TFExampleFields.detection_features, default_value=0),
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
    if options.cache_dataset:
      dataset = dataset.cache()
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
      InputFields.num_detections: [],
      InputFields.detection_boxes: [None, 4],
      InputFields.detection_classes: [None],
      InputFields.detection_scores: [None],
      InputFields.detection_features: [None, options.frcnn_feature_dims],
      InputFields.question: [NUM_CHOICES, None],
      InputFields.question_tag: [NUM_CHOICES, None],
      InputFields.question_len: [NUM_CHOICES],
      InputFields.answer_choices: [NUM_CHOICES, None],
      InputFields.answer_choices_tag: [NUM_CHOICES, None],
      InputFields.answer_choices_len: [NUM_CHOICES],
      InputFields.rationale_choices: [NUM_CHOICES, None],
      InputFields.rationale_choices_tag: [NUM_CHOICES, None],
      InputFields.rationale_choices_len: [NUM_CHOICES],
      InputFields.mixed_answer_choices: [NUM_CHOICES, None],
      InputFields.mixed_answer_choices_tag: [NUM_CHOICES, None],
      InputFields.mixed_answer_choices_len: [NUM_CHOICES],
      InputFields.mixed_rationale_choices: [NUM_CHOICES, None],
      InputFields.mixed_rationale_choices_tag: [NUM_CHOICES, None],
      InputFields.mixed_rationale_choices_len: [NUM_CHOICES],
  }
  padding_values = {
      InputFields.img_id: '',
      InputFields.annot_id: '',
      InputFields.answer_label: -1,
      InputFields.rationale_label: -1,
      InputFields.num_detections: 0,
      InputFields.detection_boxes: 0.0,
      InputFields.detection_classes: PAD_ID,
      InputFields.detection_scores: 0.0,
      InputFields.detection_features: 0.0,
      InputFields.question: PAD_ID,
      InputFields.question_tag: -1,
      InputFields.question_len: 0,
      InputFields.answer_choices: PAD_ID,
      InputFields.answer_choices_tag: -1,
      InputFields.answer_choices_len: 0,
      InputFields.rationale_choices: PAD_ID,
      InputFields.rationale_choices_tag: -1,
      InputFields.rationale_choices_len: 0,
      InputFields.mixed_answer_choices: PAD_ID,
      InputFields.mixed_answer_choices_tag: -1,
      InputFields.mixed_answer_choices_len: 0,
      InputFields.mixed_rationale_choices: PAD_ID,
      InputFields.mixed_rationale_choices_tag: -1,
      InputFields.mixed_rationale_choices_len: 0,
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
  if not isinstance(options, reader_pb2.VCRTextFRCNNReader):
    raise ValueError('options has to be an instance of TextFRCNNReader.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
