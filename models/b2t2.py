from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json
import numpy as np
import tensorflow as tf

from protos import model_pb2
from modeling.layers import token_to_id
from modeling.utils import hyperparams
from models.model_base import ModelBase

from readers.vcr_fields import InputFields
from readers.vcr_fields import NUM_CHOICES

from bert.modeling import BertConfig
from bert.modeling import BertModel
from bert.modeling import get_assignment_map_from_checkpoint

import tf_slim as slim

FIELD_ANSWER_PREDICTION = 'answer_prediction'


def _trim_to_max_num_objects(num_objects,
                             object_bboxes,
                             object_labels,
                             object_scores,
                             object_features,
                             max_num_objects=10):
  """Trims to the `max_num_objects` objects.

  Args:
    num_objects: A [batch] int tensor.
    object_bboxes: A [batch, pad_num_objects, 4] float tensor.
    object_labels: A [batch, pad_num_objects] int tensor.
    object_scores: A [batch, pad_num_objects] float tensor.
    object_features: A [batch, pad_num_objects, feature_dims] float tensor.
    max_num_objects: Maximum number of objects.

  Returns:
    num_objects: A [batch] int tensor.
    object_bboxes: A [batch, max_num_objects, 4] float tensor.
    object_labels: A [batch, max_num_objects] int tensor.
    object_scores: A [batch, max_num_objects] float tensor.
    object_features: A [batch, max_num_objects, feature_dims] float tensor.
    max_num_objects: Maximum number of objects in the batch.
  """
  max_num_objects = tf.minimum(tf.reduce_max(num_objects), max_num_objects)
  num_objects = tf.minimum(max_num_objects, num_objects)
  object_bboxes = object_bboxes[:, :max_num_objects, :]
  object_labels = object_labels[:, :max_num_objects]
  object_scores = object_scores[:, :max_num_objects]
  object_features = object_features[:, :max_num_objects, :]
  return (num_objects, object_bboxes, object_labels, object_scores,
          object_features, max_num_objects)


def _assign_invalid_tags(tags, max_num_objects):
  """Assigns `-1` to invalid tags.

  Args:
    tags: A int tensor, each value is a index in the objects array.
    max_num_objects: Maximum number of objects in the batch.
  """
  return tf.where(tags >= max_num_objects, -tf.ones_like(tags, tf.int32), tags)


def _predict_object_embeddings(object_features,
                               output_dims,
                               slim_fc_scope,
                               keep_prob=1.0,
                               is_training=False):
  """Projects object features to `output_dims` dimensions.

  Args:
    object_features: A [batch, max_num_objects, feature_dims] float tensor.
    output_dims: Dimensions of the object embeddings.
    slim_fc_scope: Slim FC scope.
    keep_prob: Keep probability of the dropout layer.
    is_training: If true, build a training graph.

  Returns:
    A [batch, max_num_objects, output_dims] float tensor.
  """
  output = object_features
  with slim.arg_scope(slim_fc_scope), tf.variable_scope('object_projection'):
    output = slim.fully_connected(output, num_outputs=output_dims)
    output = slim.dropout(output, keep_prob, is_training=is_training)
    output = slim.fully_connected(output,
                                  num_outputs=output_dims,
                                  activation_fn=None)
  return output


def _tile_objects(num_objects, object_ids, object_features):
  """Tiles object representations.

  Args:
    num_objects: A [batch] int tensor.
    object_ids: A [batch, max_num_object] int tensor.
    object_features: A [batch, max_num_objects, feature_dims] float tensor.

  Returns:
    tiled_object_masks: A [batch * NUM_CHOICES, max_num_objects] tensor.
    tiled_object_ids: A [batch * NUM_CHOICES, max_num_objects] tensor.
    tiled_object_features: A [batch * NUM_CHOICES, max_num_objects, feature_dims] tensor.
  """

  batch_size = num_objects.shape[0]
  tile_fn = lambda x: tf.gather(tf.expand_dims(x, 1), [0] * NUM_CHOICES, axis=1)

  object_masks = tf.sequence_mask(num_objects,
                                  maxlen=tf.shape(object_features)[1])
  object_masks = tf.reshape(tile_fn(object_masks),
                            [batch_size * NUM_CHOICES, -1])

  object_ids = tf.reshape(tile_fn(object_ids), [batch_size * NUM_CHOICES, -1])

  object_features = tf.reshape(
      tile_fn(object_features),
      [batch_size * NUM_CHOICES, -1, object_features.shape[-1]])
  return object_masks, object_ids, object_features


def _ground_tag_using_object_features(object_features, tags):
  """Grounds tag sequence using object features.

  Args:
    object_features: A [batch, max_num_objects, feature_dims] float tensor.
    tags: A [batch * NUM_CHOICES, max_seq_len] int tensor.
  """
  # Add ZEROs to the end.
  batch_size, _, object_feature_dims = object_features.shape
  object_features_padded = tf.concat(
      [object_features,
       tf.zeros([batch_size, 1, object_feature_dims])], 1)

  # Tile object features.
  object_features_padded_and_tiled = tf.reshape(
      tf.gather(tf.expand_dims(object_features_padded, 1), [0] * NUM_CHOICES,
                axis=1), [batch_size * NUM_CHOICES, -1, object_feature_dims])

  # Gather tag embeddings.
  feature_indices = tf.stack([
      tf.tile(tf.expand_dims(tf.range(batch_size * NUM_CHOICES), axis=1),
              [1, tf.shape(tags)[1]]), tags
  ], -1)
  grounded_object_features = tf.gather_nd(object_features_padded_and_tiled,
                                          feature_indices)
  return grounded_object_features


class B2T2(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(B2T2, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.B2T2):
      raise ValueError('Options has to be an B2T2 proto.')

    if model_proto.rationale_model:
      self._field_answer_choices = InputFields.rationale_choices
      self._field_answer_choices_tag = InputFields.rationale_choices_tag
      self._field_answer_choices_len = InputFields.rationale_choices_len
      self._field_answer_label = InputFields.rationale_label
    else:
      self._field_answer_choices = InputFields.answer_choices
      self._field_answer_choices_tag = InputFields.answer_choices_tag
      self._field_answer_choices_len = InputFields.answer_choices_len
      self._field_answer_label = InputFields.answer_label

  def _bert_model(self,
                  input_ids,
                  input_tag_embeddings,
                  input_masks,
                  bert_config,
                  bert_checkpoint_file,
                  is_training=False):
    """Creates the Bert model.

    Args:
      input_ids: A [batch, max_seq_len] int tensor.
      input_masks: A [batch, max_seq_len] int tensor.
    """
    bert_model = BertModel(bert_config,
                           is_training,
                           input_ids=input_ids,
                           input_mask=input_masks,
                           use_tag_embeddings=True,
                           tag_embeddings=input_tag_embeddings)

    # Restore from checkpoint.
    assignment_map, _ = get_assignment_map_from_checkpoint(
        tf.global_variables(), bert_checkpoint_file)
    if 'global_step' in assignment_map:
      assignment_map.pop('global_step')
    tf.compat.v1.train.init_from_checkpoint(bert_checkpoint_file,
                                            assignment_map)
    return bert_model.get_pooled_output()

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    is_training = self._is_training
    options = self._model_proto

    token_to_id_layer = token_to_id.TokenToIdLayer(options.bert_vocab_file,
                                                   options.bert_unk_token_id)
    bert_config = BertConfig.from_json_file(options.bert_config_file)
    slim_fc_scope = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                  is_training)()

    # Predict object embedding vectors.
    (num_objects, object_bboxes, object_labels, object_scores, object_features,
     max_num_objects) = _trim_to_max_num_objects(
         inputs[InputFields.num_detections],
         inputs[InputFields.detection_boxes],
         inputs[InputFields.detection_classes],
         inputs[InputFields.detection_scores],
         inputs[InputFields.detection_features],
         max_num_objects=options.max_num_objects)

    object_features = _predict_object_embeddings(
        object_features,
        bert_config.hidden_size,
        slim_fc_scope,
        keep_prob=options.dropout_keep_prob,
        is_training=is_training)

    # Gather text inputs.
    (answer_choices, answer_choices_tag,
     answer_choices_len) = (inputs[self._field_answer_choices],
                            inputs[self._field_answer_choices_tag],
                            inputs[self._field_answer_choices_len])
    batch_size = answer_choices.shape[0]

    answer_choices_tag = _assign_invalid_tags(answer_choices_tag,
                                              max_num_objects)

    # Convert tokens into token ids.
    answer_choices_token_ids = token_to_id_layer(answer_choices)
    answer_choices_token_ids = tf.reshape(answer_choices_token_ids,
                                          [batch_size * NUM_CHOICES, -1])
    answer_choices_mask = tf.sequence_mask(answer_choices_len,
                                           maxlen=tf.shape(answer_choices)[-1])
    answer_choices_mask = tf.reshape(answer_choices_mask,
                                     [batch_size * NUM_CHOICES, -1])

    # Create tag features sequence.
    answer_choices_tag = tf.reshape(answer_choices_tag,
                                    [batch_size * NUM_CHOICES, -1])
    answer_choices_tag_embeddings = _ground_tag_using_object_features(
        object_features, answer_choices_tag)

    (tiled_object_masks, tiled_object_ids,
     tiled_object_features) = _tile_objects(num_objects,
                                            token_to_id_layer(object_labels),
                                            object_features)

    # Create Bert model.
    input_ids = tf.concat([answer_choices_token_ids, tiled_object_ids], -1)
    input_tag_embeddings = tf.concat(
        [answer_choices_tag_embeddings, tiled_object_features], 1)
    input_mask = tf.concat([answer_choices_mask, tiled_object_masks], -1)

    output = self._bert_model(input_ids,
                              input_tag_embeddings,
                              input_mask,
                              bert_config,
                              bert_checkpoint_file=options.bert_checkpoint_file,
                              is_training=is_training)

    # Classification layer.
    with slim.arg_scope(slim_fc_scope):
      output = slim.fully_connected(output,
                                    num_outputs=1,
                                    activation_fn=None,
                                    scope='logits')
      output = tf.reshape(output, [batch_size, NUM_CHOICES])

    return {FIELD_ANSWER_PREDICTION: output}

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    options = self._model_proto
    loss_fn = (tf.nn.sigmoid_cross_entropy_with_logits if options.use_sigmoid
               else tf.nn.softmax_cross_entropy_with_logits)

    labels = tf.one_hot(inputs[self._field_answer_label], NUM_CHOICES)
    losses = loss_fn(labels=labels, logits=predictions[FIELD_ANSWER_PREDICTION])

    return {'crossentropy': tf.reduce_mean(losses)}

  def build_metrics(self, inputs, predictions, **kwargs):
    """Compute evaluation metrics.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    accuracy_metric = tf.keras.metrics.Accuracy()
    y_true = inputs[self._field_answer_label]
    y_pred = tf.argmax(predictions[FIELD_ANSWER_PREDICTION], -1)

    accuracy_metric.update_state(y_true, y_pred)
    return {'metrics/accuracy': accuracy_metric}

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    options = self._model_proto
    trainable_variables = tf.compat.v1.trainable_variables()

    # Look for BERT frozen variables.
    frozen_variables = []
    for var in trainable_variables:
      for name_pattern in options.frozen_variable_patterns:
        if name_pattern in var.op.name:
          frozen_variables.append(var)
          break

    # Get trainable variables.
    var_list = list(set(trainable_variables) - set(frozen_variables))
    return var_list
