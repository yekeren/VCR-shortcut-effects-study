from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json
import numpy as np
import tensorflow as tf
from collections import namedtuple

from protos import model_pb2
from modeling.models import fast_rcnn
from modeling.utils import hyperparams
from modeling.utils import visualization
from modeling.utils import checkpoints
from modeling.utils import masked_ops
from models.model_base import ModelBase

from modeling.models import bert_modeling
import tf_slim as slim

CLS_ID = 101
SEP_ID = 102

NUM_CHOICES = 4


class Detections(object):

  def __init__(self, num_detections, detection_boxes, detection_classes,
               detection_scores):
    """Initializes the fields in Detections object."""
    self.num_detections = num_detections
    self.detection_boxes = detection_boxes
    self.detection_classes = detection_classes
    self.detection_scores = detection_scores
    self.detection_features = None
    self.detection_masks = tf.sequence_mask(num_detections,
                                            maxlen=tf.shape(detection_boxes)[1])

  def __repr__(self):
    """Returns the debug information."""
    str_list = []
    str_list.append('Detections:')
    str_list.append('num_detections: %s' % self.num_detections.__repr__())
    str_list.append('detection_boxes: %s' % self.detection_boxes.__repr__())
    str_list.append('detection_classes: %s' % self.detection_classes.__repr__())
    str_list.append('detection_scores: %s' % self.detection_scores.__repr__())
    if self.detection_features is not None:
      str_list.append('detection_features: %s' %
                      self.detection_features.__repr__())
    str_list.append('detection_masks: %s' % self.detection_masks.__repr__())
    return '\n'.join(str_list)

  def remove_detections(self, max_num_detections):
    """Trims to the `max_num_detections` objects.

    Args:
      max_num_detections: Expected maximum number of detections.

    Returns:
      max_num_detections: The actual maximum number of detections.
    """
    max_num_detections = tf.minimum(max_num_detections,
                                    tf.reduce_max(self.num_detections))
    self.num_detections = tf.minimum(max_num_detections, self.num_detections)
    self.detection_boxes = self.detection_boxes[:, :max_num_detections, :]
    self.detection_classes = self.detection_classes[:, :max_num_detections]
    self.detection_scores = self.detection_scores[:, :max_num_detections]
    self.detection_masks = tf.sequence_mask(self.num_detections,
                                            maxlen=max_num_detections)
    return max_num_detections

  def convert_to_batch_coordinates(self, height, width, batch_height,
                                   batch_width):
    """Converts the coordinates to be relative to the batch images. """
    height = tf.expand_dims(tf.cast(height, tf.float32), -1)
    width = tf.expand_dims(tf.cast(width, tf.float32), -1)
    batch_height = tf.cast(batch_height, tf.float32)
    batch_width = tf.cast(batch_width, tf.float32)

    ymin, xmin, ymax, xmax = tf.unstack(self.detection_boxes, axis=-1)
    detection_boxes_converted = tf.stack([
        ymin * height / batch_height, xmin * width / batch_width,
        ymax * height / batch_height, xmax * width / batch_width
    ], -1)
    return detection_boxes_converted


class MixedSequence(object):

  def __init__(self, length, tokens, tags):
    """Initializes the fields in MixedSequence object."""
    self.length = length
    self.tokens = tokens
    self.tags = tags
    self.tag_features = None
    self.sequence_masks = tf.sequence_mask(length, maxlen=tf.shape(tokens)[1])

  def __repr__(self):
    """Returns the debug information."""
    str_list = []
    str_list.append('MixedSequence:')
    str_list.append('length: %s' % self.length.__repr__())
    str_list.append('tokens: %s' % self.tokens.__repr__())
    str_list.append('tags: %s' % self.tags.__repr__())
    if self.tag_features is not None:
      str_list.append('tag_features: %s' % self.tag_features.__repr__())
    str_list.append('sequence_masks: %s' % self.sequence_masks.__repr__())
    return '\n'.join(str_list)

  def remove_tags(self, max_num_detections):
    """Removes the tags if they are referring to the non-existing objects.
  
    Args:
      max_num_detections: The actual maximum number of detections.
  
    Returns:
      set `self.tags`.
    """
    self.tags = tf.where(self.tags >= max_num_detections,
                         -tf.ones_like(self.tags, tf.int32), self.tags)

  def ground_detections(self, detections):
    """Grounds detection features.
  
    Args:
      detections: A Detections object.

    Returns:
      set `self.tag_features`.
    """
    detection_features = detections.detection_features
    batch_size, _, dims = detection_features.shape

    detection_features_padded = tf.concat(
        [detection_features,
         tf.zeros([batch_size, 1, dims])], 1)

    batch_indices = tf.expand_dims(tf.range(batch_size), axis=1)
    batch_indices = tf.tile(batch_indices, [1, tf.shape(self.tags)[-1]])
    indices = tf.stack([batch_indices, self.tags], -1)

    self.tag_features = tf.gather_nd(detection_features_padded, indices)


class VBertFtFrcnnV2(ModelBase):
  """Finetune the VBert model to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VBertFtFrcnnV2, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VBertFtFrcnnV2):
      raise ValueError('Options has to be an VBertFtFrcnn proto.')

    options = model_proto

    self._bert_config = bert_modeling.BertConfig.from_json_file(
        options.bert_config_file)

    self._slim_fc_scope = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                        is_training)()

    if options.rationale_model:
      assert False
    else:
      self._field_label = 'answer_label'
      self._field_choice = 'answer_choice'

  def adapt_detection_features(self, detection_features):
    """Projects detection features to embedding space.

    Args:
      detection_features: Detection features.

    Returns:
      embeddings: Projected detection features.
    """
    is_training = self._is_training
    options = self._model_proto

    with tf.variable_scope('detection'):
      detection_features = slim.fully_connected(
          detection_features,
          options.detection_mlp_hidden_units,
          activation_fn=tf.nn.relu,
          scope='hidden')
      detection_features = slim.dropout(detection_features,
                                        keep_prob=options.dropout_keep_prob,
                                        is_training=is_training)
      detection_features = slim.fully_connected(detection_features,
                                                self._bert_config.hidden_size,
                                                activation_fn=None,
                                                scope='output')
    return detection_features

  def create_bert_input_tensors(self, num_detections, detection_boxes,
                                detection_classes, detection_scores,
                                detection_features, caption_ids,
                                caption_tag_ids, caption_tag_features,
                                caption_len):
    """Predicts the matching score of the given image-text pair.

    [CLS] [IMG1] [IMG2] ... [SEP] [TOKEN1] [TOKEN2] ... [SEP]

    Args:
      num_detections: A [batch] int tensor.
      detection_boxes: A [batch, max_detections, 4] float tensor.
      detection_classes: A [batch, max_detections] int tensor.
      detection_scores : A [batch, max_detections] float tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption_ids: A [batch, max_caption_len] int tensor.
      caption_tag_ids: A [batch, max_caption_len] int tensor.
      caption_tag_features: A [batch, max_caption_len, dims] int tensor.
      caption_length: A [batch] int tensor.

    Returns:
      input_ids: Token ids.
        A [batch, 1 + max_detections + 1 + max_caption_len + 1] int tensor.
      input_masks: Sequence masks.
        A [batch, 1 + max_detections + 1 + max_caption_len + 1] boolean tensor.
      input_tag_masks: Denoting whether to replace the word embedding.
        A [batch, 1 + max_detections + 1 + max_caption_len + 1] boolean tensor.
      input_tag_features: Features to replace word embedding.
        A [batch, 1 + max_detections + 1 + max_caption_len + 1, dims] float tensor.
    """
    batch_size = num_detections.shape[0]
    max_caption_len = tf.shape(caption_ids)[1]
    max_detections = tf.shape(detection_features)[1]

    # Create input ids.
    id_cls = tf.fill([batch_size, 1], CLS_ID)
    id_sep = tf.fill([batch_size, 1], SEP_ID)

    input_ids = tf.concat(
        [id_cls, detection_classes, id_sep, caption_ids, id_sep], axis=-1)

    # Create input masks.
    mask_true = tf.fill([batch_size, 1], True)
    input_masks = tf.concat([
        mask_true,
        tf.sequence_mask(num_detections, maxlen=max_detections), mask_true,
        tf.sequence_mask(caption_len, maxlen=max_caption_len), mask_true
    ], -1)

    # Create input tag masks, to denote if the token is actually a tag.
    #   Detection class labels are replaced with FRCNN features.
    mask_false = tf.fill([batch_size, 1], False)
    input_tag_masks = tf.concat([
        mask_false,
        tf.sequence_mask(num_detections, maxlen=max_detections), mask_false,
        tf.greater(caption_tag_ids, -1), mask_false
    ], -1)

    # Create input tag features, to replace BERT word embedding if specified.
    zeros = tf.fill([batch_size, 1, detection_features.shape[-1]], 0.0)
    input_tag_features = tf.concat([
        zeros,
        detection_features,
        zeros,
        caption_tag_features,
        zeros,
    ], 1)
    return input_ids, input_masks, input_tag_masks, input_tag_features

  def image_text_matching(self, detections, question, choice):
    """Predicts the matching score of the given image-text pair.

    Args:
      detections: A Detections object.
      question: A MixedSequence object.
      choice: A MixedSequence object.

    Returns:
      bert_output: A [batch, max_bert_sequence_len, dims] float tensor.
    """
    batch_size = detections.num_detections.shape[0]

    # Create `input_ids`.
    id_cls = tf.fill([batch_size, 1], CLS_ID)
    id_sep = tf.fill([batch_size, 1], SEP_ID)
    input_ids = tf.concat([
        id_cls, question.tokens, id_sep, choice.tokens, id_sep,
        detections.detection_classes, id_sep
    ], -1)

    # Create `input_masks`.
    mask_true = tf.fill([batch_size, 1], True)
    input_masks = tf.concat([
        mask_true, question.sequence_masks, mask_true, choice.sequence_masks,
        mask_true, detections.detection_masks, mask_true
    ], -1)

    # Create `token_type_ids`.
    token_type_ids = tf.concat([
        tf.fill([batch_size, 2 + tf.shape(question.tokens)[1]], 0),
        tf.fill([batch_size, 1 + tf.shape(choice.tokens)[1]], 1),
        tf.fill([batch_size, 1 + tf.shape(detections.detection_classes)[1]], 2),
    ], -1)

    # Create `input_tag_features`.
    zeros = tf.fill([batch_size, 1, detections.detection_features.shape[-1]],
                    0.0)
    token_tag_features = tf.concat([
        zeros, question.tag_features, zeros, choice.tag_features, zeros,
        detections.detection_features, zeros
    ], 1)

    bert_model = bert_modeling.BertModel(self._bert_config,
                                         self._is_training,
                                         input_ids=input_ids,
                                         input_mask=input_masks,
                                         token_type_ids=token_type_ids,
                                         token_tag_features=token_tag_features,
                                         scope='bert')
    return bert_model.get_pooled_output()

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    predictions = {}

    is_training = self._is_training
    options = self._model_proto

    # Decode input features.
    image = inputs['img_data']
    image_width = inputs['img_width']
    image_height = inputs['img_height']

    batch_size = image.shape[0]
    image_batch_shape = tf.shape(image)

    detections = Detections(inputs['detections']['num_detections'],
                            inputs['detections']['detection_boxes'],
                            inputs['detections']['detection_classes'],
                            inputs['detections']['detection_scores'])
    question = MixedSequence(inputs['question']['length'],
                             inputs['question']['tokens'],
                             inputs['question']['tags'])
    choices = []
    for i in range(NUM_CHOICES):
      choice_field_key = self._field_choice + '_%i' % i
      choices.append(
          MixedSequence(inputs[choice_field_key]['length'],
                        inputs[choice_field_key]['tokens'],
                        inputs[choice_field_key]['tags']))

    # Remove extra detections if there are too many.
    # Then, extract the Fast-RCNN features.
    max_num_detections = detections.remove_detections(
        max_num_detections=options.max_num_detections)

    frcnn_boxes = detections.convert_to_batch_coordinates(
        height=image_height,
        width=image_width,
        batch_height=image_batch_shape[1],
        batch_width=image_batch_shape[2])
    frcnn_features, _ = fast_rcnn.FastRCNN(image,
                                           frcnn_boxes,
                                           options=options.fast_rcnn_config,
                                           is_training=is_training)
    with slim.arg_scope(self._slim_fc_scope):
      frcnn_features = self.adapt_detection_features(frcnn_features)
    detections.detection_features = frcnn_features

    # Create BERT VQA model.
    reuse = False
    bert_outputs = []
    question.remove_tags(max_num_detections)
    question.ground_detections(detections)
    for choice in choices:
      choice.remove_tags(max_num_detections)
      choice.ground_detections(detections)
      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        bert_outputs.append(
            self.image_text_matching(detections, question, choice))
      reuse = True
    bert_outputs = tf.stack(bert_outputs, 1)

    # Predicting the answer.
    with slim.arg_scope(self._slim_fc_scope):
      logits = slim.fully_connected(bert_outputs,
                                    num_outputs=1,
                                    activation_fn=None,
                                    scope='logits')
    predictions.update({'answer_prediction': tf.squeeze(logits, -1)})

    # Restore from BERT checkpoint.
    assignment_map, _ = checkpoints.get_assignment_map_from_checkpoint(
        [x for x in tf.global_variables() if x.op.name.startswith('bert')
        ],  # IMPORTANT to filter using `bert`.
        options.bert_checkpoint_file)
    tf.train.init_from_checkpoint(options.bert_checkpoint_file, assignment_map)

    return predictions

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    options = self._model_proto

    # Primary loss, predict the answer.
    loss_fn = (tf.nn.sigmoid_cross_entropy_with_logits
               if options.use_sigmoid_loss else
               tf.nn.softmax_cross_entropy_with_logits)
    labels = tf.one_hot(inputs[self._field_label], NUM_CHOICES)
    losses = loss_fn(labels=labels, logits=predictions['answer_prediction'])

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
    options = self._model_proto

    # Primary metric.
    accuracy_metric = tf.keras.metrics.Accuracy()
    y_true = inputs[self._field_label]
    y_pred = tf.argmax(predictions['answer_prediction'], -1)

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
