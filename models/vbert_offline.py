from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
import tf_slim as slim

from protos import model_pb2

import bert2.modeling as modeling
from bert2.modeling import BertConfig
from bert2.modeling import BertModel
from bert2.modeling import get_assignment_map_from_checkpoint

from readers.vcr_fields import InputFields
from readers.vcr_fields import NUM_CHOICES

from modeling.layers import token_to_id
from modeling.models import fast_rcnn
from modeling.utils import hyperparams
from modeling.utils import visualization
from models.model_base import ModelBase

UNK = '[UNK]'
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'

IMG = '[unused400]'


def remove_detections(num_detections,
                      detection_scores,
                      detection_classes,
                      detection_boxes,
                      detection_features,
                      max_num_detections=10):
  """Trims to the `max_num_detections` objects.

  Args:
    num_detections: A [batch] int tensor.
    detection_scores: A [batch, pad_num_detections] float tensor.
    detection_classes: A [batch, pad_num_detections] int tensor.
    detection_boxes: A [batch, pad_num_detections, 4] float tensor.
    max_num_detections: Maximum number of objects.

  Returns:
    num_detections: A [batch] int tensor.
    detection_scores: A [batch, max_num_detections] float tensor.
    detection_classes: A [batch, max_num_detections] int tensor.
    detection_boxes: A [batch, max_num_detections, 4] float tensor.
  """
  max_num_detections = tf.minimum(tf.reduce_max(num_detections),
                                  max_num_detections)

  num_detections = tf.minimum(max_num_detections, num_detections)
  detection_boxes = detection_boxes[:, :max_num_detections, :]
  detection_classes = detection_classes[:, :max_num_detections]
  detection_scores = detection_scores[:, :max_num_detections]
  detection_features = detection_features[:, :max_num_detections, :]
  return (max_num_detections, num_detections, detection_scores,
          detection_classes, detection_boxes, detection_features)


def trim_captions(caption, caption_len, max_caption_len):
  """Trims the captions to the `max_caption_len`.

  Args:
    caption: A [batch, pad_caption_len] string tensor.
    caption_len: A [batch] int tensor.
    max_caption_len: Maximum caption length.

  Returns:
    caption: A [batch, max_caption_len] string tensor.
    caption_len: A [batch] int tensor.
  """
  max_caption_len = tf.minimum(tf.reduce_max(caption_len), max_caption_len)

  caption = caption[:, :max_caption_len]
  caption_len = tf.minimum(max_caption_len, caption_len)
  return caption, caption_len


def convert_to_batch_coordinates(detection_boxes, height, width, batch_height,
                                 batch_width):
  """Converts the coordinates to be relative to the batch images. """
  height = tf.expand_dims(tf.cast(height, tf.float32), -1)
  width = tf.expand_dims(tf.cast(width, tf.float32), -1)
  batch_height = tf.cast(batch_height, tf.float32)
  batch_width = tf.cast(batch_width, tf.float32)

  ymin, xmin, ymax, xmax = tf.unstack(detection_boxes, axis=-1)
  detection_boxes_converted = tf.stack([
      ymin * height / batch_height, xmin * width / batch_width,
      ymax * height / batch_height, xmax * width / batch_width
  ], -1)
  return detection_boxes_converted


def insert_image_box(num_detections, detection_scores, detection_classes,
                     detection_boxes):
  """Insert image-box to the 0-th position. """
  batch_size = num_detections.shape[0]

  num_detections += 1
  detection_scores = tf.concat(
      [tf.fill([batch_size, 1], 1.0), detection_scores], axis=-1)
  detection_classes = tf.concat(
      [tf.fill([batch_size, 1], IMG), detection_classes], axis=-1)
  detection_boxes = tf.concat([
      tf.gather(tf.constant([[[0, 0, 1, 1]]], dtype=tf.float32),
                [0] * batch_size,
                axis=0), detection_boxes
  ], 1)
  return num_detections, detection_scores, detection_classes, detection_boxes


class VBertOffline(ModelBase):
  """Wraps the BiLSTM layer to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VBertOffline, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VBertOffline):
      raise ValueError('Options has to be an VBertOffline proto.')

    options = model_proto

    self._token_to_id_func = token_to_id.TokenToIdLayer(
        options.bert_vocab_file, options.bert_unk_token_id)
    self._bert_config = BertConfig.from_json_file(options.bert_config_file)

    self._slim_fc_scope = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                        is_training)()

  def project_detection_features(self, detection_features):
    """Projects detection features to embedding space.

    Args:
      detection_features: Detection features.

    Returns:
      embeddings: Projected detection features.
    """
    is_training = self._is_training
    options = self._model_proto

    if options.detection_adaptation == model_pb2.LINEAR:
      detection_features = slim.fully_connected(detection_features,
                                                self._bert_config.hidden_size,
                                                activation_fn=None,
                                                scope='detection/project')
      return detection_features

    elif options.detection_adaptation == model_pb2.MLP:
      detection_features = slim.fully_connected(detection_features,
                                                self._bert_config.hidden_size,
                                                activation_fn=tf.nn.relu,
                                                scope='detection/project')
      detection_features = slim.dropout(detection_features,
                                        keep_prob=options.dropout_keep_prob,
                                        is_training=is_training)
      detection_features = slim.fully_connected(detection_features,
                                                self._bert_config.hidden_size,
                                                activation_fn=None,
                                                scope='detection/adaptation')
      return detection_features

    elif options.detection_adaptation == model_pb2.MLP_RES:
      detection_features = slim.fully_connected(detection_features,
                                                self._bert_config.hidden_size,
                                                activation_fn=None,
                                                scope='detection/project')
      detection_features_res = slim.dropout(tf.nn.relu(detection_features),
                                            keep_prob=options.dropout_keep_prob,
                                            is_training=is_training)
      detection_features_res = slim.fully_connected(
          detection_features_res,
          self._bert_config.hidden_size,
          activation_fn=None,
          scope='detection/adaptation')
      return detection_features + detection_features_res

    raise ValueError('Invalid `detection_adaptation`.')

  def create_bert_input_tensors(self, num_detections, detection_classes,
                                detection_features, caption, caption_len):
    """Predicts the matching score of the given image-text pair.

    Args:
      num_detections: A [batch] int tensor.
      detection_classes: A [batch, max_detections] string tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption: A [batch, max_caption_len] string tensor.
      caption_len: A [batch] int tensor.

    Returns:
      input_ids: A [batch, 1 + max_detections + max_caption_len + 1] int tensor.
      input_masks: A [batch, 1 + max_detections + max_caption_len + 1] boolean tensor.
      input_features: A [batch, 1 + max_detections + max_caption_len + 1, dims] float tensor.
    """
    batch_size = num_detections.shape[0]
    token_to_id_func = self._token_to_id_func

    # Create input masks.
    mask_one = tf.fill([batch_size, 1], True)
    max_caption_len = tf.shape(caption)[1]
    max_detections = tf.shape(detection_features)[1]
    input_masks = tf.concat([
        mask_one,
        tf.sequence_mask(num_detections, maxlen=max_detections),
        tf.sequence_mask(caption_len, maxlen=max_caption_len), mask_one
    ], -1)

    # Create input tokens.
    token_cls = tf.fill([batch_size, 1], CLS)
    token_sep = tf.fill([batch_size, 1], SEP)
    detection_classes_masked = tf.fill([batch_size, max_detections], MASK)
    input_tokens = tf.concat(
        [token_cls, detection_classes_masked, caption, token_sep],
        axis=-1)
    input_ids = token_to_id_func(input_tokens)

    # Create input features.
    feature_dims = detection_features.shape[-1]
    input_features = tf.concat([
        tf.fill([batch_size, 1, feature_dims], 0.0), detection_features,
        tf.fill([batch_size, max_caption_len + 1, feature_dims], 0.0)
    ], 1)
    return input_ids, input_masks, input_features

  def decode_bert_output(self, input_tensor, output_weights):
    """Decodes bert output.

    Args:
      input_tensor: A [batch, hidden_size] float tensor.
      output_weights: A [vocab_size, hidden_size] float tensor, the embedding matrix.

    Returns:
      logits: A [batch, vocab_size] float tensor.
    """
    bert_config = self._bert_config

    with tf.variable_scope("cls/predictions"):
      # We apply one more non-linear transformation before the output layer.
      # This matrix is not used after pre-training.
      with tf.variable_scope("transform"):
        input_tensor = tf.layers.dense(
            input_tensor,
            units=bert_config.hidden_size,
            activation=modeling.get_activation(bert_config.hidden_act),
            kernel_initializer=modeling.create_initializer(
                bert_config.initializer_range))
        input_tensor = modeling.layer_norm(input_tensor)

      # The output weights are the same as the input embeddings, but there is
      # an output-only bias for each token.
      output_bias = tf.get_variable("output_bias",
                                    shape=[bert_config.vocab_size],
                                    initializer=tf.zeros_initializer())
      logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
    return logits

  def masked_language_modeling(self,
                               num_detections,
                               detection_classes,
                               detection_features,
                               caption,
                               caption_len,
                               mask_prob=0.15):
    """Predicts the matching score of the given image-text pair.

    Args:
      num_detections: A [batch] int tensor.
      detection_classes: A [batch, max_detections] string tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption: A [batch, max_caption_len] string tensor.
      caption_len: A [batch] int tensor.

    Returns:
      masked_ids: Actual token ids that is masked out, a [B] int tensor.
      masked_logits: Prediction of the `[MASK]` tokens, a [B, vocab_size] float tensor.
    """
    # Mask the caption with a probability of `mask_prob`.
    batch_size = num_detections.shape[0]
    max_caption_len = tf.shape(caption)[1]

    masked_lm_mask = tf.logical_and(
        tf.sequence_mask(caption_len, maxlen=max_caption_len),
        tf.less_equal(
            tf.random.uniform([batch_size, max_caption_len], 0.0, 1.0),
            mask_prob))
    masked_caption = tf.where(masked_lm_mask,
                              tf.fill([batch_size, max_caption_len], MASK),
                              caption)

    # BERT sequence prediction using the `masked_caption`.
    (input_ids, input_masks, input_features) = self.create_bert_input_tensors(
        num_detections, detection_classes, detection_features, masked_caption,
        caption_len)
    bert_model = BertModel(self._bert_config,
                           self._is_training,
                           input_ids=input_ids,
                           input_mask=input_masks,
                           input_features=input_features,
                           scope='bert')
    sequence_output = bert_model.get_sequence_output()

    # Decode the predict words.
    masked_lm_positions = tf.where(masked_lm_mask)
    masked_tokens = tf.gather_nd(caption, masked_lm_positions)

    predicted_features = tf.gather_nd(sequence_output, masked_lm_positions)
    predicted_logits = self.decode_bert_output(
        predicted_features, output_weights=bert_model.get_embedding_table())

    return self._token_to_id_func(masked_tokens), predicted_logits

  def image_text_matching(self, num_detections, detection_classes,
                          detection_features, caption, caption_len):
    """Predicts the matching score of the given image-text pair.

    Args:
      num_detections: A [batch] int tensor.
      detection_classes: A [batch, max_detections] string tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption: A [batch, max_caption_len] string tensor.
      caption_len: A [batch] int tensor.

    Returns:
      matching_score: A [batch] float tensor.
    """
    (input_ids, input_masks, input_features) = self.create_bert_input_tensors(
        num_detections, detection_classes, detection_features, caption,
        caption_len)
    bert_model = BertModel(self._bert_config,
                           self._is_training,
                           input_ids=input_ids,
                           input_mask=input_masks,
                           input_features=input_features,
                           scope='bert')
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

    # Decode fields from `inputs`.
    (num_detections, detection_boxes, detection_classes, detection_scores,
     detection_features) = (
         inputs[InputFields.num_detections],
         inputs[InputFields.detection_boxes],
         inputs[InputFields.detection_classes],
         inputs[InputFields.detection_scores],
         inputs[InputFields.detection_features],
     )
    (answer_choices, answer_choices_len, answer_label) = (
        inputs[InputFields.answer_choices],
        inputs[InputFields.answer_choices_len],
        inputs[InputFields.answer_label],
    )
    batch_size = num_detections.shape[0]

    # Remove boxes if there are too many.
    (max_num_detections, num_detections, detection_scores, detection_classes,
     detection_boxes,
     detection_features) = remove_detections(num_detections,
                                             detection_scores,
                                             detection_classes,
                                             detection_boxes,
                                             detection_features,
                                             max_num_detections=10)

    # Extract groundtruth answer.
    answer_indices = tf.stack([tf.range(batch_size), answer_label], -1)
    caption = tf.gather_nd(answer_choices, answer_indices)
    caption_len = tf.gather_nd(answer_choices_len, answer_indices)

    # Remove boxes/tokens if there are too many.
    (_, num_detections, detection_scores, detection_classes, detection_boxes,
     detection_features) = remove_detections(
         num_detections,
         detection_scores,
         detection_classes,
         detection_boxes,
         detection_features,
         max_num_detections=options.max_num_detections)
    caption, caption_len = trim_captions(
        caption, caption_len, max_caption_len=options.max_caption_len)

    # Project Fast-RCNN features.
    with slim.arg_scope(self._slim_fc_scope):
      detection_features = self.project_detection_features(detection_features)

    # Pre-training task 2: Masked Language Modeling (MLM).
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
      mlm_labels, mlm_logits = self.masked_language_modeling(
          num_detections,
          detection_classes,
          detection_features,
          caption,
          caption_len,
          mask_prob=options.mask_probability)

    # Pre-training task 1: Image-Text Matching (ITM).
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      feature_to_predict_1 = self.image_text_matching(num_detections,
                                                      detection_classes,
                                                      detection_features,
                                                      caption, caption_len)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      feature_to_predict_0 = []
      base_indices = tf.range(batch_size, dtype=tf.int32)
      for index_offset in range(1, batch_size):
        negative_indices = (base_indices + index_offset) % batch_size
        feature_to_predict_0.append(
            self.image_text_matching(
                num_detections, detection_classes, detection_features,
                tf.gather(caption, negative_indices, axis=0),
                tf.gather(caption_len, negative_indices, axis=0)))
        if options.sample_one:
          break

    with slim.arg_scope(self._slim_fc_scope):
      itm_feature = tf.stack([feature_to_predict_1] + feature_to_predict_0, 1)
      itm_logits = slim.fully_connected(itm_feature,
                                        num_outputs=1,
                                        activation_fn=None,
                                        scope='itm/logits')
      itm_logits = tf.squeeze(itm_logits, -1)

    itm_labels = tf.concat([
        tf.fill([batch_size, 1], 1.0),
        tf.fill([batch_size, itm_logits.shape[-1] - 1], 0.0)
    ], -1)

    # Restore from BERT checkpoint.
    assignment_map, _ = get_assignment_map_from_checkpoint(
        tf.global_variables(), options.bert_checkpoint_file)
    if 'global_step' in assignment_map:
      assignment_map.pop('global_step')
    tf.train.init_from_checkpoint(options.bert_checkpoint_file, assignment_map)

    return {
        'itm_logits': itm_logits,
        'itm_labels': itm_labels,
        'mlm_logits': mlm_logits,
        'mlm_labels': mlm_labels
    }

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    options = self._model_proto

    itm_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=predictions['itm_labels'], logits=predictions['itm_logits'])
    mlm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=predictions['mlm_labels'], logits=predictions['mlm_logits'])
    return {
        'itm_sigmoid_cross_entropy': tf.reduce_mean(itm_losses),
        'mlm_sparse_softmax_cross_entropy': tf.reduce_mean(mlm_losses),
    }

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

    # ITM metrics.
    itm_accuracy_neg_1 = tf.keras.metrics.Accuracy()
    itm_accuracy_neg_all = tf.keras.metrics.Accuracy()

    itm_logits = predictions['itm_logits']
    itm_pred_neg_1 = tf.argmax(itm_logits[:, :2], -1)
    itm_pred_neg_all = tf.argmax(itm_logits, -1)
    itm_true = tf.argmax(predictions['itm_labels'], -1)

    itm_accuracy_neg_1.update_state(itm_true, itm_pred_neg_1)
    itm_accuracy_neg_all.update_state(itm_true, itm_pred_neg_all)

    # MLM metrics.
    mlm_accuracy = tf.keras.metrics.Accuracy()
    mlm_true = predictions['mlm_labels']
    mlm_pred = tf.argmax(predictions['mlm_logits'], -1)
    mlm_accuracy.update_state(mlm_true, mlm_pred)

    return {
        'metrics/itm_accuracy_neg_1': itm_accuracy_neg_1,
        'metrics/itm_accuracy_neg_all': itm_accuracy_neg_all,
        'metrics/mlm_accuracy': mlm_accuracy,
    }

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    options = self._model_proto

    if options.freeze_first_stage_feature_extractor:
      trainable_variables = [
          x for x in tf.trainable_variables()
          if 'FirstStageFeatureExtractor' not in x.op.name
      ]
      return trainable_variables

    return None
