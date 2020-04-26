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


def extract_ground_truth_annotation(choices, choices_tag, choices_len, label):
  """Extracts the ground-truth using the label.

  Args:
    choices: A [batch, NUM_CHOICES, pad_choice_len] string tensor.
    choices_tag: A [batch, NUM_CHOICES, pad_choice_len] int tensor.
    choices_len: A [batch, NUM_CHOICES] int tensor.

  Returns:
    choice: A [batch, pad_choice_len] string tensor.
    choice_len: A [batch] int tensor.
  """
  batch_size = choices.shape[0]
  choice_indices = tf.stack([tf.range(batch_size), label], -1)
  choice = tf.gather_nd(choices, choice_indices)
  choice_tag = tf.gather_nd(choices_tag, choice_indices)
  choice_len = tf.gather_nd(choices_len, choice_indices)
  return choice, choice_tag, choice_len


def ground_detection_features(detection_features, tags):
  """Grounds tag sequence using detection features.

  Args:
    detection_features: A [batch, max_num_detections, feature_dims] float tensor.
    tags: A [batch, NUM_CHOICES, max_seq_len] int tensor.

  Returns:
    grounded_detection_features: A [batch, NUM_CHOICES, max_seq_len, 
      feature_dims] float tensor.
  """
  # Add ZEROs to the end.
  batch_size, _, dims = detection_features.shape
  detection_features_padded = tf.concat(
      [detection_features, tf.zeros([batch_size, 1, dims])], 1)

  indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1),
                    [1, tf.shape(tags)[1]])
  indices = tf.stack([indices, tags], -1)
  return tf.gather_nd(detection_features_padded, indices)


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
      input_ids: A [batch, 1 + max_detections + 1 + max_caption_len + 1] int tensor.
      input_masks: A [batch, 1 + max_detections + 1 + max_caption_len + 1] boolean tensor.
      input_features: A [batch, 1 + max_detections + 1 + max_caption_len + 1, dims] float tensor.
    """
    batch_size = num_detections.shape[0]
    token_to_id_func = self._token_to_id_func

    # Create input masks.
    mask_one = tf.fill([batch_size, 1], True)
    max_caption_len = tf.shape(caption)[1]
    max_detections = tf.shape(detection_features)[1]
    input_masks = tf.concat([
        mask_one,
        tf.sequence_mask(num_detections, maxlen=max_detections), mask_one,
        tf.sequence_mask(caption_len, maxlen=max_caption_len), mask_one
    ], -1)

    # Create input tokens.
    token_cls = tf.fill([batch_size, 1], CLS)
    token_sep = tf.fill([batch_size, 1], SEP)
    detection_classes_masked = tf.fill([batch_size, max_detections], MASK)
    input_tokens = tf.concat(
        [token_cls, detection_classes_masked, token_sep, caption, token_sep],
        axis=-1)
    input_ids = token_to_id_func(input_tokens)

    # Create input features.
    feature_dims = detection_features.shape[-1]
    feature_zeros = tf.fill([batch_size, 1, feature_dims], 0.0)
    tag_features = tf.fill([batch_size, max_caption_len, feature_dims], 0.0)
    input_features = tf.concat([
        feature_zeros, detection_features, feature_zeros, tag_features,
        feature_zeros
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
                               caption_tag,
                               caption_len,
                               mask_prob=0.15):
    """Predicts the matching score of the given image-text pair.

    Args:
      num_detections: A [batch] int tensor.
      detection_classes: A [batch, max_detections] string tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption: A [batch, max_caption_len] string tensor.
      caption_tag: A [batch, max_caption_len] int tensor.
      caption_tag_feature: A [batch, max_caption_len, dims] float tensor.
      caption_len: A [batch] int tensor.

    Returns:
      masked_ids: Actual token ids that is masked out, a [B] int tensor.
      masked_logits: Prediction of the `[MASK]` tokens, a [B, vocab_size] float tensor.
    """
    # Mask the caption with a probability of `mask_prob`.
    batch_size = num_detections.shape[0]
    max_caption_len = tf.shape(caption)[1]

    masked_lm_mask = tf.less_equal(
        tf.random.uniform([batch_size, max_caption_len], 0.0, 1.0), mask_prob)
    masked_lm_mask = tf.logical_and(
        masked_lm_mask, tf.sequence_mask(caption_len, maxlen=max_caption_len))
    # TODO: DONT predict person name.
    # masked_lm_mask = tf.logical_and(masked_lm_mask, caption_tag != 'PERSON')

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
                          detection_features, caption, caption_tag,
                          caption_len):
    """Predicts the matching score of the given image-text pair.

    Args:
      num_detections: A [batch] int tensor.
      detection_classes: A [batch, max_detections] string tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption: A [batch, max_caption_len] string tensor.
      caption_tag: A [batch, max_caption_len] int tensor.
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
    img_id = inputs[InputFields.img_id]
    (num_detections, detection_boxes, detection_classes, detection_scores,
     detection_features) = (
         inputs[InputFields.num_detections],
         inputs[InputFields.detection_boxes],
         inputs[InputFields.detection_classes],
         inputs[InputFields.detection_scores],
         inputs[InputFields.detection_features],
     )
    batch_size = num_detections.shape[0]

    # Project Fast-RCNN features.
    with slim.arg_scope(self._slim_fc_scope):
      detection_features = self.project_detection_features(detection_features)

    # Preprocess answers and rationales.
    (answer, answer_tag, answer_len) = extract_ground_truth_annotation(
        inputs[InputFields.answer_choices],
        inputs[InputFields.answer_choices_tag],
        inputs[InputFields.answer_choices_len],
        inputs[InputFields.answer_label],
    )
    (rationale, rationale_tag, rationale_len) = extract_ground_truth_annotation(
        inputs[InputFields.rationale_choices],
        inputs[InputFields.rationale_choices_tag],
        inputs[InputFields.rationale_choices_len],
        inputs[InputFields.rationale_label],
    )

    # Pre-training task 1: Masked Language Modeling (MLM).
    predictions = {}

    reuse = False
    for choice, choice_tag, choice_len, choice_type in [
        (answer, answer_tag, answer_len, 'answer'),
        (rationale, rationale_tag, rationale_len, 'rationale')
    ]:
      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        mlm_labels, mlm_logits = self.masked_language_modeling(
            num_detections,
            detection_classes,
            detection_features,
            choice,
            choice_tag,
            choice_len,
            mask_prob=options.mask_probability)
        predictions.update({
            'mlm/{}/logits'.format(choice_type): mlm_logits,
            'mlm/{}/labels'.format(choice_type): mlm_labels,
        })
        reuse = True

    # Pre-training task 2: Image-Text Matching (ITM).
    if options.use_image_text_matching_task:

      # Generate binary labels.
      img_ids_list = []
      base_indices = tf.range(batch_size, dtype=tf.int32)
      for index_offset in range(batch_size):
        indices = (base_indices + index_offset) % batch_size
        img_ids_list.append(tf.gather(img_id, indices, axis=0))
      img_ids = tf.stack(img_ids_list, 1)
      itm_labels = tf.equal(img_ids, tf.expand_dims(img_ids_list[0], 1))
      itm_labels = tf.cast(itm_labels, tf.float32)
      predictions.update({'itm/img_ids': img_ids, 'itm/labels': itm_labels})

      for choice, choice_tag, choice_len, choice_type in [
          (answer, answer_tag, answer_len, 'answer'),
          (rationale, rationale_tag, rationale_len, 'rationale')
      ]:
        # Generate prediction for either answer or rationale.
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          bert_features = []
          for index_offset in range(batch_size):
            indices = (base_indices + index_offset) % batch_size
            bert_features.append(
                self.image_text_matching(num_detections, detection_classes,
                                         detection_features,
                                         tf.gather(choice, indices, axis=0),
                                         tf.gather(choice_tag, indices, axis=0),
                                         tf.gather(choice_len, indices,
                                                   axis=0)))

        # Generate logits.
        with slim.arg_scope(self._slim_fc_scope):
          itm_logits = slim.fully_connected(
              tf.stack(bert_features, 1),
              num_outputs=1,
              activation_fn=None,
              scope='itm/{}/logits'.format(choice_type))
          itm_logits = tf.squeeze(itm_logits, -1)
        predictions.update({
            'itm/{}/logits'.format(choice_type): itm_logits,
        })
      # END for answer, rationale

    # Restore from BERT checkpoint.
    assignment_map, _ = get_assignment_map_from_checkpoint(
        tf.global_variables(), options.bert_checkpoint_file)
    if 'global_step' in assignment_map:
      assignment_map.pop('global_step')
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

    loss_dict = {}

    for choice_type in ['answer', 'rationale']:
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=predictions['mlm/{}/labels'.format(choice_type)],
          logits=predictions['mlm/{}/logits'.format(choice_type)])
      loss_dict['mlm_{}_sparse_softmax_cross_entropy'.format(
          choice_type)] = tf.reduce_mean(losses)

      if options.use_image_text_matching_task:
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=predictions['itm/labels'],
            logits=predictions['itm/{}/logits'.format(choice_type)])
        loss_dict['itm_{}_sigmoid_cross_entropy'.format(
            choice_type)] = tf.reduce_mean(losses)
    return loss_dict

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

    metric_dict = {}

    for choice_type in ['answer', 'rationale']:
      # MLM metrics.
      accuracy = tf.keras.metrics.Accuracy()
      true = predictions['mlm/{}/labels'.format(choice_type)]
      pred = tf.argmax(predictions['mlm/{}/logits'.format(choice_type)], -1)
      accuracy.update_state(true, pred)
      metric_dict['metrics/mlm_{}_accuracy'.format(choice_type)] = accuracy

      # ITM metrics.
      if options.use_image_text_matching_task:
        true = predictions['itm/labels']
        pred = predictions['itm/{}/logits'.format(choice_type)]

        accuracy = tf.keras.metrics.Accuracy()
        accuracy.update_state(tf.argmax(true, -1), tf.argmax(pred, -1))
        metric_dict['metrics/itm_{}_accuracy_neg@all'.format(
            choice_type)] = accuracy

        accuracy = tf.keras.metrics.Accuracy()
        accuracy.update_state(tf.argmax(true[:, :2], -1),
                              tf.argmax(pred[:, :2], -1))
        metric_dict['metrics/itm_{}_accuracy_neg@1'.format(
            choice_type)] = accuracy
    return metric_dict

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
