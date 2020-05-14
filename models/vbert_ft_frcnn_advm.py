from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json
import numpy as np
import tensorflow as tf

from protos import model_pb2
from modeling.models import fast_rcnn
from modeling.models import rnn
from modeling.utils import hyperparams
from modeling.utils import visualization
from modeling.utils import checkpoints
from modeling.utils import masked_ops
from models.model_base import ModelBase

from readers.vcr_fields import InputFields
from readers.vcr_fields import NUM_CHOICES

import bert_vcr.modeling as bert_modeling
import tf_slim as slim

# UNK = '[UNK]'
# CLS = '[CLS]'
# SEP = '[SEP]'
# MASK = '[MASK]'
# IMG = '[unused400]'

UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103
IMG_ID = 405

INF = 1e10
EPSILON = 0.01

RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical


def remove_detections(num_detections,
                      detection_boxes,
                      detection_classes,
                      detection_scores,
                      max_num_detections=10):
  """Trims to the `max_num_detections` objects.

  Args:
    num_detections: A [batch] int tensor.
    detection_boxes: A [batch, pad_num_detections, 4] float tensor.
    detection_classes: A [batch, pad_num_detections] int tensor.
    detection_scores: A [batch, pad_num_detections] float tensor.
    max_num_detections: Expected maximum number of detections.

  Returns:
    max_num_detections: Maximum detection boxes.
    num_detections: A [batch] int tensor.
    detection_boxes: A [batch, max_num_detections, 4] float tensor.
    detection_classes: A [batch, max_num_detections] int tensor.
    detection_scores: A [batch, max_num_detections] float tensor.
  """
  max_num_detections = tf.minimum(tf.reduce_max(num_detections),
                                  max_num_detections)

  num_detections = tf.minimum(max_num_detections, num_detections)
  detection_boxes = detection_boxes[:, :max_num_detections, :]
  detection_classes = detection_classes[:, :max_num_detections]
  detection_scores = detection_scores[:, :max_num_detections]
  return (max_num_detections, num_detections, detection_boxes,
          detection_classes, detection_scores)


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


def preprocess_tags(tags, max_num_detections):
  """Preprocesses tags.

  Args:
    tags: A [batch, NUM_CHOICES, max_caption_len] int tensor.
    max_num_detections: A scalar int tensor.
  """
  ones = tf.ones_like(tags, tf.int32)
  return tf.where(tags >= max_num_detections, -ones, tags)


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
  detection_features = tf.concat(
      [detection_features, tf.zeros([batch_size, 1, dims])], 1)

  tag_features = []
  base_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1),
                         [1, tf.shape(tags)[-1]])
  for tag_tensor in tf.unstack(tags, axis=1):
    indices = tf.stack([base_indices, tag_tensor], -1)
    tag_features.append(tf.gather_nd(detection_features, indices))

  return tf.stack(tag_features, axis=1)


class VBertFtFrcnnAdvM(ModelBase):
  """Finetune the VBert model to solve the VCR task."""

  def __init__(self, model_proto, is_training):
    super(VBertFtFrcnnAdvM, self).__init__(model_proto, is_training)

    if not isinstance(model_proto, model_pb2.VBertFtFrcnnAdvM):
      raise ValueError('Options has to be an VBertFtFrcnnAdvM proto.')

    options = model_proto

    self._bert_config = bert_modeling.BertConfig.from_json_file(
        options.bert_config_file)

    self._slim_fc_scope = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                        is_training)()

    if options.rationale_model:
      self._field_label = InputFields.rationale_label
      self._field_choices = InputFields.mixed_rationale_choices
      self._field_choices_tag = InputFields.mixed_rationale_choices_tag
      self._field_choices_len = InputFields.mixed_rationale_choices_len
    else:
      self._field_label = InputFields.answer_label
      self._field_choices = InputFields.mixed_answer_choices
      self._field_choices_tag = InputFields.mixed_answer_choices_tag
      self._field_choices_len = InputFields.mixed_answer_choices_len

  def project_detection_features(self, detection_features):
    """Projects detection features to embedding space.

    Args:
      detection_features: Detection features.

    Returns:
      embeddings: Projected detection features.
    """
    is_training = self._is_training
    options = self._model_proto

    if options.detection_adaptation == model_pb2.MLP:

      detection_features = slim.fully_connected(
          detection_features,
          options.detection_mlp_hidden_units,
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

    elif options.detection_adaptation == model_pb2.LINEAR:

      detection_features = slim.fully_connected(detection_features,
                                                self._bert_config.hidden_size,
                                                activation_fn=None,
                                                scope='detection/adaptation')
      return detection_features

    raise ValueError('Invalid detection adaptation method.')

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
            activation=bert_modeling.get_activation(bert_config.hidden_act),
            kernel_initializer=bert_modeling.create_initializer(
                bert_config.initializer_range))
        input_tensor = bert_modeling.layer_norm(input_tensor)

      # The output weights are the same as the input embeddings, but there is
      # an output-only bias for each token.
      output_bias = tf.get_variable("output_bias",
                                    shape=[bert_config.vocab_size],
                                    initializer=tf.zeros_initializer())
      logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
    return logits

  def create_bert_input_tensors(self, num_detections, detection_boxes,
                                detection_classes, detection_scores,
                                detection_features, caption_ids,
                                caption_tag_ids, caption_tag_features,
                                caption_adv_masks, caption_len):
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

    # Create adversarial masks.
    mask_zero = tf.fill([batch_size, 1], 0.0)
    input_adv_masks = tf.concat([
        mask_zero,
        tf.fill([batch_size, max_detections], 0.0), mask_zero,
        caption_adv_masks, mask_zero
    ], 1)

    return input_ids, input_masks, input_tag_masks, input_tag_features, input_adv_masks

  def image_text_matching(self, num_detections, detection_boxes,
                          detection_classes, detection_scores,
                          detection_features, caption_ids, caption_tag_ids,
                          caption_tag_features, caption_adv_masks,
                          caption_length):
    """Predicts the matching score of the given image-text pair.

    Args:
      num_detections: A [batch] int tensor.
      detection_classes: A [batch, max_detections] string tensor.
      detection_features: A [batch, max_detections, dims] float tensor.
      caption_ids: A [batch, max_caption_len] int tensor.
      caption_tag_ids: A [batch, max_caption_len] int tensor.
      caption_tag_features: A [batch, max_caption_len, dims] float tensor.
      caption_length: A [batch] int tensor.

    Returns:
      bert_feature: A [batch, 1 + max_detections + 1 + max_caption_len + 1, dims] float tensor.
      embedding_table: A [vocab_size, dims] float tensor.
    """
    (input_ids, input_masks, input_tag_masks,
     input_tag_features, input_adv_masks) = self.create_bert_input_tensors(
         num_detections, detection_boxes, detection_classes, detection_scores,
         detection_features, caption_ids, caption_tag_ids, caption_tag_features,
         caption_adv_masks, caption_length)
    bert_model = bert_modeling.BertModel(self._bert_config,
                                         self._is_training,
                                         input_ids=input_ids,
                                         input_mask=input_masks,
                                         input_tag_mask=input_tag_masks,
                                         input_tag_feature=input_tag_features,
                                         input_adv_masks=input_adv_masks,
                                         scope='bert')
    return (bert_model.get_pooled_output(), bert_model.get_sequence_output(),
            bert_model.get_embedding_table())

  def rnn_with_label_embedding(self, choice_ids, choice_lengths, labels):
    """Creates RNN architectures with label embeddings.
    """
    pass

  def generate_adversarial_masks(self,
                                 choice_ids,
                                 choice_lengths,
                                 question_lengths,
                                 labels,
                                 hard=True):
    """Masked language modeling."""
    options = self._model_proto
    is_training = self._is_training

    batch_size = choice_ids.shape[0]
    max_choice_len = tf.shape(choice_ids)[-1]

    with tf.variable_scope('adversarial'):
      # Lookup for token embeddings.
      # Note: DONOT share it with BERT, use a brand new embedding matrix instead.
      with tf.variable_scope("embeddings", reuse=False):
        (choice_embeddings_reshaped, _) = bert_modeling.embedding_lookup(
            input_ids=tf.reshape(choice_ids, [batch_size * NUM_CHOICES, -1]),
            vocab_size=self._bert_config.vocab_size,
            embedding_size=self._bert_config.hidden_size,
            initializer_range=self._bert_config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=False)
        choice_lengths_reshaped = tf.reshape(choice_lengths, [-1])

      # Create label embedding.
      full_label_embeddings = tf.get_variable(
          name='label_embedding',
          shape=[2, self._bert_config.hidden_size],
          initializer=bert_modeling.create_initializer(
              self._bert_config.initializer_range))
      one_hot_labels = tf.one_hot(labels, NUM_CHOICES, on_value=1, off_value=0)
      label_embeddings = tf.nn.embedding_lookup(full_label_embeddings,
                                                one_hot_labels)
      label_embeddings_reshaped = tf.reshape(
          label_embeddings,
          [batch_size * NUM_CHOICES, 1, self._bert_config.hidden_size])

      # Layer norm.
      choice_embeddings_reshaped = bert_modeling.layer_norm_and_dropout(
          choice_embeddings_reshaped + label_embeddings_reshaped,
          dropout_prob=self._bert_config.hidden_dropout_prob)

      # RNN.
      choice_features_reshaped, _ = rnn.RNN(choice_embeddings_reshaped,
                                            choice_lengths_reshaped,
                                            options=options.adversarial_rnn,
                                            is_training=is_training)

      # Fully-connected layer
      choice_features = tf.reshape(
          choice_features_reshaped,
          [batch_size, NUM_CHOICES, -1, choice_features_reshaped.shape[-1]])
      choice_shortcut_logits = slim.fully_connected(choice_features,
                                                    num_outputs=1,
                                                    activation_fn=None,
                                                    scope='logits')
      choice_shortcut_logits = tf.squeeze(choice_shortcut_logits, -1)
    # END - with tf.variable_scope('adversarial'):

    # Gumbel-Softmax to get the probable shortcut.
    choice_masks = tf.logical_and(
        tf.sequence_mask(choice_lengths, maxlen=max_choice_len),
        tf.logical_not(tf.sequence_mask(question_lengths,
                                        maxlen=max_choice_len)))
    choice_masks = tf.cast(choice_masks, tf.float32)

    tvar = tf.Variable(np.sqrt(options.initial_temperature),
                       name='adversarial/temperature_var',
                       trainable=True)
    temperature = tf.square(tvar) + EPSILON

    tf.summary.histogram('shortcut/logtis', choice_shortcut_logits)
    tf.summary.scalar('metrics/temperature', temperature)

    choice_shortcut_logits = choice_shortcut_logits - INF * (1.0 - choice_masks)
    a_sample = RelaxedOneHotCategorical(temperature,
                                        logits=choice_shortcut_logits,
                                        allow_nan_stats=False).sample()

    if hard:
      k = tf.shape(choice_shortcut_logits)[-1]
      a_hard_sample = tf.cast(tf.one_hot(tf.argmax(a_sample, -1), k),
                              a_sample.dtype)
      a_sample = tf.stop_gradient(a_hard_sample - a_sample) + a_sample

    # Returns the mask sampled from the distribution.
    return a_sample, choice_shortcut_logits, choice_features

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

    # Decode image fields from `inputs`.
    (image, image_height, image_width, num_detections, detection_boxes,
     detection_classes, detection_scores) = (
         inputs[InputFields.img_data],
         inputs[InputFields.img_height],
         inputs[InputFields.img_width],
         inputs[InputFields.num_detections],
         inputs[InputFields.detection_boxes],
         inputs[InputFields.detection_classes],
         inputs[InputFields.detection_scores],
     )
    batch_size = image.shape[0]
    (max_num_detections, num_detections, detection_boxes, detection_classes,
     detection_scores) = remove_detections(
         num_detections,
         detection_boxes,
         detection_classes,
         detection_scores,
         max_num_detections=options.max_num_detections)

    # Extract Fast-RCNN features.
    image_batch_shape = tf.shape(image)
    detection_boxes = convert_to_batch_coordinates(detection_boxes,
                                                   image_height, image_width,
                                                   image_batch_shape[1],
                                                   image_batch_shape[2])
    detection_features, _ = fast_rcnn.FastRCNN(image,
                                               detection_boxes,
                                               options=options.fast_rcnn_config,
                                               is_training=is_training)
    predictions.update({'detection_features': detection_features})
    with slim.arg_scope(self._slim_fc_scope):
      detection_features = self.project_detection_features(detection_features)

    # Ground objects.
    (choice_ids, choice_tag_ids,
     choice_lengths) = (inputs[self._field_choices],
                        inputs[self._field_choices_tag],
                        inputs[self._field_choices_len])

    choice_tag_ids = preprocess_tags(choice_tag_ids, max_num_detections)
    choice_tag_features = ground_detection_features(detection_features,
                                                    choice_tag_ids)
    choice_ids_raw = choice_ids

    # Create MLM predictions for masked tokens.

    if options.rationale_model:
      assert False
    else:
      question_lengths = 1 + inputs[InputFields.question_len]

    (choice_adv_masks, choice_shortcut_logits,
     choice_embeddings) = self.generate_adversarial_masks(
         choice_ids,
         choice_lengths,
         question_lengths,
         labels=inputs[self._field_label])

    if not is_training:
      choice_adv_masks = tf.zeros_like(choice_adv_masks, dtype=tf.float32)
    else:
      choice_adv_masks = choice_adv_masks
      random_masks = tf.less_equal(
          tf.random.uniform(
              [batch_size, NUM_CHOICES,
               tf.shape(choice_adv_masks)[-1]], 0.0, 1.0), 0.5)
      random_masks = tf.cast(random_masks, tf.float32)
      choice_adv_masks = choice_adv_masks * random_masks

    predictions.update({
        'adversarial_masks': choice_adv_masks,
        'shortcut_logits': choice_shortcut_logits,
        'shortcut_probas': tf.nn.softmax(choice_shortcut_logits),
        'choice_ids': choice_ids,
        'label': inputs[self._field_label],
        'choice_embeddings': choice_embeddings,
    })

    choice_ids_list = tf.unstack(choice_ids, axis=1)
    choice_tag_ids_list = tf.unstack(choice_tag_ids, axis=1)
    choice_tag_features_list = tf.unstack(choice_tag_features, axis=1)
    choice_lengths_list = tf.unstack(choice_lengths, axis=1)
    choice_adv_masks_list = tf.unstack(choice_adv_masks, axis=1)

    # Create ITM predictions for matching the ansewrs.
    reuse = False
    feature_to_predict_choices = []
    feature_to_predict_masks = []
    for caption_ids, caption_tag_ids, caption_tag_features, caption_adv_masks, caption_length in zip(
        choice_ids_list, choice_tag_ids_list, choice_tag_features_list,
        choice_adv_masks_list, choice_lengths_list):
      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        (bert_output,
         bert_sequence_output, embedding_table) = self.image_text_matching(
             num_detections, detection_boxes, detection_classes,
             detection_scores, detection_features, caption_ids, caption_tag_ids,
             caption_tag_features, caption_adv_masks, caption_length)
        feature_to_predict_choices.append(bert_output)
        feature_to_predict_masks.append(bert_sequence_output)
      reuse = True
    bert_outputs = tf.stack(feature_to_predict_choices, 1)
    bert_sequence_outputs = tf.stack(feature_to_predict_masks, 1)

    with slim.arg_scope(self._slim_fc_scope):
      logits = slim.fully_connected(bert_outputs,
                                    num_outputs=1,
                                    activation_fn=None,
                                    scope='itm/logits')
    predictions.update({'answer_prediction': tf.squeeze(logits, -1)})

    # Restore from BERT checkpoint.
    assignment_map, _ = checkpoints.get_assignment_map_from_checkpoint(
        [
            x for x in tf.global_variables()
            if x.op.name.startswith('bert') or x.op.name.startswith('cls')
        ],  # IMPORTANT to filter using `bert` and `cls`.
        options.bert_checkpoint_file)
    tf.train.init_from_checkpoint(options.bert_checkpoint_file, assignment_map)

    assignment_map = {
        'bert/embeddings/word_embeddings':
            'adversarial/embeddings/word_embeddings'
    }
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
    loss_dict = {'crossentropy': tf.reduce_mean(losses)}

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

    # Primary metric.
    y_true = inputs[self._field_label]
    y_pred = tf.argmax(predictions['answer_prediction'], -1)

    accuracy_metric = tf.keras.metrics.Accuracy()
    accuracy_metric.update_state(y_true, y_pred)
    metric_dict = {'metrics/accuracy': accuracy_metric}

    return metric_dict

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

    # Look for adversarial variables.
    adversarial_variables = [
        var for var in trainable_variables
        if var.op.name.startswith('adversarial/')
    ]

    # Get trainable variables.
    var_list = list(
        set(trainable_variables) - set(frozen_variables) -
        set(adversarial_variables))
    return var_list

  def get_adversarial_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    options = self._model_proto
    trainable_variables = tf.compat.v1.trainable_variables()

    # Look for adversarial variables.
    adversarial_variables = [
        var for var in trainable_variables
        if var.op.name.startswith('adversarial/')
    ]

    return adversarial_variables
