from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

PAD = '[PAD]'
NUM_CHOICES = 4
PAD_ID = 0


class TFExampleFields(object):
  """Fields in the tf.train.Example."""
  annot_id = 'annot_id'
  answer_label = 'answer_label'
  rationale_label = 'rationale_label'

  img_id = 'img_id'
  img_encoded = 'image/encoded'
  img_format = 'image/format'

  detection_classes = "image/object/bbox/label"
  detection_scores = "image/object/bbox/score"
  detection_features = 'image/object/bbox/feature'
  detection_boxes_ymin = 'image/object/bbox/ymin'
  detection_boxes_ymax = 'image/object/bbox/ymax'
  detection_boxes_xmin = 'image/object/bbox/xmin'
  detection_boxes_xmax = 'image/object/bbox/xmax'
  detection_boxes_scope = "image/object/bbox/"
  detection_boxes_keys = ['ymin', 'xmin', 'ymax', 'xmax']

  question = 'question'
  question_tag = 'question_tag'
  answer_choice = 'answer_choice'
  answer_choice_tag = 'answer_choice_tag'
  rationale_choice = 'rationale_choice'
  rationale_choice_tag = 'rationale_choice_tag'


class InputFields(object):
  """Names of the input tensors."""
  # Meta information.
  annot_id = 'annot_id'
  answer_label = 'answer_label'
  rationale_label = 'rationale_label'

  img_id = 'img_id'
  img_data = 'image'
  img_height = 'image_height'
  img_width = 'image_width'

  # Objects fields.
  num_detections = 'num_detections'
  detection_boxes = 'detection_boxes'
  detection_classes = 'detection_classes'
  detection_scores = 'detection_scores'
  detection_features = 'detection_features'

  # Question.
  question = 'question'
  question_tag = 'question_tag'
  question_len = 'question_len'
  answer = 'answer'
  answer_tag = 'answer_tag'
  answer_len = 'answer_len'

  # Answer choices.
  answer_choices = 'answer_choices'
  answer_choices_tag = 'answer_choices_tag'
  answer_choices_len = 'answer_choices_len'
  mixed_answer_choices = 'mixed_answer_choices'
  mixed_answer_choices_tag = 'mixed_answer_choices_tag'
  mixed_answer_choices_len = 'mixed_answer_choices_len'

  # Rationale choices.
  rationale_choices = 'rationale_choices'
  rationale_choices_tag = 'rationale_choices_tag'
  rationale_choices_len = 'rationale_choices_len'
  mixed_rationale_choices = 'mixed_rationale_choices'
  mixed_rationale_choices_tag = 'mixed_rationale_choices_tag'
  mixed_rationale_choices_len = 'mixed_rationale_choices_len'
