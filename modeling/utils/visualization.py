from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import tensorflow as tf

from object_detection.utils.visualization_utils import STANDARD_COLORS
from object_detection.utils.visualization_utils import draw_bounding_box_on_image_array


def visualization_py_func_fn(*args):
  """Visualization function that can be wrapped in a tf.py_func.

  Args:
    *args: First 4 positional arguments must be:
      image - uint8 numpy array with shape (height, width, 3).
      total - a integer denoting the actual number of boxes.
      boxes - a numpy array of shape [max_pad_num, 4].
      labels - a numpy array of shape [max_pad_num].
      scores - a numpy array of shape [max_pad_num].

  Returns:
    uint8 numpy array with shape (height, width, 3) with overlaid boxes.
  """
  image, total, boxes, labels, scores = args
  for i in range(total):
    ymin, xmin, ymax, xmax = boxes[i]
    display_str = '%i%% %s' % (int(scores[i] * 100), labels[i].decode('utf8'))
    color = STANDARD_COLORS[i % len(STANDARD_COLORS)]

    draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color,
                                     display_str_list=[display_str])
  return image


def draw_bounding_boxes_on_image_tensors(image, total, boxes, labels, scores):
  """Draws bounding boxes on batch of image tensors.

  Args:
    image: A [batch, height, width, 3] uint8 tensor.
    total: A [batch] int tensor denoting number of boxes.
    boxes: A [batch, max_pad_num, 4] float tensor, normalized boxes,
      in the format of [ymin, xmin, ymax, xmax].
    labels: A [batch, max_pad_num] string tensor denoting labels.
    scores: A [batch, max_pad_num] float tensor denoting scores.

  Returns:
    a [batch, height, width, 3] uint8 tensor with boxes drawn on top.
  """

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    image_with_boxes = tf.py_func(visualization_py_func_fn,
                                  image_and_detections, tf.uint8)
    return image_with_boxes

  elems = [image, total, boxes, labels, scores]
  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images
