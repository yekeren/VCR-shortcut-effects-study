from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class ModelBase(abc.ABC):
  """Model interface."""

  def __init__(self, model_proto, is_training=False):
    """Initializes the model.

    Args:
      model_proto: A model_pb2.Model proto.
      is_training: if True, training graph will be built.
    """
    self._model_proto = model_proto
    self._is_training = is_training

  @abc.abstractmethod
  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    pass

  @abc.abstractmethod
  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    pass

  @abc.abstractmethod
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
    pass

  def get_variables_to_train(self):
    """Returns model variables.
      
    Returns:
      A list of trainable model variables.
    """
    return None

  def get_scaffold(self):
    """Returns a scaffold object used to initialize variables.

    Returns:
      A tf.train.Scaffold instance.
    """
    return None
