from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import reader_pb2
from readers import vcr_reader

_READERS = {
    'vcr_reader': vcr_reader,
}


def get_input_fn(options, is_training):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.Reader):
    raise ValueError('options has to be an instance of Reader.')

  reader_oneof = options.WhichOneof('reader_oneof')
  if not reader_oneof in _READERS:
    raise ValueError('Invalid reader %s!' % reader_oneof)

  return _READERS[reader_oneof].get_input_fn(getattr(options, reader_oneof),
                                             is_training=is_training)
