from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl import app
from absl import flags
from absl import logging

import hashlib
import io
import zipfile
import numpy as np
import PIL.Image
import tensorflow as tf

from bert import tokenization

from google.protobuf import text_format
from protos import rcnn_pb2
from modeling.models import rcnn

flags.DEFINE_string('rcnn_config', 'configs/rcnn/resnet152.pbtxt',
                    'Path to the RCNN config file.')

flags.DEFINE_string('annotations_jsonl_file', 'data/vcr1annots/val.jsonl',
                    'Path to the annotations file in jsonl format.')

flags.DEFINE_integer('num_shards', 10,
                     'Number of shards of the output tfrecord files.')

flags.DEFINE_integer('shard_id', 0, 'Shard id of the current process.')

flags.DEFINE_string('image_zip_file', 'data/vcr1images.zip',
                    'Path to the zip file of images.')

flags.DEFINE_string('output_rcnn_feature_dir', 'output/rcnn/resnet152',
                    'Path to the directory saving features.')

FLAGS = flags.FLAGS

_NUM_PARTITIONS = 100


def get_partition_id(annot_id, num_partitions=_NUM_PARTITIONS):
  split, number = annot_id.split('-')
  return int(number) % num_partitions


def _load_annotations(filename):
  """Loads annotations from file.

  Args:
    filename: Path to the jsonl annotations file.

  Returns:
    A list of python dictionary, each is parsed from a json object.
  """
  with tf.io.gfile.GFile(filename, 'r') as f:
    return [json.loads(x.strip('\n')) for x in f]


def main(_):
  logging.set_verbosity(logging.DEBUG)

  for i in range(_NUM_PARTITIONS):
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_rcnn_feature_dir,
                                      '%02d' % i))

  with tf.io.gfile.GFile(FLAGS.rcnn_config, 'r') as fp:
    rcnn_config = text_format.Merge(fp.read(), rcnn_pb2.RCNN())

  image_placeholder = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
  proposals_placeholder = tf.placeholder(shape=[None, 4], dtype=tf.float32)
  rcnn_features, _ = rcnn.RCNN(inputs=tf.expand_dims(image_placeholder, 0),
                               proposals=tf.expand_dims(proposals_placeholder,
                                                        0),
                               options=rcnn_config,
                               is_training=False)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.compat.v1.Session(config=config)

  sess.run(tf.global_variables_initializer())
  for name in sess.run(tf.compat.v1.report_uninitialized_variables()):
    logging.warn('%s is uninitialized!', name)

  # Load annotations.
  annots = _load_annotations(FLAGS.annotations_jsonl_file)
  logging.info('Loaded %i annotations.', len(annots))

  shard_id, num_shards = FLAGS.shard_id, FLAGS.num_shards
  assert 0 <= shard_id < num_shards

  with zipfile.ZipFile(FLAGS.image_zip_file) as image_zip:
    for idx, annot in enumerate(annots):
      if (idx + 1) % 100 == 0:
        logging.info('On example %i/%i.', idx + 1, len(annots))

      annot_id = int(annot['annot_id'].split('-')[-1])
      if annot_id % num_shards != shard_id:
        continue

      # Check npy file.
      part_id = get_partition_id(annot['annot_id'])
      output_file = os.path.join(FLAGS.output_rcnn_feature_dir,
                                 '%02d' % part_id, annot['annot_id'] + '.npy')
      if os.path.isfile(output_file):
        logging.info('%s is there.', output_file)
        continue

      # Read meta data.
      meta_fn = os.path.join('vcr1images', annot['metadata_fn'])
      try:
        with image_zip.open(meta_fn, 'r') as f:
          meta = json.load(f)
      except Exception as ex:
        logging.warn('Skip %s.', meta_fn)
        continue

      # Read image data.
      img_fn = os.path.join('vcr1images', annot['img_fn'])
      try:
        with image_zip.open(img_fn, 'r') as f:
          encoded_jpg = f.read()
      except Exception as ex:
        logging.warn('Skip %s.', img_fn)
        continue

      # Decode and resize image.
      image = PIL.Image.open(io.BytesIO(encoded_jpg))
      assert image.format == 'JPEG'

      image = np.array(image)
      boxes_and_scores = np.array(meta['boxes'])
      xmin, ymin, xmax, ymax, score = [boxes_and_scores[:, i] for i in range(5)]
      xmin /= meta['width']
      ymin /= meta['height']
      xmax /= meta['width']
      ymax /= meta['height']
      boxes = np.stack([ymin, xmin, ymax, xmax], -1)
      boxes = np.concatenate([[[0, 0, 1, 1]], boxes], 0)
      box_features = sess.run(rcnn_features,
                              feed_dict={
                                  image_placeholder: image,
                                  proposals_placeholder: boxes
                              })
      with open(output_file, 'wb') as f:
        np.save(f, box_features[0])

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
