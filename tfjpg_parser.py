# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ImageNet preprocessing for ResNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

IMAGE_SIZE = 512
CROP_PADDING = 32

#FLAGS = flags.FLAGS
class Namespace:
  pass

FLAGS = Namespace()
FLAGS.cache_decoded_image = False


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label):
  """Build an Example proto for an example.

  Args:
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network

  Returns:
    Example proto
  """

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/class/label': _int64_feature(label),
              'image/encoded': _bytes_feature(image_buffer)
          }))
  return example


class ImageNet(object):

  @staticmethod
  def set_shapes(image_size, channels, transpose_input, train_batch_size, batch_size, num_cores, features, labels):
    """Statically set the batch_size dimension."""
    dick = isinstance(features, dict)
    images = features["images"] if dick else features
    if transpose_input:
      if train_batch_size // num_cores > 8:
        shape = [image_size, image_size, channels, batch_size]
      else:
        shape = [image_size, image_size, batch_size, channels]
      images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
      images = tf.reshape(images, [-1])
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, image_size, image_size, channels])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    if dick:
      features["images"] = images
    else:
      features = images
    return features, labels

  @staticmethod
  def dataset_parser_static(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

       This only decodes the image, which is prepared for caching.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/embedding': tf.VarLenFeature(tf.float32),
        'image/width': tf.FixedLenFeature([], tf.int64, -1),
        'image/height': tf.FixedLenFeature([], tf.int64, -1),
        'image/filename': tf.FixedLenFeature([], tf.string, ''),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    parsed['image/hash'] = tf.raw_ops.Fingerprint(data=[parsed['image/encoded']], method='farmhash64')[0]
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image = tf.io.decode_image(image_bytes, 3)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 0

    embedding = parsed['image/class/embedding'].values

    embedding = tf.cond(
        tf.math.greater(tf.shape(embedding)[0], 0),
        lambda: embedding,
        lambda: tf.one_hot(label, 1000))

    return {
      'image': image,
      'label': label,
      'embedding': embedding,
      'parsed': parsed,
    }

  @staticmethod
  def get_current_host(params):
    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      return params['context'].current_input_fn_deployment()[1]
    elif 'dataset_index' in params:
      return params['dataset_index']
    else:
      return 0

  @staticmethod
  def get_num_hosts(params):
    if 'context' in params:
     return params['context'].num_hosts
    elif 'dataset_index' in params:
      return params['dataset_num_shards']
    else:
      return 1

  @staticmethod
  def get_num_cores(params):
    return 8 * ImageNet.get_num_hosts(params)

  @staticmethod
  def make_dataset(data_dirs, index, num_hosts,
                   seed=None, shuffle_filenames=False,
                   num_parallel_calls = 64):

    if shuffle_filenames:
      assert seed is not None

    file_patterns = [x.strip() for x in data_dirs.split(',') if len(x.strip()) > 0]

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = None
    for pattern in file_patterns:
      x = tf.data.Dataset.list_files(pattern, shuffle=shuffle_filenames, seed=seed)
      dataset = x if dataset is None else dataset.concatenate(x)
    dataset = dataset.shard(num_hosts, index)

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=num_parallel_calls, sloppy=True))

    dataset = dataset.map(
        ImageNet.dataset_parser_static,
        num_parallel_calls=num_parallel_calls)

    return dataset



from tensorflow.python.framework.errors_impl import OutOfRangeError

def iterate_dataset(dataset, n = -1, session=None):
  if session is None:
    session = tf.get_default_session()
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  while n != 0:
    n -= 1
    try:
      yield session.run(next_element)
    except OutOfRangeError:
      return
