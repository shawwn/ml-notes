
import re
import time

from google.cloud import storage # sudo pip3 install google-cloud-storage

import tensorflow as tf


class State:
  pass


if 'state' not in globals():
  state = State()
  state.client = None


import ring

@ring.lru(expire=60)
def gs_filesize(filename):
  """tf.string.length unfortunately fails for files larger than 2GB due to its result being a 32-bit integer. Punt by asking gsutil for the filesize."""
  import subprocess
  result = int(subprocess.run(['gsutil', 'du', '-s', filename], stdout=subprocess.PIPE, check=True).stdout.split()[0])
  if result <= 0:
    raise FileNotFoundError("Blob path does not exist or is zero length: {!r}".format(filename))
  return result


def tf_file_contents(filename):
  size = gs_filesize(filename)
  data = tf.raw_ops.ReadFile(filename=filename);
  return data, size


def tf_file_data(filename, out_dtype=None):
  data, size = tf_file_contents(filename)
  if out_dtype == tf.string:
    out_dtype = None
  if out_dtype is not None:
    if size % out_dtype.size != 0:
      raise ValueError("Size of file isn't divisible by dtype size. File size: {!r} dtype size: {!r} dtype: {!r}".format(size, out_dtype.size, out_dtype))
    data = tf.io.decode_raw(data, out_dtype);
    data.set_shape((size // out_dtype.size,));
  return data, size

_VALID_SCOPE_NAME_REGEX = re.compile("^[A-Za-z0-9_.\\-/>]*$")
_VALID_OP_NAME_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\-/>]*$")

def tf_sanitize_op_name(name, invalid='_'):
  return ''.join([x if _VALID_OP_NAME_REGEX.match(x) else invalid for x in name])

def tf_file_shard(filename, out_dtype, current_host, num_hosts):
  data, size = tf_file_data(filename, out_dtype=out_dtype)
  n = data.shape[0].value
  #assert n % num_hosts == 0
  k = n // num_hosts
  i = current_host * k
  j = (current_host + 1) * k
  return data[i:j]

def tf_file_variable(filename, dtype, **kws):
  data, size = tf_file_data(filename, out_dtype=dtype)
  collections = kws.pop('collections', ['local_variables'])
  trainable = kws.pop('trainable', False)
  if 'name' in kws:
    name = kws.pop('name')
  else:
    name = tf_sanitize_op_name(filename)
  v = tf.Variable(data, dtype=dtype, collections=collections, trainable=trainable, name=name, **kws)
  return v

def tf_shard_variable(filename, dtype, current_host, num_hosts, **kws):
  data = tf_file_shard(filename, out_dtype=dtype, current_host=current_host, num_hosts=num_hosts)
  collections = kws.pop('collections', ['local_variables'])
  trainable = kws.pop('trainable', False)
  if 'name' in kws:
    name = kws.pop('name')
  else:
    name = tf_sanitize_op_name(filename + '_%05d_of_%05d' % (current_host, num_hosts))
  v = tf.Variable(data, dtype=dtype, collections=collections, trainable=trainable, name=name, **kws)
  return v


from tensorflow.core.protobuf import config_pb2
from functools import partial

def tf_session_run_timeout(session, timeout_in_ms=10000):
  return partial(session.run, options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms))


def tf_foo():
  print('foo2')


def is_cloud_path(x):
  return ':/' in x and x.index(':') < x.index('/')


def path_parse(x):
  if '/' not in x:
    return '', '', x
  root = ''
  if is_cloud_path(x):
    root, x = x.split(':/', 1)
    root += ':/'
  else:
    while x.startswith('/'):
      root += '/'
      x = x[1:]
  dirname = ''
  if '/' in x:
    dirname, x = x.rsplit('/', 1)
    dirname += '/'
  return root, dirname, x


def gs_client(client=None):
  if client is not None:
    return client
  if state.client is None:
    state.client = storage.Client()
  return state.client


def gs_path_parse(path):
  root, dirname, basename = path_parse(path)
  if root != 'gs:/':
    raise ValueError("expected path like gs://foo/bar, got {!r}".format(path))
  assert dirname.startswith('/')
  if dirname == '/':
    bucket = basename
    basename = ''
  else:
    bucket, dirname = dirname.lstrip('/').split('/', 1)
  blob = dirname.rstrip('/') + '/' + basename.lstrip('/')
  blob = blob.lstrip('/') # remove leading slash from blob path
  return bucket, blob


def gs_blob(path, client=None):
  bucket_name, blob_path = gs_path_parse(path)
  client = gs_client(client)
  bucket = client.get_bucket(bucket_name)
  blob = bucket.get_blob(blob_path)
  return blob


def gs_size(path):
  fp = gs_blob(path)
  return fp.size
  


