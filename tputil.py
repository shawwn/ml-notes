
import re
import time
import tqdm

from google.cloud import storage # sudo pip3 install google-cloud-storage

import tensorflow as tf
tf1 = tf.compat.v1
from threading import Lock

from braces import braceexpand


class State:
  pass


if 'state' not in globals():
  state = State()
  state.client = None
  state.filesize_mutex = Lock()
  state.filesize_cache = {}


from tensorflow.python.data.experimental.ops import random_ops


def random_seeds(seed=None):
  if seed is not None:
    seed = tf.convert_to_tensor(seed, dtype=tf.int64)
    if len(seed.shape) == 1 and seed.shape[0].value == 2:
      return seed
  result = random_ops.RandomDataset(seed).batch(2).make_one_shot_iterator().get_next()
  result.set_shape([2])
  return result


def tf_sample(count, weights, dtype=tf.int32, seed=None):
  weights = tf.convert_to_tensor(weights)
  if len(weights.shape) <= 0:
    return tf.no_op()
  if len(weights.shape) <= 1:
    weights = tf.expand_dims(weights, axis=0)
  logits = tf.math.log(weights)
  seed = random_seeds(seed=seed)
  return tf.random.stateless_categorical(logits, count, seed=seed, dtype=dtype)[0]


from tensorflow.python.ops import control_flow_ops


def tf_infer_branch_dtype(branches):
  return control_flow_ops.cond_v2.indexed_case(tf.constant(0), branches).dtype


#tf.map_fn(lambda i: tf.switch_case(i, [lambda: tf.constant(42), lambda: tf.constant(99)], default=lambda: tf.constant(420)), tf_sample(160, [0.1, 1.0, 0.2], seed=0))


def tf_choice(choices, count=1, dtype=None, seed=None):
  choices = [x for x in choices]
  if len(choices) <= 0:
    return tf.no_op()
  weights = []
  branches = []
  for choice in choices:
    weight = 1.0
    value = None
    if isinstance(choice, (list, tuple)):
      if len(choice) >= 2:
        weight, value = choice[0:2]
      elif len(choice) >= 1:
        value = choice[0]
    elif isinstance(choice, dict):
      weight = choice.get('weight', 1.0)
      value = choice['value']
    else:
      value = choice
    if value is None:
      continue
    weights.append(weight)
    if callable(value):
      branches.append(value)
    else:
      def thunk(v):
        branches.append(lambda: v)
      thunk(tf.convert_to_tensor(value))
  indices = tf_sample(count, weights, seed=seed)
  #return weights, branches, indices
  if dtype is None:
    dtype = tf_infer_branch_dtype(branches)
  result = tf.map_fn(lambda i: tf.switch_case(i, branches), indices, dtype=dtype)
  if count == 1:
    result = result[0]
  return result



from tensorflow.python.data.experimental.ops import random_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


def sample_from(datasets, weights=None, seed=None):
  """Samples elements at random from the datasets in `datasets`.

  Args:
    datasets: A list of `tf.data.Dataset` objects with compatible structure.
    weights: (Optional.) A list of `len(datasets)` floating-point values where
      `weights[i]` represents the probability with which an element should be
      sampled from `datasets[i]`, or a `tf.data.Dataset` object where each
      element is such a list. Defaults to a uniform distribution across
      `datasets`.
    seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
      random seed that will be used to create the distribution. See
      `tf.random.set_seed` for behavior.

  Returns:
    A dataset that interleaves elements from `datasets` at random, according to
    `weights` if provided, otherwise with uniform probability.

  Raises:
    TypeError: If the `datasets` or `weights` arguments have the wrong type.
    ValueError: If the `weights` argument is specified and does not match the
      length of the `datasets` element.
  """
  num_datasets = len(datasets)
  if not isinstance(weights, dataset_ops.DatasetV2):
    if weights is None:
      # Select inputs with uniform probability.
      logits = [[1.0] * num_datasets]
    else:
      # Use the given `weights` as the probability of choosing the respective
      # input.
      weights = ops.convert_to_tensor(weights, name="weights")
      if weights.dtype not in (dtypes.float32, dtypes.float64):
        raise TypeError("`weights` must be convertible to a tensor of "
                        "`tf.float32` or `tf.float64` elements.")
      if not weights.shape.is_compatible_with([num_datasets]):
        raise ValueError(
            "`weights` must be a vector of length `len(datasets)`.")
      # The `stateless_multinomial()` op expects log-probabilities, as opposed
      # to weights.
      logits = array_ops.expand_dims(math_ops.log(weights, name="logits"), 0)
    # NOTE(mrry): We only specialize when `weights` is not a `Dataset`. When it
    # is a `Dataset`, it is possible that evaluating it has a side effect the
    # user depends on.
    if len(datasets) == 1:
      return datasets[0]
    def select_dataset_constant_logits(seed):
      return array_ops.squeeze(
          gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed),
          axis=[0, 1])
    selector_input = dataset_ops.MapDataset(
        random_ops.RandomDataset(seed).batch(2),
        select_dataset_constant_logits,
        use_inter_op_parallelism=False)
  else:
    # Use each element of the given `weights` dataset as the probability of
    # choosing the respective input.
    # The `stateless_multinomial()` op expects log-probabilities, as opposed to
    # weights.
    logits_ds = weights.map(lambda *p: math_ops.log(p, name="logits"))
    def select_dataset_varying_logits(logits, seed):
      return array_ops.squeeze(
          gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed),
          axis=[0, 1])
    logits_and_seeds = dataset_ops.Dataset.zip(
        (logits_ds, random_ops.RandomDataset(seed).batch(2)))
    selector_input = dataset_ops.MapDataset(
        logits_and_seeds,
        select_dataset_varying_logits,
        use_inter_op_parallelism=False)
  return selector_input



import ring

def expand_patterns(pattern):
  if isinstance(pattern, bytes):
    pattern = pattern.decode('utf8')
  results = []
  if not pattern.startswith('{') or not pattern.endswith('}'):
    pattern = '{' + pattern + '}'
  for result in braceexpand(pattern):
    if result.startswith('{') and result.endswith('}'):
      result = result[1:-1]
    results.append(result)
  return results


#@ring.lru(expire=60)
def gs_sizeof(pattern):
  out = subprocess.run(['gsutil', 'du', pattern], stdout=subprocess.PIPE, check=True).stdout
  if isinstance(out, bytes):
    out = out.decode('utf8')
  lines = out.splitlines()
  results = []
  for line in lines:
    size, name = line.split(maxsplit=1)
    size = int(size)
    results.append((name, size))
  return results
  # return [(name, int(size)) for size, name in line.split(maxsplit=1) for line in out.splitlines()]
  # for line in :
  #   if isinstance(line, bytes):
  #     line = line.decode('utf8')
  #   size, name = line.split(maxsplit=1)
  #   size = int(size)
  #   yield name, size
  # for line in lines:
  #   out = result.stdout
  #   size = int(out.split(maxsplit=1)[0])
  # else:
  #   result = subprocess.run(['gsutil', 'stat', pattern], stdout=subprocess.PIPE, check=True)
  # print(pattern, repr(out))
  # size = 0
  # return size


def gs_totalsize(patterns):
  if isinstance(patterns, bytes):
    patterns = patterns.decode('utf8')
  if isinstance(patterns, str):
    patterns = patterns.split(',')
  return sum(parallel(gs_sizeof, patterns, threads=4))


def gs_filesize(filename, force=False):
  """tf.string.length unfortunately fails for files larger than 2GB due to its result being a 32-bit integer. Punt by asking gsutil for the filesize."""
  if isinstance(filename, bytes):
    filename = filename.decode('utf8')
  import subprocess
  if not force:
    results = state.filesize_cache.get(filename)
    if results is not None:
      return results
  lines = subprocess.run(['gsutil', 'du', filename], stdout=subprocess.PIPE, check=True).stdout.splitlines()
  if len(lines) <= 0:
    raise FileNotFoundError("Blob path does not exist or is zero length: {!r}".format(filename))
  results = []
  for line in lines:
    size, name = line.split(maxsplit=1)
    if isinstance(name, bytes):
      name = name.decode('utf8')
    size = int(size)
    results.append([name, size])
    state.filesize_cache[name] = size
    if name == filename:
      return size
  state.filesize_cache[filename] = results
  return results


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
  v = tf1.Variable(data, dtype=dtype, collections=collections, trainable=trainable, name=name, **kws)
  return v

def tf_shard_variable(filename, dtype, current_host, num_hosts, **kws):
  data = tf_file_shard(filename, out_dtype=dtype, current_host=current_host, num_hosts=num_hosts)
  collections = kws.pop('collections', ['local_variables'])
  trainable = kws.pop('trainable', False)
  if 'name' in kws:
    name = kws.pop('name')
  else:
    name = tf_sanitize_op_name(filename + '_%05d_of_%05d' % (current_host, num_hosts))
  v = tf1.Variable(data, dtype=dtype, collections=collections, trainable=trainable, name=name, **kws)
  return v


# given a bin that holds `total` elements, return a random
# position such that you can take the next `subset` elements
# without going out of bounds. E.g. randpos(1,10) will return
# [0..9], randpos(2,10) will return [0..8], etc.
def randpos(subset, total, dtype=tf.int64, batch_size=1):
  assert subset <= total
  return tf.random.uniform([batch_size], maxval=(total - subset) + 1, dtype=dtype)


def sample(chunk, chunk_size, tokens_per_example, batch_size=1):
  pos = randpos(tokens_per_example, chunk_size, batch_size=batch_size)
  part = tf.tile(tf.expand_dims(tf.range(tokens_per_example, dtype=tf.int64), axis=0), [batch_size, 1])
  indices = part + tf.expand_dims(pos, axis=1)
  tokens = tf.gather(chunk, indices)
  return tokens


import math


def is_pow2(n):
  return math.log(n, 2).is_integer()


def sample_tokens(chunk, chunk_size, tokens_per_example, batch_size=1):
  pos = randpos(tokens_per_example + 1, chunk_size, batch_size=batch_size)
  part = tf.tile(tf.expand_dims(tf.range(tokens_per_example, dtype=tf.int64), axis=0), [batch_size, 1])
  indices = part + tf.expand_dims(pos, axis=1)
  feature = tf.gather(chunk, indices)
  label = tf.gather(chunk, indices + 1)
  feature = tf.cast(feature, dtype=tf.int32)
  label = tf.cast(label, dtype=tf.int32)
  factor = int(os.environ.get('SPATIAL_PARTITIONING', '1'))
  if factor > 1:
    assert is_pow2(tokens_per_example)
    N = batch_size
    H = math.sqrt(tokens_per_example)
    W = math.sqrt(tokens_per_example)
    C = 1
    tf.logging.info("Using SPATIAL_PARTITIONING=%d; reshaping tokens from [%d, %d] to [N=%d, H=%d, W=%d, C=1]",
        factor,
        batch_size, tokens_per_example, 
        N, H, W, C)
    feature = tf.reshape(feature, [N, H, W, C])
    label = tf.reshape(label, [N, H, W, C])
  return feature, label
     


def sample_text(chunk, amount, batch_size=1):
  #chunk_size = tf.size(chunk, out_type=tf.dtypes.int64)
  chunk_size = chunk.shape[0].value
  return sample_tokens(chunk=chunk, chunk_size=chunk_size, tokens_per_example=amount, batch_size=batch_size)


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
  


def tf_io_encode_raw(x):
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.string:
    return x
  unit_size = x.dtype.size
  total_size = tf.size(x, out_type=tf.int64) * unit_size
  serialized = tf.serialize_tensor(x)
  serialized_size = tf.size(tf.strings.bytes_split(serialized), out_type=tf.int64)
  offset = serialized_size - total_size
  return tf.strings.substr(serialized, offset, -1)


def tf_io_decode_raw(x, dtype):
  return tf.io.decode_raw(x, dtype)


def tf_encode(x):
  return tf_io_encode_raw(x)


def tf_decode(x, dtype):
  return tf.io.decode_raw(x, dtype)

def tf_glob(files):
  if isinstance(files, str):
    files = files.split(',')
  results = []
  for x in files:
    if '*' in x:
      results.extend(tf.io.gfile.glob(x))
    else:
      results.append(x)
  return results


from multiprocessing.dummy import Pool as ThreadPool

import traceback


def parallel(f, xs, threads=None, verbose=False, reraise_errors=True):
  pool = ThreadPool(threads)
  xs = [x for x in xs]
  n = len(xs)
  ys = [None] * n
  pbar = tqdm.tqdm(total=n) if verbose else None
  errors = set()
  def thunk(i):
    x = xs[i]
    try:
      y = f(x)
      ys[i] = y
    except Exception as e:
      if reraise_errors:
        ys[i] = e
        errors.add(i)
      else:
        traceback.print_exc()
    if pbar is not None:
      pbar.update(1)
  pool.map(thunk, range(n))
  pool.close()
  pool.join()
  if len(errors) > 0:
    raise list(errors)[0]
  return ys


def tf_globsize(files, threads=100, verbose=True):
  files = tf_glob(files)
  return parallel(lambda filename: (filename, gs_size(filename)), files, threads=threads, verbose=verbose)


class TFTok16Reader:
  def __init__(self, graph=None, name="tok16reader"):
    if graph is None:
      graph = tf.get_default_graph()
    with graph.as_default():
      self.queue = tf.FIFOQueue(100000, [tf.string], shapes=(), name=name+"_queue")
      self.reader = tf.FixedLengthRecordReader(
          record_bytes=2,
          header_bytes=0,
          footer_bytes=0,
          hop_bytes=0,
          encoding=None,
          name=name)
    self.files = tf.placeholder(tf.string, shape=(), name=name+"_files")
    self.count = tf.placeholder(tf.int64, shape=(), name=name+"_count")
    self.key, self.value = self.reader.read_up_to(self.queue, self.count)
    self.enqueue_op = self.queue.enqueue_many(tf.io.matching_files(self.files))
    read1k = tf.squeeze(tf.io.decode_raw(reader.reader.read_up_to(reader.queue, 1000)[1], tf.uint16), axis=-1); read1k.set_shape([1000])
    load_op = toks.scatter_nd_update(tf.expand_dims(tf.range(1000, dtype=tf.int64) + reader.reader.num_records_produced(), axis=-1), read1k)
  def load(self, pattern, session=None):
    if session is None:
      session = tf.get_default_session()
    assert session is not None
    patterns = pattern.split(',')
    for pattern in patterns:
      session.run(self.enqueue_op, {self.files: pattern})
  def read(self, session=None):
    if session is None:
      session = tf.get_default_session()
    assert session is not None
    return session.run((self.key, self.value))


      
      

# def tf_tok16_reader(filename):
#   FIFOQueue(


import numpy as np
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def _SparseTensorPlaceholder(dtype=None):
  if dtype is None:
    dtype = dtypes.int32
  return sparse_tensor_lib.SparseTensor(
      array_ops.placeholder(dtypes.int64),
      array_ops.placeholder(dtype), array_ops.placeholder(dtypes.int64))


def _SparseTensorValue_5x6(permutation):
  ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2],
                  [3, 3]]).astype(np.int64)
  val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)
  ind = ind[permutation]
  val = val[permutation]
  shape = np.array([5, 6]).astype(np.int64)
  return sparse_tensor_lib.SparseTensorValue(ind, val, shape)


def _SparseTensorValue_3x4(permutation):
  ind = np.array([[0, 0], [1, 0], [1, 2], [1, 3], [2, 2],
                  [2, 3]]).astype(np.int64)
  val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)
  ind = ind[permutation]
  val = val[permutation]
  shape = np.array([3, 4]).astype(np.int64)
  return sparse_tensor_lib.SparseTensorValue(ind, val, shape)


def _SparseTensorValue_1x1x1():
  ind = np.array([[0, 0, 0]]).astype(np.int64)
  val = np.array([0]).astype(np.int32)
  shape = np.array([3, 4, 5]).astype(np.int64)
  return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

from tensorflow.python.ops import variables

from functools import partial


class LocalVariable(variables.RefVariable):
  def __init__(self, initial_value=None, trainable=None,
      collections=None, validate_shape=None, caching_device=None,
      name=None, variable_def=None, dtype=None, expected_shape=None,
      import_scope=None, constraint=None, use_resource=None,
      synchronization=tf1.VariableSynchronization.AUTO,
      aggregation=tf1.VariableAggregation.NONE, shape=None):
    if collections is None:
      collections = [tf.GraphKeys.LOCAL_VARIABLES]
    if trainable is None:
      trainable = False
    if validate_shape is None:
      validate_shape = True
    if use_resource:
      raise NotImplementedError()
    super(LocalVariable, self).__init__(initial_value=initial_value,
        trainable=trainable, collections=collections,
        validate_shape=validate_shape, caching_device=caching_device,
        name=name, variable_def=variable_def, dtype=dtype,
        expected_shape=expected_shape, import_scope=import_scope,
        constraint=constraint,
        synchronization=synchronization, aggregation=aggregation,
        shape=shape)
  # def __new__(cls, initial_value=None, trainable=None,
  #     collections=None, validate_shape=None, caching_device=None,
  #     name=None, variable_def=None, dtype=None, expected_shape=None,
  #     import_scope=None, constraint=None, use_resource=None,
  #     synchronization=tf1.VariableSynchronization.AUTO,
  #     aggregation=tf1.VariableAggregation.NONE, shape=None):
  #   if collections is None:
  #     collections = [tf.GraphKeys.LOCAL_VARIABLES]
  #   if trainable is None:
  #     trainable = False
  #   if validate_shape is None:
  #     validate_shape = True
  #   if use_resource:
  #     raise NotImplementedError()
  #   if name is None:
  #     return super(LocalVariable, cls).__new__(cls)
  #   elif tf_hasvar(name):
  #     existing = tf_var(name)
  #     # TODO: check shape and dtype is compatible
  #     return existing
  #   else:
  #     self = super(LocalVariable, cls).__new__(cls)
  #     tf_var(name, 
  #     return super(LocalVariable, cls).__new__(cls)
  #   self = super(LocalVariable, cls).__new__(cls)
  #   return self
  # # initial_value=initial_value,
  # #         trainable=trainable, collections=collections,
  # #         validate_shape=validate_shape, caching_device=caching_device,
  # #         name=name, variable_def=variable_def, dtype=dtype,
  # #         expected_shape=expected_shape, import_scope=import_scope,
  # #         constraint=constraint,
  # #         synchronization=synchronization, aggregation=aggregation,
  # #         shape=shape)


from tensorflow.python.ops import variable_scope as vs
    

def tf_varstore():
  #tf.get_collection(('__variable_store',))[0]
  return vs._get_default_variable_store()


def tf_vars():
  return tf_varstore()._vars


def tf_varname(name):
  #return os.path.join(tf.get_variable_scope().name, name)
  with tf.variable_scope(name) as scope:
    return scope.name


def tf_hasvar(name):
  fqn = tf_varname(name)
  return fqn in tf_vars()


def absolute_name_scope(scope):
    return tf.name_scope(scope + "/")


def absolute_variable_scope(scope, *, default_name=None, reuse=False, **kwargs):
    return tf.variable_scope(tf.VariableScope(name=scope, reuse=reuse, **kwargs), default_name=default_name, auxiliary_name_scope=False)



def tf_var(name, create=None):
  fqn = tf_varname(name)
  result = tf_vars().get(fqn)
  if result is None:
    if create is None:
      raise ValueError("Variable %s does not exist" % name)
    elif callable(create):
      with absolute_variable_scope('', default_name=fqn):
        result = create()
      if isinstance(result, tf1.Variable):
        tf_vars()[fqn] = result
    else:
      result = create
  return result


def tf_local(name, initial_value, *args, **kws):
  return tf_var(name, lambda: LocalVariable(initial_value=initial_value() if callable(initial_value) else initial_value, *args, name=name, **kws))


# def local_variable(name, creator):
#   ujjj
#   initial_value = tf.convert_to_tensor(initial_value)


# #   with ops.name_scope(name, "matching_filenames", [pattern]) as name:
# #     return vs.variable(
# #         name=name, initial_value=io_ops.matching_files(pattern),
# #         trainable=False, validate_shape=False,
# #         collections=[ops.GraphKeys.LOCAL_VARIABLES])

# v = tf1.Variable([1, 2, 3, 4, 5, 6, 7, 8])
# indices = tf.constant([[4], [3], [1] ,[7]])
# updates = tf.constant([9, 10, 11, 12])
# op = v.scatter_nd_assign(indices, updates)


def is_string(x):
  return isinstance(x, str)


def is_number(x):
  return isinstance(x, (int, float))


def char(s=None, n=None):
  __n8 = n or 0
  if __n8 >= 0 and __n8 < len(s):
    return s[__n8]


def code(s=None, n=None):
  __x4 = char(s, n)
  if __x4:
    return ord(__x4)


def is_number_code(n):
  return n > 47 and n < 58


def number(x, base=None):
  if is_string(x):
    try:
      return int(x, base=10 if base is None else base)
    except ValueError:
      pass
    if base is None:
      try:
        return float(x)
      except ValueError:
        pass
  elif is_number(x):
    return x


def is_hex_prefix(s):
  __e = None
  if code(s, 0) == 45:
    __e = 1
  else:
    __e = 0
  __i = __e
  __id2 = code(s, __i) == 48
  __e1 = None
  if __id2:
    __i = __i + 1
    __n = code(s, __i)
    __e1 = __n == 120 or __n == 88
  else:
    __e1 = __id2
  return __e1


def maybe_number(x):
  if isinstance(x, bytes):
    x = x.decode('latin1')
  if is_string(x):
    if is_hex_prefix(x):
      return number(x, base=16)
    elif is_number_code(code(x, len(x)-1)):
      return number(x)
  elif is_number(x):
    return x


def read_value_1(x):
  v = maybe_number(x)
  if v is not None:
    return v
  return x


import ast


def read_value(x):
  try:
    return ast.literal_eval(x)
  except ValueError:
    return x
  except SyntaxError:
    return x


from urllib import parse

import re

import braces

 # can't use '?' for query_char because it means wildcard match on GCE storage path names
def parse_patterns(patterns, query_char='&'):
  if isinstance(patterns, str):
    pats = []
    for pattern in braces.braceexpand(patterns):
      pats.extend(re.split(r',\s*(?:(?=\w+://)|(?=/)|(?=[.]))', pattern))
    patterns = pats
  results = []
  for pattern in patterns:
    pat, query = pattern.split(query_char, 1) if query_char in pattern else (pattern, '')
    props = dict(parse.parse_qsl(query))
    props = {k: read_value(v) for k, v in props.items()}
    if 'weight' not in props:
      props['weight'] = 1.0
    if 'from' not in props:
      props['from'] = None
    if 'upto' not in props:
      props['upto'] = None
    results.append((pat, props))
  return results


def tf_glob(patterns, query_char='&'):
  results = []
  for pat, props in parse_patterns(patterns, query_char=query_char):
    # tf.io.gfile.glob seems to be deterministic; no need to sort? but
    # sort anyway.
    tf.logging.info('tf_glob {pat!r}, {props!r}'.format(pat=pat, props=props))
    files = list(sorted(tf.io.gfile.glob(pat)))
    files = files[props['from']:props['upto']]
    if len(files) <= 0:
      raise ValueError("Pattern {pat} failed to match any files".format(pat=pat))
    results.append((pat, props, files))
  return results



def tf_parse_file(filename):
  buffer_size = 8 * 1024 * 1024  # 8 MiB per file
  dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
  return dataset

def tf_parse_files(filenames, num_parallel_calls=64):
  if isinstance(filenames, (tuple, list)):
    filenames = tf.data.Dataset.from_tensor_slices(filenames)

  # Read the data from disk in parallel
  dataset = filenames.apply(
      tf.contrib.data.parallel_interleave(
          tf_parse_file, cycle_length=num_parallel_calls, sloppy=True))

  return dataset


def tf_sharded_datasets(pattern, num_hosts=1, current_host=0, parse_fn=tf_parse_files):
  datasets = []
  weights = []
  for pat, props, files in tf_glob(pattern):
    sharded_files = files[current_host::num_hosts]
    tf.logging.info('host {current_host} of {num_hosts}: Dataset pattern %s with props %s matched %s'.format(current_host=current_host, num_hosts=num_hosts), pat, props, sharded_files)
    #ds = tf.data.Dataset.from_tensor_slices(sharded_files)
    #ds = ds.shard(num_hosts, current_host)
    ds = parse_fn(sharded_files)
    #ds = ds.repeat()
    # ds = ds.shuffle(1000)
    # if parse_fn:
    #   ds = ds.map(parse_fn)
    #   # cache parsed results
    #   ds = ds.cache()
    datasets.append(ds)
    weights.append(props['weight'])
  dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
  return dataset

