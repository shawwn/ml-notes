import tensorflow as tf
import numpy as np
from glob import glob
import os
import re
from tensorflow.python import pywrap_tensorflow
import tqdm
import h5py
import shutil
import tempfile
import traceback
import time
import threading

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.distribute.cluster_resolver import TPUClusterResolver as BaseTPUClusterResolver
from tensorflow.python.training import server_lib
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
import importlib

from pprint import pprint as pp

def prn(x):
  pp(x);
  return x

def reload():
  os.system("git pull")
  module = importlib.import_module("tflex")
  importlib.reload(module)

class _DefaultState(threading.local):
  def __init__(self, **kws):
    super(_DefaultState, self).__init__()
    for k, v in kws.items():
      setattr(self, k, v)

  def save(self):
    return [(k, v) for k, v in self.__dict__.items()]

  def restore(self, state):
    for k, v in state:
      setattr(self, k, v)

local = _DefaultState()
lock = threading.RLock()

def with_defaults(thunk):
  with lock:
    state = local.save()
    session = tf.get_default_session() or get_default_session()
    graph = tf.get_default_graph() or get_default_graph()
  def f(*args, **kws):
    with lock:
      local.restore(state)
    lock.acquire()
    with session.as_default() if session else nullcontext():
      with graph.as_default() if graph else nullcontext():
        lock.release()
        result = thunk(*args, **kws)
        lock.acquire()
    lock.release()
    return result
  return f

def get_default(name, required=True):
  with lock:
    value = getattr(local, name) if hasattr(local, name) else None
  if required:
    assert value is not None
  return value

def set_default(name, value):
  with lock:
    setattr(local, name, value)

def ensure_default(name, value):
  with lock:
    current = get_default(name, required=False)
    if current is None:
      set_default(name, value)
    return value

def get_default_session(required=False):
  result = get_default('session', required=required)
  if result is None:
    result = tf.get_default_session()
  #assert result is not None
  return result

def get_default_graph(required=False):
  result = get_default('graph', required=required)
  if result is None:
    result = tf.get_default_graph()
  assert result is not None
  return result

class Future(object):
  def __init__(self, dependencies, thunk, *args, **kws):
    if isinstance(dependencies, Future):
      dependencies = [dependencies]
    self.dependencies = [defer(_) if callable(_) else _ for _ in dependencies]
    if thunk is None:
      thunk = lambda: None
    self.thunk = thunk
    self.args = args
    self.kws = kws
    self.result = None
    self.complete = False
    self.thread = None
    self.daemon = True
    self.error = None
  def run(self):
    try:
      self.result = self.thunk(*self.args, **self.kws)
    except Exception as e:
      traceback.print_exc()
      self.error = e
    self.complete = True
  def run_async(self):
    assert self.thread is None
    def thunk():
      [_.join() for _ in self.dependencies]
      self.run()
    self.thread = threading.Thread(target=with_defaults(thunk), daemon=self.daemon)
    self.thread.start()
  def join(self):
    if not self.complete:
      assert self.thread
      while not self.complete:
        time.sleep(1.0)
    return self.result

def defer(thunk, *args, **kws):
  dependencies = []
  if 'dependencies' in kws:
    dependencies = kws.pop('dependencies')
  future = Future(dependencies=dependencies, thunk=thunk, *args, **kws)
  future.run_async()
  return future

def parallelize(xs, thunk, *args, daemon=True):
  threads = []
  for x in xs:
    thread = threading.Thread(target=with_defaults(thunk), args=(x, *args), daemon=daemon)
    thread.start()
    threads.append(thread)
  return threads

def parallelize_verbose(label, xs, thunk, *args, daemon=True):
  xs = [x for x in xs]
  with tqdm.tqdm(total=len(xs)) as pbar:
    pbar.set_description(label)
    def run(*args, **kws):
      try:
        return thunk(*args, **kws)
      finally:
        pbar.update(1)
    return parallelize(xs, run, *args, daemon=daemon)

def parallelize_verbose(label, xs, thunk, *args, daemon=True, synchronous=False):
  xs = [x for x in xs]
  if synchronous:
    for i in tqdm.trange(len(xs), desc=label):
      x = xs[i]
      thunk(x, *args)
  else:
    with tqdm.tqdm(total=len(xs)) as pbar:
      pbar.set_description(label)
      threads = parallelize(xs, thunk, *args, daemon=daemon)
      while len(threads) > 0:
        for i in range(len(threads)):
          if not threads[i].is_alive():
            pbar.update(1)
            threads.remove(threads[i])
            break
        time.sleep(0.1)

# http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
import itertools
def group(n, iterable, fillvalue=None):
    "group(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def tuples(*args, **kws):
  return [x for x in group(*args, **kws)]

class Namespace(object):
  pass

if 'state' not in globals():
  state = Namespace()

if not hasattr(state, 'noisy'):
  state.noisy = 'NOISY' in os.environ

if not hasattr(state, 'debug'):
  state.debug = 'DEBUG' in os.environ

if not hasattr(state, 'noisy_backtrace'):
  state.noisy_backtrace = 'NOISY_BACKTRACE' in os.environ

if not hasattr(state, 'break_next_run'):
  state.break_next_run = False

def reroute(addr, host=None):
  if host is None or host is False:
    return addr
  if addr.startswith('grpc://'):
    return 'grpc://' + reroute(addr[len('grpc://'):], host=host)
  if not re.match('[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+[:]8470', addr):
    return addr
  if not addr.endswith(':8470'):
    return addr
  a, b, c, d = [int(x) for x in addr.split(':')[0].split('.')]
  if a == 10 and b in [48, 49]:
    assert (d == 2)
    port = b * 1000 + c
  elif a == 10 and b in range(2, 66) and c == 0:
    port = b * 1000 + d
  else:
    return addr
  return host + ':' + str(port)


class TPUClusterResolver(BaseTPUClusterResolver):
  def __init__(self, *args, host=None, node_count=None, node_offset=None, **kws):
    super(TPUClusterResolver, self).__init__(*args, **kws)
    if host is None:
      if 'TPU_HOST' in os.environ:
        host = os.environ['TPU_HOST']
    self._host = host
    if node_count is None:
      if 'TPU_NODE_COUNT' in os.environ:
        node_count = int(os.environ['TPU_NODE_COUNT'])
    self._node_count = node_count
    if node_offset is None:
      if 'TPU_NODE_OFFSET' in os.environ:
        node_offset = int(os.environ['TPU_NODE_OFFSET'])
    self._node_offset = node_offset

  def master(self, *args, **kws):
    ip = super(TPUClusterResolver, self).master(*args, **kws)
    return reroute(ip, host=self._host)

  def cluster_spec(self):
    spec = super(TPUClusterResolver, self).cluster_spec()
    r = dict()
    for k, v in spec.as_dict().items():
      r[k] = [reroute(ip, host=self._host) for ip in v]
    i = self._node_count or len(r['worker'])
    j = self._node_offset or 0
    r['worker'] = [r['worker'][0]] + r['worker'][(j+1):(j+1)+(i-1)]
    spec2 = server_lib.ClusterSpec(r)
    print(spec2.as_cluster_def())
    return spec2

if not hasattr(state, 'timeout_in_ms'):
  state.timeout_in_ms = 5 * 60 * 1000 # no TPU operation should last more than 5 minutes

def get_session_timeout_in_ms(timeout_in_ms=None):
  if timeout_in_ms is None:
    timeout_in_ms = state.timeout_in_ms
  return timeout_in_ms

def init_tpu_config(name=None, host=None, timeout_in_ms=None):
  if name is None:
    name = os.environ['TPU_NAME']
  timeout_in_ms = get_session_timeout_in_ms(timeout_in_ms)
  cluster_resolver = TPUClusterResolver(name, host=host)
  config = tf.ConfigProto(operation_timeout_in_ms=timeout_in_ms,
                          graph_options=tf.GraphOptions(
                            rewrite_options=rewriter_config_pb2.RewriterConfig(
                              disable_meta_optimizer=True)),
                          isolate_session_state=True)
  cluster_spec = cluster_resolver.cluster_spec()
  if cluster_spec:
    config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
  master = cluster_resolver.get_master()
  return master, config

class CountingSessionCreator(object):
  """A creator that counts the number of created sessions."""

  def __init__(self):
    self._create_session_calls = 0

  @property
  def number_of_sessions_created(self):
    return self._create_session_calls

  def create_session(self):
    self._create_session_calls += 1
    return self.creator()


class TPUSessionCreator(CountingSessionCreator):
  """A creator that counts the number of created sessions."""

  def __init__(self, *args, **kws):
    super(TPUSessionCreator, self).__init__()
    self._args = args
    self._kws = kws

  def creator(self):
    sess, resolver = init_tpu(*self._args, **self._kws)
    sess._tflex_resolver = resolver
    return sess

from tensorflow.python.training import monitored_session

class TPUSession(monitored_session._RecoverableSession):
  def __init__(self, name, host=None, timeout_in_ms=None, interactive=False, graph=None, initialize=True):
    super(TPUSession, self).__init__(TPUSessionCreator(name=name, host=host, timeout_in_ms=timeout_in_ms, interactive=interactive, graph=graph, initialize=initialize))

  def list_devices(self):
    return self._sess.list_devices()

def init_tpu(name, host=None, timeout_in_ms=None, interactive=False, graph=None, initialize=True):
  timeout_in_ms = get_session_timeout_in_ms(timeout_in_ms)
  cluster_resolver = TPUClusterResolver(name, host=host)
  graph = get_graph(graph)
  config = tf.ConfigProto(operation_timeout_in_ms=timeout_in_ms,
                          graph_options=tf.GraphOptions(
                            rewrite_options=rewriter_config_pb2.RewriterConfig(
                              disable_meta_optimizer=True)),
                          isolate_session_state=True)
  cluster_spec = cluster_resolver.cluster_spec()
  if cluster_spec:
    config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
  init_sess = (tf.InteractiveSession if interactive else tf.Session)(cluster_resolver.get_master(), config=config, graph=graph)
  if initialize:
    with graph.as_default():
      with absolute_name_scope('tflex'):
        tpu_init = [op for op in get_graph().get_operations() if op.name == 'ConfigureDistributedTPU']
        if len(tpu_init) <= 0:
          tpu_init = [tpu.initialize_system()]
    init_sess.run(tpu_init)
  return init_sess, cluster_resolver

def get_graph(graph=None):
  if graph is None:
    graph = get_default_graph()
  return graph

def get_session(session=None):
  if session is None:
    session = get_default_session()
  return session

from natsort import natsorted

def sort_devices(devices):
  return list(natsorted(devices, key=lambda x: x.name))

def get_devices(session=None):
  session = get_session(session)
  if hasattr(session, '_cached_devices'):
    devices = session._cached_devices
  else:
    devices = session._cached_devices = sort_devices(session.list_devices())
  return devices

def has_gpu(session=None):
  session = get_session(session)
  if hasattr(session, '_has_gpu'):
    result = session._has_gpu
  else:
    devices = get_devices(session=session)
    result = session._has_gpu = len([x for x in devices if ':GPU:' in x.name]) > 0
  return result

def has_tpu(session=None):
  session = get_session(session)
  if hasattr(session, '_has_tpu'):
    result = session._has_tpu
  else:
    devices = get_devices(session=session)
    result = session._has_tpu = len([x for x in devices if ':TPU:' in x.name]) > 0
  return result

def get_cores_from_devices(devices):
  cores = [x for x in devices if ':TPU:' in x.name]
  if len(cores) <= 0:
    cores = [x for x in devices if ':GPU:' in x.name]
  if len(cores) <= 0:
    cores = [x for x in devices if ':CPU:' in x.name]
  #return sort_devices(cores) # TODO: assert sorted order
  return cores

def get_cores(session=None, devices=None):
  if devices is None:
    devices = get_devices(session=session)
  return get_cores_from_devices(devices)

def get_cpus(session=None, devices=None):
  if devices is None:
    devices = get_devices(session=session)
  cpus = [x for x in devices if ':CPU:' in x.name]
  return cpus

def get_tpu_resolver(tpu_name='auto'):
  # Get the TPU's location
  if tpu_name != 'auto':
    return TPUClusterResolver(tpu_name)
  elif 'COLAB_TPU_ADDR' in os.environ:
    return TPUClusterResolver()
  elif 'TPU_NAME' in os.environ:
    return TPUClusterResolver(os.environ['TPU_NAME'])

def pretty(x, ellipsize=120):
  r = str(x)
  if len(r) > ellipsize:
    return r[0:ellipsize - 3] + '...'
  return r

def print_backtrace():
  try:
    raise Exception("Printing traceback...")
  except:
    import traceback
    traceback.print_exc()

class Session(tf.Session):
  def __init__(self, target='auto', graph=None, config=None, id=None, timeout_in_ms=None):
    if config is None:
      timeout_in_ms = get_session_timeout_in_ms(timeout_in_ms)
      config = tf.ConfigProto(operation_timeout_in_ms=timeout_in_ms,
                              graph_options=tf.GraphOptions(
                                rewrite_options=rewriter_config_pb2.RewriterConfig(
                                  disable_meta_optimizer=True)),
                              isolate_session_state=True)
    config.isolate_session_state = True
    resolver = get_tpu_resolver(target)
    if resolver is not None:
      target = resolver.get_master()
      cluster_spec = resolver.cluster_spec()
      if cluster_spec:
        config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    else:
      if target == 'auto':
        target = None
    super().__init__(target, graph=graph, config=config)
    self.id = id
    self._tflex_resolver = resolver
    self._tflex_target = target
    self._tflex_config = config
    ensure_default('session', self)
    ensure_default('devices', self.list_devices())
    ensure_default('graph', self.graph)

  @property
  def _spec(self):
    return '#%d' % self.id if self.id is not None else ''

  def run(self, *args, **kws):
    if state.break_next_run:
      import pdb; pdb.set_trace()
    if state.debug:
      check_commands()
    if state.noisy:
      print(self._spec, 'Session.run', *[pretty(x) for x in args], *[pretty(k)+'='+pretty(v) for k, v in kws.items()])
      if state.noisy_backtrace:
        print_backtrace()
    with with_elapsed(super(Session, self).run, *args, **kws) as (elapsed, result):
      if state.noisy:
        print(self._spec, 'Session.run (finished in %.2fs)' % elapsed, pretty(result), *[pretty(x) for x in args], *[pretty(k)+'='+pretty(v) for k, v in kws.items()])
        if state.noisy_backtrace:
          print_backtrace()
      return result


def split_by_params(vs, n=None, f=None):
  if n is None:
    #n = 2e6
   n = 1
  if f is None:
    f = lambda x: np.prod(x.shape.as_list())
  i = 0
  xs = []
  for variable in vs:
    xs.append(variable)
    count = f(variable)
    i += count
    if i >= n:
      yield xs
      xs = []
      i = 0
  yield xs

def latest_checkpoint(checkpoint_dir, latest_filename=None):
  paths = [x for x in glob(os.path.join(checkpoint_dir, 'model-*.*')) if not x.endswith(".tmp")]
  ctrs = np.array([[int(y) for y in re.findall(r'model-([0-9]+)(?:-[0-9]+)?[.](?:npy|hdf5)', x)] for x in paths]).flatten()
  if len(ctrs) <= 0:
    ckpt = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=latest_filename)
    return ckpt
  ctr = ctrs.max()
  return os.path.join(checkpoint_dir, 'model-{}').format(ctr)

def truncate_value(variable, value, reshape=True):
  if not reshape:
    return value
  shape = variable.shape.as_list()
  params = np.prod(shape)
  params2 = np.prod(value.shape)
  if params == params2:
    return value
  print('Truncating {} from shape {} to shape {}'.format(variable.name, value.shape, shape))
  value = np.array(value)
  value = value.reshape([-1])
  value = value[0:params]
  value = value.reshape(shape)
  return value

from tensorflow.core.protobuf import config_pb2

def initialize_tpu(session=None, timeout_in_ms=None):
  session = session or get_default_session()
  with session.as_default():
    op = tpu.initialize_system()
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(op, options=options)

def load(variable, value, session=None, timeout_in_ms=None):
  session = session or get_default_session()
  ops = variable.initializer
  vals = dict([(variable.initializer.inputs[1], value)])
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(ops, vals, options=options)

def eval(variable, session=None, timeout_in_ms=None):
  session = session or get_default_session()
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(variable, options=options)

def grab_values(variables, reader, reshape=False):
  for variable in variables:
    name = variable_name(variable).split(':')[0]
    value = reader.get_tensor(name)
    value = truncate_value(variable, value, reshape=reshape)
    yield variable, value

import collections

def is_list(x):
  return isinstance(x, collections.Sequence)

def element_count(x):
  if is_list(x):
    return sum([element_count(v) for v in x])
  if hasattr(x, 'shape'):
    x = x.shape
  if hasattr(x, 'as_list'):
    x = x.as_list()
  return int(np.prod(x))

from contextlib import contextmanager

@contextmanager
def with_elapsed(thunk, *args, **kws):
  start = time.time()
  result = thunk(*args, **kws)
  elapsed = time.time() - start
  yield elapsed, result

@contextmanager
def on_elapsed(callback):
  start = time.time()
  result = yield
  if callback is not None:
    elapsed = time.time() - start
    callback(elapsed)
  return result

def assign_values(variables, values, session=None, timeout_in_ms=600000):
  session = session or get_default_session()
  variables = [x for x in variables]
  values = [x for x in values]
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value.value() if isinstance(value, tf.Variable) else value) for x, value in zip(variables, values)]) # TODO: bfloat16 support
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  tf.logging.info('Loading %s elements to TPU', num(element_count(variables)))
  with with_elapsed(session.run, ops, vals, options=options) as (elapsed, result):
    tf.logging.info('Loaded %s elements to TPU in %.2fs', num(element_count(variables)), elapsed)

def load_snapshot(ckpt, session=None, var_list=None, reshape=False):
  session = session or get_default_session()
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  vs = var_list or tf.trainable_variables()
  for variables in tqdm.tqdm(list(split_by_params(vs))):
    values = [value for variable, value in grab_values(variables, reader, reshape=reshape)]
    assign_values(variables, values, session=session)

def get_variable(name, var_list=None):
  name, num = name.split(':') if ':' in name else (name, '0')
  num = int(num)
  name = os.path.join(tf.get_variable_scope().name, name)
  vs = var_list or tf.trainable_variables()
  for x in vs:
      if x.name.startswith(name + ':%d' % num):
          return x

def load_weights(ckpt, session=None, var_list=None, reshape=False):
  session = session or get_default_session()
  vs = var_list or tf.trainable_variables()
  files = list(sorted(glob(ckpt + '-*.npy')))
  for out in tqdm.tqdm(files):
    for name, value in np.load(out, allow_pickle=True):
      variable = get_variable(name)
      if variable is None:
        print('Warning: variable %s not loaded' % name)
      else:
        value = truncate_value(variable, value, reshape=reshape)
        variable.load(value, session)

def get_values(variables, f, reshape=False, ignore_missing=False):
  for x in variables:
    k = variable_name(x)
    if ignore_missing:
      try:
        value = f[k]
      except KeyError:
        print('Ignoring missing variable {}'.format(k))
        continue
    else:
      value = f[k]
    yield truncate_value(x, value, reshape=reshape)

def load_variables(ckpt, session=None, var_list=None, reshape=False, ignore_missing=False):
  session = session or get_default_session()
  vs = var_list or tf.trainable_variables()
  with h5py.File(ckpt, "r") as f:
    for variables in tqdm.tqdm(list(split_by_params(vs))):
      values = get_values(variables, f, reshape=reshape, ignore_missing=ignore_missing)
      assign_values(variables, values, session=session)

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

state.cache_ops = {}

def cast_variables(variables, graph=None, cache_ops=None):
  if graph is None:
    graph = get_default_graph()
  if cache_ops is None:
    cache_ops = state.cache_ops
  if graph not in cache_ops:
    cache_ops[graph] = {}
  cache = cache_ops[graph]
  ops = []
  for variable in variables:
    if variable in cache:
      op = cache[variable]
    elif variable.dtype == dtypes.bfloat16_ref or variable.dtype == tf.bfloat16:
      op = tf.cast(variable, tf.float32)
    else:
      op = variable
    cache[variable] = op
    ops.append(op)
  return ops

import re

def variable_name(variable):
  if re.match(r'core[0-9]+/', variable.name):
    return variable.name.split('/', 1)[-1]
  return variable.name

def save_variables(ckpt, session=None, var_list=None):
    session = session or get_default_session()
    vs = var_list or tf.trainable_variables()
    maketree(os.path.dirname(ckpt))
    fname = ckpt+'.tmp'
    with h5py.File(fname, "w") as f:
      for variables in tqdm.tqdm(list(split_by_params(vs))):
        ops = cast_variables(variables)
        values = session.run(ops)
        for value, variable in zip(values, variables):
          name = variable_name(variable)
          shape = variable.shape.as_list()
          dtype = variable.dtype
          dset = f.create_dataset(name, shape, dtype=np.float32)
          dset[:] = value
    print('Writing snapshot %s' % ckpt)
    os.rename(ckpt+'.tmp', ckpt)

def fetch_variables(session=None, var_list=None):
    session = session or get_default_session()
    vs = var_list or tf.trainable_variables()
    for variables in tqdm.tqdm(list(split_by_params(vs))):
      values = session.run(variables)
      yield variables, values

def partition_variables(session=None, var_list=None):
    session = session or get_default_session()
    vs = var_list or tf.trainable_variables()
    for variables in tqdm.tqdm(list(split_by_params(vs))):
      yield variables

class Saver(object):
  def __init__(
    self,
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None):
    self.var_list = var_list
    self.reshape = reshape
    self.sharded = sharded
    self.max_to_keep = max_to_keep
    self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self.name = name
    self.restore_sequentially = restore_sequentially
    self.saver_def = saver_def
    self.builder = builder
    self.defer_build = defer_build
    self.allow_empty = allow_empty
    self.write_version = write_version
    self.pad_step_number = pad_step_number
    self.save_relative_paths = save_relative_paths
    self.filename = filename
    self.checkpoints = []

  def restore(self, sess, save_path, ignore_missing=False):
    if save_path.endswith('.ckpt'):
      load_snapshot(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif save_path.endswith('.hdf5'):
      load_variables(save_path, session=sess, var_list=self.var_list, reshape=self.reshape, ignore_missing=ignore_missing)
    elif os.path.exists(save_path + '.npy') or os.path.exists(save_path + '-0.npy'):
      load_weights(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif os.path.exists(save_path + '.hdf5'):
      load_variables(save_path + '.hdf5', session=sess, var_list=self.var_list, reshape=self.reshape, ignore_missing=ignore_missing)
    else:
      raise Exception("Can't load checkpoint %s" % save_path)

  def save(self,
        sess,
        save_path,
        global_step=None,
        latest_filename=None,
        meta_graph_suffix="meta",
        write_meta_graph=True,
        write_state=True,
        strip_default_attrs=False,
        save_debug_info=False):
    if global_step is not None:
      name = '%s-%d.hdf5' % (save_path, global_step)
    else:
      name = '%s.hdf5' % save_path
    save_variables(name, session=sess, var_list=self.var_list)
    self.checkpoints.append(name)
    if self.max_to_keep > 0:
      while len(self.checkpoints) > self.max_to_keep:
        fname = self.checkpoints[0]
        if fname != name:
          print('Truncating %s' % fname)
          try:
            with open(fname, "wb") as f:
              pass
          except:
            print('Failed to truncate %s' % fname)
        self.checkpoints = self.checkpoints[1:]

  def fetch(self, sess, var_list=None):
    if var_list == None:
      var_list = self.var_list
    for variables, values in fetch_variables(session=sess, var_list=var_list):
      yield variables, values

  def variables(self, sess, var_list=None):
    if var_list == None:
      var_list = self.var_list
    for variables in partition_variables(session=sess, var_list=var_list):
      yield variables

  def assign(self, sess, variables, values):
    return assign_values(variables, values, session=sess)

class Commands(object):
  def __init__(self, path='commands'):
    self.path = path
    self.commands = []
    self.args = []
    self.keys = {}
    self.frozen = False

  def has(self, name, **keys):
    if 'action' in keys:
      action = keys.pop('action')
      for name1, action1 in self.commands:
        if name == name1 and action1 == action:
          return True
    else:
      for name1, action1 in self.commands:
        if name == name1:
          return True
    return False

  def add(self, name, action=None):
    if not self.has(name=name, action=action):
      self.commands.append((name, action))
      full = self.full_path(name)
      maketree(full)

  def full_path(self, name):
    return os.path.join(self.path, name)

  def check(self, *args, **keys):
    if not self.frozen:
      heartbeat()
    ops = []
    seen = set()
    for name, action in self.commands:
      full = self.full_path(name)
      if not os.path.isdir(full):
        if name not in seen:
          seen.add(name)
          ops.append(name)
    for op in ops:
      self.run(op, *args, **keys)
    return ops

  def run(self, op):
    ran = False
    for name, action in self.commands:
      if name == op:
        print('Running command', name, action)
        if not ran:
          full = self.full_path(op)
          maketree(full)
          ran = True
        if action:
          action()
    if not ran:
      raise Exception('Commands.execute failed: no such command: {}'.format(op))
  
  def run_with_args(self, op, *args, **keys):
    with CommandArgs(*args, **keys):
      return self.run(op)

commander = None

def commands(**keys):
  global commander
  if commander is None:
    commander = Commands()
  cmds = keys.pop('commands') if 'commands' in keys else None
  if cmds is not None:
    for cmd in cmds:
      action = None
      if isinstance(cmd, str):
        name = cmd
      elif len(cmd) >= 2:
        name, action = cmd
      elif len(cmd) >= 1:
        name = cmd[0]
      else:
        continue
      commander.add(name=name, action=action)
  return commander

class CommandArgs(object):
  def __init__(self, *args, **keys):
    self.args = list(args)
    self.keys = keys.copy()
    self.cmdr = commands()

  def __enter__(self):
    self.args_prev = self.cmdr.args
    self.keys_prev = self.cmdr.keys
    self.cmdr.args = self.args
    self.cmdr.keys = self.keys

  def __exit__(self, *excinfo):
    self.cmdr.args = self.args_prev
    self.cmdr.keys = self.keys_prev

def check_commands():
  try:
    cmdr = commands()
    return cmdr.check()
  except:
    traceback.print_exc()

def check_commands_with_args(*args, **keys):
  try:
    cmdr = commands()
    with CommandArgs(*args, **keys):
      return cmdr.check()
  except:
    traceback.print_exc()

def add_command(name, action=None, **keys):
  cmdr = commands()
  return cmdr.add(name=name, action=action)

def register_command(*args, **keys):
  fn = args[0]
  if isinstance(fn, str):
    add_command(fn)
  else:
    name = fn.__qualname__
    name = name.replace('.<locals>.', '_command_')
    if name.endswith('_command_save'):
      name = 'save'
    name = name.replace('___', '/')
    action = fn
    print(name, action)
    add_command(name, action)
  return fn

def has_command(name):
  cmdr = commands()
  return cmdr.has(name)

def run_command(command_name):
  cmdr = commands()
  return cmdr.run(command_name)

def run_command_with_args(command_name, *args, **keys):
  cmdr = commands()
  return cmdr.run_with_args(command_name, *args, **keys)

def command_arg(x, unset=None):
  cmdr = commands()
  if isinstance(x, int):
    try:
      return cmdr.args[x]
    except:
      return unset
  else:
    if x in cmdr.keys:
      return cmdr.keys[x]
    return unset

def command_args():
  cmdr = commands()
  return cmdr.args, cmdr.keys

@register_command
def attach_debugger():
  import pdb
  pdb.set_trace()

from pprint import pprint

@register_command
def print_status():
  args, props = command_args()
  for k, v in enumerate(args):
    pprint(v)
  for k, v in props.items():
    pprint({k: v})


#
# return current UTC timestamp.
#
def utc():
    from datetime import datetime
    d = datetime.utcnow()
    import calendar
    return calendar.timegm(d.utctimetuple())

def heartbeat():
  pongfile=os.environ['PONG'] if 'PONG' in os.environ else 'pong.txt'
  with open(pongfile, "a+") as f:
    nonce = os.urandom(8).hex()
    now=utc()
    out="pid{}_time{}_nonce{}\n".format(os.getpid(), now, nonce)
    #print("PONG! Writing {} to {}".format(out, pongfile))
    f.write(out)
    f.flush()

import time

@register_command
def freeze_forever():
  cmdr = commands()
  if cmdr.frozen:
    print("Already frozen.")
    return
  prev = cmdr.frozen
  cmdr.frozen = True
  print('Simulating a freeze; going into an infinite loop:')
  prev=time.time()
  try:
    while not should_quit():
      elapsed=time.time() - prev
      print('Frozen for {}s'.format(elapsed))
      time.sleep(1)
      check_commands()
  finally:
    cmdr.frozen = prev

_quit = False

import posix

@register_command
def quit():
  global _quit
  if _quit:
    print("Failed to quit; terminating via posix._exit(1)")
    posix._exit(1)
  else:
    print("Signaling to main program that we should quit...")
    _quit = True

def should_quit():
  return _quit

@register_command
def save_and_quit():
  global _quit
  if has_command('save'):
    print("Saving...")
    run_command('save')
  quit()

@register_command
def throw_exception():
  raise Exception("This exception should be caught and logged by the tflex command system")


import tensorflow as tf
from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def set_override_device(value, session=None):
  session = get_session(session)
  session._override_device = value
  return value

def has_override_device(session=None):
  session = get_session(session)
  return hasattr(session, '_override_device')

def get_override_device(session=None):
  session = get_session(session)
  if hasattr(session, '_override_device'):
    return session._override_device

def set_override_cores(value, session=None):
  session = get_session(session)
  session._override_cores = value
  return value

def has_override_cores(session=None):
  session = get_session(session)
  return hasattr(session, '_override_cores')

def get_override_cores(session=None):
  session = get_session(session)
  if hasattr(session, '_override_cores'):
    return session._override_cores

# returns either 'worker' or 'tpu_worker'
# (this function is to work around some tensorflow oddities on
# different versions)
def get_job_name(session=None):
  result = [x for x in get_cpus(session=session)[0].name.lstrip('/').split('/') if x.startswith('job:')][0].split(':')[1]
  return result

def device_for_tpu_core(task=0, core=0, job_name=None, session=None):
  if job_name is None:
    job_name = get_job_name(session=session)
  return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job_name, task, core)

def device_name(name, session=None):
  session = get_session(session)
  if name is None:
    return name
  if name.startswith('/gpu:'):
    i = int(name.split(':', 1)[-1])
    return get_cores(session=session)[i].name
  if name.startswith('/tpu:'):
    i = int(name.split(':', 1)[-1])
    return device_for_tpu_core(core=i)
  if name.startswith('/cpu:'):
    i = int(name.split(':', 1)[-1])
    return get_cpus(session=session)[i].name
  return name

def dev(name, session=None):
  return tf.device(device_name(name, session))

def cpu(index, session=None):
  return dev("/cpu:%d" % index, session=session)

def gpu(index, session=None):
  return dev("/gpu:%d" % index, session=session)

def device(name='', session=None):
  session = get_session(session)
  if has_override_device(session=session):
    return nullcontext()
  if has_override_cores(session=session):
    if name is None:
      return tf.device(name)
    if name.startswith('/gpu:'):
      i = int(name.split(':', 1)[-1])
      return tf.device(get_cores(session=session)[i].name)
    if name.startswith('/tpu:'):
      i = int(name.split(':', 1)[-1])
      return tf.device(device_for_tpu_core(core=i))
    if name.startswith('/cpu:'):
      i = int(name.split(':', 1)[-1])
      return tf.device(get_cpus(session=session)[i].name)
    return nullcontext()
  if name is None:
    return tf.device(None)
  if 'gpu' in name:
    if has_gpu(session=session):
      return tf.device(name)
  if 'cpu' in name:
    return tf.device(name)
  return nullcontext()

def get_global_step(graph=None):
  """Get the global step tensor.

  The global step tensor must be an integer variable. We first try to find it
  in the collection `GLOBAL_STEP`, or by name `global_step:0`.

  Args:
    graph: The graph to find the global step in. If missing, use default graph.

  Returns:
    The global step variable, or `None` if none was found.

  Raises:
    TypeError: If the global step tensor has a non-integer type, or if it is not
      a `Variable`.
  """
  graph = graph or ops.get_default_graph()
  global_step_tensor = None
  global_step_tensors = graph.get_collection(ops.GraphKeys.GLOBAL_STEP)
  if len(global_step_tensors) == 1:
    global_step_tensor = global_step_tensors[0]
  elif not global_step_tensors:
    try:
      global_step_tensor = graph.get_tensor_by_name('global_step:0')
    except KeyError:
      return None
  else:
    logging.error('Multiple tensors in global_step collection.')
    return None

  #assert_global_step(global_step_tensor)
  return global_step_tensor

def create_global_step(graph=None):
  """Create global step tensor in graph.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    Global step tensor.

  Raises:
    ValueError: if global step tensor is already defined.
  """
  graph = graph or ops.get_default_graph()
  if get_global_step(graph) is not None:
    raise ValueError('"global_step" already exists.')
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    return variable_scope.get_variable(
        ops.GraphKeys.GLOBAL_STEP,
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        use_resource=True,
        aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])

def get_or_create_global_step(graph=None):
  """Returns and create (if necessary) the global step tensor.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    The global step tensor.
  """
  graph = graph or ops.get_default_graph()
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    global_step_tensor = create_global_step(graph)
  return global_step_tensor

def create_var_with_large_initial_value(initial_value: np.ndarray, *args, **kwargs):
    assert isinstance(initial_value, np.ndarray)
    var = tf.Variable(initial_value, *args, **kwargs)
    return var.initialized_value()

def create_var_with_large_initial_value2(initial_value: np.ndarray, *args, **kwargs):
    """Create tf.Variable with large initial value without bloating the tf graph."""
    assert isinstance(initial_value, np.ndarray)
    zeros = tf.zeros(initial_value.shape, initial_value.dtype)
    var = tf.Variable(zeros, *args, **kwargs)
    return var, tf.assign(var, initial_value)

def create_var_with_large_initial_value3(initial_value: np.ndarray, *args, **kwargs):
    """Create tf.Variable with large initial value without bloating the tf graph."""
    assert isinstance(initial_value, np.ndarray)
    var, finalize = create_var_with_large_initial_value2(initial_value, *args, **kwargs)
    with tf.control_dependencies([finalize]):
      return tf.identity(var)

def absolute_name_scope(scope):
    return tf.name_scope(scope + "/")

def absolute_variable_scope(scope, **kwargs):
    return tf.variable_scope(tf.VariableScope(name=scope, **kwargs), auxiliary_name_scope=False)

def get_session_run_options(kws):
  if 'options' not in kws and 'timeout' not in kws:
    return None
  if 'timeout' not in kws:
    return kws.pop('options')
  options = config_pb2.RunOptions()
  if 'options' in kws:
    options.MergeFrom(kws.pop('options'))
  timeout = kws.pop('timeout')
  options.timeout_in_ms = int(timeout * 1000.0)
  return options

def run(session, *args, **kws):
  options = get_session_run_options(kws)
  session = get_session(session)
  return session.run(*args, options=options, **kws)

def num_cores(session=None):
  return len(get_cores(session=session))

def all_equal(l):
  prev = None
  for i, x in enumerate(l):
    if i > 0 and prev != x:
      return False
    prev = x
  return True

def flatten(l):
  r = []
  for i in range(len(l)):
    x = l[i]
    for j in range(len(x)):
      y = x[j]
      r.append(y)
  return r

def tpu_parallel(f, inputs=[], session=None):
  session = get_session(session)
  assert has_tpu(session=session)
  num_shards = num_cores(session=session)
  if callable(inputs):
    inputs = prn([inputs(i) for i in range(num_shards)])
    assert all_equal([len(x) for x in inputs])
    inputs = [flatten(inputs)]
    #inputs = [np.hstack(inputs)]
  (results,) = tpu.shard(f, inputs=inputs, num_shards=num_shards, outputs_from_all_shards=True)
  return results

# ----------- misc --------------------

from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook

class FakeHook(session_run_hook.SessionRunHook):

  def __init__(self):
    self.should_stop = False
    self.request = None
    self.call_counter = collections.Counter()
    self.last_run_context = None
    self.last_run_values = None

  def begin(self):
    self.call_counter['begin'] += 1

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    self.call_counter['after_create_session'] += 1

  def before_run(self, run_context):
    self.call_counter['before_run'] += 1
    self.last_run_context = run_context
    return self.request

  def after_run(self, run_context, run_values):
    self.call_counter['after_run'] += 1
    self.last_run_values = run_values
    if self.should_stop:
      run_context.request_stop()

  def end(self, session):
    self.call_counter['end'] += 1


def get_variable(name, var_list=None):
  name = os.path.join(tf.get_variable_scope().name, name)
  if var_list is None:
    var_list = tf.global_variables()
  for x in var_list:
    if x.name.startswith(name + ':'):
      return x

def create_variable(name, value, **kws):
  use_resource = kws.pop('use_resource') if 'use_resource' in kws else True
  return tf.Variable(value, name=name, use_resource=use_resource, **kws)

def get_or_create_variable(name, value, **kws):
  v = get_variable(name)
  if v is None:
    v = create_variable(name, value, **kws)
  return v

def create_untrainable_variable(name, value, **kws):
  return create_variable(name, value, trainable=False, **kws)

def get_or_create_untrainable_variable(name, value, **kws):
  return get_or_create_variable(name, value, trainable=False, **kws)

# Sample 1024(+1) tokens from the stitched together text
def sample_text(x, amount):
  s = tf.size(x)
  r = tf.random.uniform([], maxval=s-(amount+1), dtype=tf.dtypes.int32)
  r1 = tf.range(r, r+amount)
  r2 = tf.range(r+1, (r+1)+amount)
  r1 = tf.reshape(r1, [amount]) # Somehow, this makes the compiler happy
  r2 = tf.reshape(r2, [amount]) # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
  vals1 = tf.gather(x, r1)
  vals2 = tf.gather(x, r2)
  features, labels = vals1, vals2
  return features, labels

def read_bucket(path, mode='rb'):
  if os.path.isfile(path):
    with open(path, mode) as f:
      return f.read()
  else:
    import tensorflow as tf
    with tf.io.gfile.GFile(path, mode=mode) as f:
      return f.read()

import tempfile
from contextlib import contextmanager

@contextmanager
def bucket_file(path):
  if os.path.isfile(path):
    with open(path, "rb") as f:
      data = f.read()
    yield path, data
  else:
    data = read_bucket(path)
    with tempfile.NamedTemporaryFile() as tmp:
      tmp.write(data)
      tmp.seek(0)
      yield tmp.name, data

def bucket_path(path, *parts):
  if len(parts) <= 0:
    return path
  if path.startswith('gs://'):
    sep = '/'
  else:
    sep = os.sep
  if not path.endswith(sep):
    path = path + sep
  path = path + parts[0]
  return bucket_path(path, *parts[1:])

def make_tokens(tokens=None):
  if isinstance(tokens, str):
    path = tokens
    data = read_bucket(path)
    tokens = np.frombuffer(data, dtype=np.uint16)
    tf.logging.info("Loaded %d tokens from %s", len(tokens), path)
  if tokens is None:
    tokens = np.array([_ for _ in range(2048)], dtype=np.uint16)
  if isinstance(tokens, tf.Tensor):
    return tokens
  with cpu(0):
    #tokens = get_or_create_untrainable_variable("tokens", lambda: tokens, dtype=tf.int32)
    tokens = create_untrainable_variable("tokens", lambda: tokens, dtype=tf.int32)
  return tf.identity(tokens.initialized_value(), name="read_tokens")

def get_tpu_step(model_fn, params, tokens=None):
  gs = get_or_create_global_step()
  def input_fn(host_index):
    toks = make_tokens(tokens=tokens)
    features, labels = sample_text(toks, 1024);
    return features, labels
  def tpu_step(x):
    features, labels = x[0], x[1]
    features = tf.reshape(tf.cast(features, tf.int32), [1, -1])
    labels = tf.reshape(tf.cast(labels, tf.int32), [1, -1])
    with tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE):
      estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN, params);
    with tf.control_dependencies([estimator_spec.train_op]):
      return tf.identity(estimator_spec.loss, name="tpu_loss_op")
  def predict_tpu_step(x):
    pparams = dict(params)
    features, labels = x[0], None
    pparams['text_len'] = tf.size(features)
    pparams['length'] = 32
    features = tf.reshape(tf.cast(features, tf.int32), [1, -1])
    with tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE):
      estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.PREDICT, pparams);
    #return tf.identity(estimator_spec.predictions['tokens'], name="tpu_predict_op")
    #return estimator_spec.predictions['tokens']
    return tf.constant([42], dtype=tf.int32)
  @tpu_function.on_device_training_loop
  def tpu_loop():
    return wrap_computation_in_while_loop(tpu_step, params['iterations'])
  train_op = tpu_parallel(tpu_step, inputs=input_fn)
  #train_op = tpu_parallel(tpu_loop, inputs=input_fn)
  predict_op = tpu_parallel(predict_tpu_step, inputs=input_fn)
  #return tpu_step, input_fn
  return train_op, predict_op

def restore(ckpt=None, session=None, var_list=None):
  session = get_session(session)
  if ckpt is None:
    ckpt = 'gs://danbooru-euw4a/models/117M/model.ckpt'
  if var_list is None:
    var_list = tf.trainable_variables()
  saver = tf.train.Saver(var_list=prn(var_list));
  saver.restore(session, ckpt)
  return var_list

def timeit(f, verbose=True):
  start = time.time();
  result = f();
  dt = (time.time() - start);
  if verbose:
    print('%.2fsec' % dt);
  return result, dt


def wrap_computation_in_while_loop_vanilla(op_fn, n, parallel_iterations=1):
  """Wraps the ops generated by `op_fn` in tf.while_loop."""

  def computation(i):
    ops = op_fn()
    if not isinstance(ops, list):
      ops = [ops]
    with tf.control_dependencies(ops):
      return i + 1

  return tf.while_loop(
      lambda i: tf.less(i, n),
      computation, [tf.constant(0)],
      parallel_iterations=parallel_iterations)



def wrap_computation_in_while_loop(op_fn, n, parallel_iterations=1, as_callable=False):
  """Wraps the ops generated by `op_fn` in tf.while_loop."""

  def computation(i):
    ops = op_fn()
    if not isinstance(ops, list):
      ops = [ops]
    with tf.control_dependencies(ops):
      return i + 1

  def step():
    if as_callable:
      return op_fn()
    else:
      return op_fn

  def cond(*args):
      return True

  def body(i, output):
    loss = step()
    with tf.control_dependencies([loss]):
      pp(loss)
      pp(output)
      #res = tf.stack([output[0], loss], axis=0)
      res = tf.concat([output, [loss]], axis=0)
      pp(res)
      #return [
      #    #tf.concat([output, loss], axis=1)
      #    output
      #]
      return [
          i + 1,
          res
      ]

  return tf.while_loop(
      #lambda i: tf.less(i, n),
      cond,
      body,
      maximum_iterations=n,
      loop_vars=[
          #context_output['presents'],
          #context[:, -1],
          #context,
        #tf.constant([], dtype=tf.float32)
       tf.constant(0),
       tf.constant([[0.0] * 8], dtype=tf.float32)
      ],
      shape_invariants=[
          #tf.TensorShape(gpt2.past_shape(params=params, batch_size=batch_size)),
          tf.TensorShape(None),
          tf.TensorShape([None, 8]),
          #tf.TensorShape([None, None]),
      ],
      back_prop=False,
      parallel_iterations=parallel_iterations)


def get_pending_host_calls(graph=None):
  graph = graph or get_default_graph()
  if not hasattr(graph, 'pending_host_calls'):
    graph.pending_host_calls = []
  return graph.pending_host_calls

def get_pending_host_call(graph=None):
  graph = graph or get_default_graph()
  if graph is None:
    return None
  pending = get_pending_host_calls(graph=graph)
  try:
    return pending.pop()
  except IndexError:
    pass

def add_pending_host_call(thunk, graph=None):
  graph = graph or get_default_graph()
  pending = get_pending_host_calls(graph=graph)
  pending.append(thunk)

def get_graph_lock(graph):
  if not hasattr(graph, 'tflex_lock'):
    graph.tflex_lock = threading.RLock()
  return graph.tflex_lock

@contextmanager
def with_graph(graph=None, allow_mutations=True, make_graph_default=True, use_locking=True):
  graph = graph or get_default_graph()
  if graph is None:
    result = yield graph
    return result
  graph.switch_to_thread_local()
  lock = get_graph_lock(graph) if use_locking else nullcontext()
  with lock:
    finalized = graph.finalized
    if finalized and allow_mutations:
      graph._unsafe_unfinalize()
    try:
      if make_graph_default:
        with graph.as_default():
          lock.release()
          try:
            result = yield graph
          finally:
            lock.acquire()
      else:
        lock.release()
        try:
          result = yield graph
        finally:
          lock.acquire()
      return result
    finally:
      if finalized:
        graph.finalize()


def set_graph_local(self, name, value):
  self = self.graph if hasattr(self, 'graph') else self
  assert self._thread_local is not None
  setattr(self._thread_local, name, value)

import functools

def flush(session=None):
  session = session or get_default_session()
  results = []
  if session is None:
    tf.logging.warn('tflex.flush called, but no default session was set')
    return results

  host_callbacks = []
  while True:
    host_callback = get_pending_host_call(graph=session.graph)
    if host_callback is None:
      break
    host_callbacks.append(host_callback)
  def host_fn(host_callback):
    with session.graph.as_default(), session.as_default():
      value = host_callback(session=session)
      results.append(value)
  tf.logging.info('Running %d host callbacks %s for session %s...', len(host_callbacks), host_callbacks, session)
  session.graph.switch_to_thread_local()
  for i, thread in enumerate(parallelize(host_callbacks, host_fn)):
    thread.join()
    tf.logging.info('Host callback %d of %d finished.', i, len(host_callbacks))
  # with with_graph(session.graph) as g:
  #   with with_elapsed(host_callback, session=session) as (elapsed, value):
  #     tf.logging.info('Finished host callback in %.2f: %s', elapsed, pretty(value))
  #     results.append(value)
  return results


class Dictator(dict):
  def __getattr__(self, k):
    try:
      return self[k]
    except KeyError:
      raise AttributeError(k)
  def __setattr__(self, k, v):
    self[k] = v
  def __delattr__(self, k):
    del self[k]

class DatasetFunctionIterator:
  def __init__(self, parent):
    self._parent = parent
    self.initializer = parent._init_fn()
    add_pending_host_call(parent._upload_fn)

  def get_next(self):
    return self._parent._sample_fn()

class DatasetFunction:
  def __init__(self, sample_fn, init_fn, upload_fn):
    self._sample_fn = sample_fn
    self._init_fn = init_fn
    self._upload_fn = upload_fn
    assert callable(sample_fn)
    assert callable(init_fn)
    assert callable(upload_fn)

  def make_initializable_iterator(self):
    return DatasetFunctionIterator(self)

make_dataset_function = DatasetFunction


def is_integer(x):
  return np.can_cast(x, np.int32)

def is_float(x):
  return np.can_cast(x, np.float32)

def is_exact(x):
  return is_integer(x) or is_float(x) and x == int(x)

def num(x, digits_after_decimal=2):
  if is_integer(x):
    spec = '{:,d}'
  else:
    spec = '{:,.%df}' % digits_after_decimal
  return spec.format(x)

