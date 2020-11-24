import tensorflow as tf
tf1 = tf.compat.v1
import numpy as np
import sys
import os

from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.tpu import tpu
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import topology as topology_lib



def after(op, then):
  op = [op] if not isinstance(op, (list, tuple)) else op
  with tf.control_dependencies(op):
    return tf.identity(then())


def count_tpu_cores(session=None):
  if session is None:
    session = tf1.get_default_session()
  return len([x for x in session.list_devices() if ':TPU:' in x.name])

import functools

def tpu_shard(op, num_shards=None, device_assignment=None, outputs_from_all_shards=True, **kws):
  if num_shards is None:
    if device_assignment is not None:
      num_shards = len(device_assignment.core_assignment)
    else:
      num_shards = count_tpu_cores()
  return tpu.shard(op, outputs_from_all_shards=outputs_from_all_shards, num_shards=num_shards, device_assignment=device_assignment, **kws)

def tpu_id():
  # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
  replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
  return replica_id


def tpu_cpu(f, *args, **kws):
  graph = tf.get_default_graph()
  context = graph._get_control_flow_context()
  #print(context._outside_compilation_cluster)
  #print(context._outside_compilation_counter)
  if context is not None and context._outside_compilation_counter > 0:
    return f(*args, **kws)
  else:
    return tpu.outside_compilation(f, *args, **kws)

def tpu_now():
  return tpu_cpu(lambda: tf.identity(tf.timestamp(), name="timestamp"))


def tf_fmt(str, *args, remove_quotes=False):
  def on_cpu(str):
    str = tf.strings.format(str, args)
    if remove_quotes:
      str = tf.strings.regex_replace(str, '["]', '')
    return str
  return tpu_cpu(on_cpu, str)


def tf_get(table, key):
  return tpu_cpu(lambda: table.lookup(key))


def tf_set(table, key, value):
  return tpu_cpu(lambda: table.insert(key, value))


def tf_escape(value):
  return tf_fmt("{}", value)


def tf_index(h, *ks):
  parts = [h]
  for k in ks:
    parts += ["[", tf_escape(k), "]"]
  return tf.strings.join(parts)


def tf_hget(table, symbol, property):
  def cpu(symbol, property):
    k = tf_index(symbol, property)
    return table.lookup(k)
  return tpu_cpu(cpu, symbol, property)


def tf_hset(table, symbol, property, value):
  def cpu(symbol, property, value):
    k = tf_index(symbol, property)
    return table.insert(k, value)
  return tpu_cpu(cpu, symbol, property, value)



def table_new(key_dtype=tf.string, value_dtype=tf.float32, default_value=float('nan'), empty_key="", deleted_key="[deleted]", name='MutableDenseHashTable'):
  table = tf.lookup.experimental.DenseHashTable(key_dtype=key_dtype, value_dtype=value_dtype, default_value=default_value, empty_key=empty_key, deleted_key=deleted_key, name=name)
  table.key = tf.placeholder(key_dtype, shape=[None], name="table_key")
  table.val = tf.placeholder(value_dtype, shape=None, name="table_key")
  table.get_op = table.lookup(table.key, "table_get_op")
  table.set_op = table.insert(table.key, table.val, "table_set_op")
  table.wipe_op = table.erase(table.key, "table_wipe_op")
  table.export_op = table.export()
  table.len_op = table.size(name="table_len_op")
  return table


def table_export(table, session=None):
  if session is None:
    session = tf.get_default_session()
  ks, vs = session.run(table.export_op)
  return [(k[0].decode('utf8'), vs[i][0]) for i, k in enumerate(ks) if k != b'']


def table_update(table, pydict, session=None):
  if session is None:
    session = tf.get_default_session()
  entries = list(pydict.items())
  keys = [k for k, v in entries]
  vals = [v for k, v in entries]
  session.run(table.set_op, {table.key: keys, table.val: vals})



def table_obarray(session=None):
  if session is None:
    session = tf.get_default_session()
  obarray = table_new(tf.string, tf.int32, -1)
  #ks, vs = session.run(obarray.export_op)
  ks = obarray.export_op[0]
  #obarray.strings = tf.Variable(obarray.export_op[0:1], dtype=tf.string, shape=[None, 1], use_resource=True)
  obarray.strings = tf.Variable(ks, dtype=tf.string, shape=ks.shape, use_resource=True)
  session.run(obarray.strings.initializer)
  def intern(k):
    def cpu(k):
      i = obarray.lookup(k)
      def body():
        n = tf.cast(obarray.size(), tf.int32)
        with tf.control_dependencies([obarray.insert(k, n)]):
          i0 = obarray.lookup(k)
          # indices = tf.constant([[4], [3], [1] ,[7]])
          # updates = tf.constant([9, 10, 11, 12])
          # op = ref.scatter_nd_update(indices, updates)
          #indices = tf.reshape(i0, (1, 1))
          #updates = tf.reshape(k, (1,))
          #with tf.control_dependencies([obarray.strings.scatter_nd_update(indices, updates)]):
          with tf.control_dependencies([obarray.strings[i0].assign([k])]):
            return tf.identity(i0)
      return tf.cond(i >= 0, lambda: i, body)
    return tpu_cpu(cpu, k)
  obarray.intern = intern
  return obarray


def table_example(session=None):
  if session is None:
    session = tf.get_default_session()
  table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string, value_dtype=tf.float32, default_value=0, empty_key="", deleted_key="[deleted]")
  with tf.control_dependencies([
    table.insert("lr", 0.5),
    table.insert("x", 99),
    table.insert("y[0]", 99),
    table.insert("y[1]", 9),
    table.insert("y.shape", abs(99-9)),
    table.insert("y.size", 2),
    table.insert("i", 0),
    ]):
    return table, tf.no_op()

def table_example_2(*, session=None, **table_settings):
  if session is None:
    session = tf.get_default_session()
  table = table_new(**table_settings)
  session.run(table.set_op, {
    table.key: "lr   x   y[0] y[1] y.shape    y.size i".split(),
    table.val: [0.5, 99, 99,  9,   abs(99-9), 2,     0]
    })
  return table


def enq(*values, name):
  return tpu_cpu(lambda vs: tf.raw_ops.Stage(values=vs, container=name, shared_name=name), values)

def dtypes_of(xs):
  return [x.dtype if hasattr(x, 'dtype') else x for x in xs]

def deq(*dtypes, name):
  return tpu_cpu(lambda: tf.raw_ops.Unstage(dtypes=dtypes_of(dtypes), container=name, shared_name=name))


def enq_metric(name, *values):
    return enq(get_or_create_global_step(), tpu_id(), tpu_now(), *values, name=name)

def deq_metric(name, *dtypes):
    return deq(tf.int64, tf.int32, tf.float64, *dtypes, name=name)


import functools
import sys

class SparseSum(tf.compat.v1.SparseConditionalAccumulator):
  def __init__(self, dtype=tf.float32, reduction_type='SUM', **kws):
    super(SparseSum, self).__init__(dtype=dtype, reduction_type=reduction_type, **kws)
    self.size_op = self.num_accumulated()
    self.take_op = self.take_grad(1)
    self.safe_op = tf.cond(self.size_op > 0, lambda: tuple(self.take_grad(1)), lambda: (tf.zeros([], dtype=tf.int64), tf.zeros([], dtype=dtype), tf.constant([], tf.int64)))
  def apply_grad(self, grad_indices, grad_values, grad_shape=None, local_step=sys.maxsize, name=None):
    return super(SparseSum, self).apply_grad(grad_indices, grad_values, grad_shape, local_step=local_step, name=name)
  def apply_indexed_slices_grad(self, grad, local_step=sys.maxsize, name=None):
    return super(SparseSum, self).apply_indexed_slices_grad(grad, local_step=local_step, name=name)
  def take_grad(self, num_required=1, name=None):
    return super(SparseSum, self).take_grad(num_required=num_required, name=name)
  def take(self):
    indices, values, shape = self.safe_op
    return tf.identity(indices, name='indices'), tf.identity(values, name='values'), tf.identity(shape, name='shape'),



from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables


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



import gin

def parse_string(s, included=[]):
  if isinstance(s, list):
    s = '\n'.join(s)
  parser = gin.config_parser.ConfigParser(s, gin.config.ParserDelegate(skip_unknown=True))
  for statement in parser:
    if isinstance(statement, gin.config_parser.IncludeStatement):
      if statement.filename in included:
        print('Skipping circular dependency: {}'.format(statement.filename))
      else:
        body = include(statement.filename)
        for k, v in parse_string(body, included.union([statement.filename])):
          yield k, v
    elif isinstance(statement, gin.config_parser.ImportStatement):
      yield statement.module, '@import'
    elif hasattr(statement, 'selector'):
      v = statement.value
      k = statement.arg_name
      if isinstance(k, str) and len(k.strip()) > 0:
        k = '{}.{}'.format(statement.selector, statement.arg_name)
      else:
        k = statement.selector
      k = os.path.join(statement.scope or '', k)
      v = statement.value
      yield k, v
    else:
      raise Exception("Bad statement {}".format(statement))

