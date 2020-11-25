import tensorflow as tf
tf1 = tf.compat.v1
import numpy as np
import sys
import os

from tensorflow.python.platform import tf_logging as logging

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
    #return tf.identity(then())
    return then()


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
  obarray.strings = tf.Variable(ks, dtype=tf.string, shape=ks.shape, use_resource=True, trainable=False, collections=['local_variables'])
  obarray.scratch = tf.Variable(-1, dtype=tf.int32, shape=[], use_resource=True, trainable=False, collections=['local_variables'])
  session.run([obarray.strings.initializer, obarray.scratch.initializer])
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


def enq(name, *values):
  return tpu_cpu(lambda vs: tf.raw_ops.Stage(values=vs, container=name, shared_name=name), values)

def dtypes_of(xs):
  return [x.dtype if hasattr(x, 'dtype') else x for x in xs]

def deq(name, *dtypes):
  return tpu_cpu(lambda: tf.raw_ops.Unstage(dtypes=dtypes_of(dtypes), container=name, shared_name=name))


def enq_metric(name, *values):
    #return enq(name, get_or_create_global_step(), tpu_id(), tpu_now(), *values)
    return enq(name, get_or_create_global_step(), tpu_id(), *values)

def deq_metric(name, *dtypes):
    #return deq(name, tf.int64, tf.int32, tf.float64, *dtypes)
    return deq(name, tf.int64, tf.int32, *dtypes)


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



def create_sum_step(name, graph=None):
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
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    return variable_scope.get_variable(
        name,
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        use_resource=True,
        aggregation=variables.VariableAggregation.SUM,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])



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



def difference(l1, l2):
  """List difference"""
  # TODO: support other types?
  return [i for i in l1 + l2 if i not in l1 or i not in l2]


#from tensorflow.python.tpu import tensor_tracer_report


from tensorflow.python.ops import control_flow_util


_DEVICE_TYPE_TPU = 'tpu'
_DEVICE_TYPE_CPU = 'cpu'


def loop_cond_op(op):
  return op.type in ('LoopCond', 'RefLoopCond')


def while_loop_op(op):
  """Returns true if op is one of the special ops of in a while loop.

  Args:
     op: A tf.Operation.

  Returns:
     True if the given op is one of [Switch, Merge, Enter, Exit,
     NextIteration, LoopCond], which are all building blocks for TF while
     loops.
  """
  return  (control_flow_util.IsLoopSwitch(op) or
           control_flow_util.IsLoopMerge(op) or
           control_flow_util.IsLoopEnter(op) or
           control_flow_util.IsLoopExit(op) or
           loop_cond_op(op) or
           op.type in ('RefNextIteration', 'NextIteration'))


def control_flow_op(op):
  """Returns true if op is one of the special ops of in a while loop.

  Args:
     op: A tf.Operation.

  Returns:
     True if the given op is one of [Switch, Merge, Enter, Exit,
     NextIteration, LoopCond], which are all building blocks for TF while
     loops.
  """
  return  (control_flow_util.IsSwitch(op) or
           control_flow_util.IsMerge(op))


def unsafe_op(op):
  """Returns True if this op is not safe to be traced."""

  # Reasons for not including following op types:
  #    Assign: cause incorrect result with CPU tracing.
  if op.type == 'Assign':
    return True
  return False


def device_mismatch(device_type, op):
  if device_type == _DEVICE_TYPE_TPU:
    # pylint: disable=protected-access
    return tpu._TPU_REPLICATE_ATTR not in op.node_def.attr
    # pylint: enable=protected-access
  return False


def unsafe_scalar_trace(op):
  """Return true if scalar output tensor from Op is not safe to be traced."""

  # Tracing the following causes cycle in the graph on TPU.
  if op.type in ('LoopCond', 'Enter', 'Merge', 'Const',
                 'Switch', 'Less', 'ReadVariableOp'):
    return True
  # Tracing the following will cause casting-issue
  # with the norm tracing mode or other compilation issues on CPU.
  if op.type in ('VarHandleOp', 'IteratorToStringHandle',
                 'IteratorGetNext', 'OneShotIterator',
                 'IteratorV2', 'MakeIterator',
                 'BatchDatasetV2', 'MapDataset',
                 'FixedLengthRecordDataset', 'TakeDataset', 'ZipDataset',
                 'Placeholder', 'PlaceholderWithDefault', 'StridedSlice'):
    return True
  return False




def topological_sort(operations=None):
  """Performs topological sort on the given graph.

  Args:
     operations: graph operations to sort topologically.

  Returns:
     A pair where the first element indicates if the topological
     sort succeeded (True if there is no cycle found; False if a
     cycle is found) and the second element is either the sorted
     list of nodes or the cycle of nodes found.
  """
  if operations is None:
    operations = tf.get_default_graph().get_operations()
  def _is_loop_edge(op):
    """Returns true if the op is the end of a while-loop creating a cycle."""
    return op.type in ['NextIteration']

  def _in_op_degree(op):
    """Returns the number of incoming edges to the given op.

    The edge calculation skips the edges that come from 'NextIteration' ops.
    NextIteration creates a cycle in the graph. We break cycles by treating
    this op as 'sink' and ignoring all outgoing edges from it.
    Args:
      op: Tf.Operation
    Returns:
      the number of incoming edges.
    """
    count = 0
    for op in op.control_inputs + [in_tensor.op for in_tensor in op.inputs]:
      if not _is_loop_edge(op):
        count += 1
    return count

  sorted_ops = []
  op_in_degree = {op: _in_op_degree(op) for op in operations}

  frontier = [op for (op, degree) in op_in_degree.items() if degree == 0]
  frontier.sort(key=lambda op: op.name)
  while frontier:
    op = frontier.pop()
    # Remove the op from graph, and remove its outgoing edges.
    sorted_ops.append(op)
    if _is_loop_edge(op):
      continue
    # pylint: disable=protected-access
    consumers = list(op._control_outputs)
    # pylint: enable=protected-access
    for out_tensor in op.outputs:
      consumers += [consumer_op for consumer_op in out_tensor.consumers()]
    consumers.sort(key=lambda op: op.name)
    for consumer in consumers:
      # For each deleted edge shift the bucket of the vertex.
      op_in_degree[consumer] -= 1
      if op_in_degree[consumer] == 0:
        frontier.append(consumer)
      if op_in_degree[consumer] < 0:
        raise ValueError('consumer:%s degree mismatch'%consumer.name)

  left_ops = set(op for (op, degree) in op_in_degree.items() if degree > 0)
  if left_ops:
    return (True, left_ops)
  else:
    assert len(operations) == len(sorted_ops)
    return (False, sorted_ops)



import collections


def sort_tensors_and_ops(graph=None):
  """Returns a wrapper that has consistent tensor and op orders."""
  if graph is None:
    graph = tf.get_default_graph()
  graph_wrapper = collections.namedtuple('GraphWrapper',
                                         ['graph', 'operations', 'op_to_idx',
                                          'tensors', 'tensor_to_idx',
                                          'contains_cycle',
                                          'topological_order_or_cycle'])
  contains_cycle, topological_order_or_cycle = topological_sort(graph.get_operations())
  if not contains_cycle:
    operations = topological_order_or_cycle
  else:
    operations = graph.get_operations()
  op_to_idx = {op.name: index for index, op
               in enumerate(operations)}
  tensors = []
  for op in operations:
    tensors.extend(op.outputs)
  tensor_to_idx = {tensor.name: index for index, tensor in
                   enumerate(tensors)}
  return graph_wrapper(graph=graph, operations=operations, op_to_idx=op_to_idx,
                       tensors=tensors, tensor_to_idx=tensor_to_idx,
                       contains_cycle=contains_cycle,
                       topological_order_or_cycle=topological_order_or_cycle)



def _process_tensor_fetches(tensor_fetches):
  """Check that tensor_fetches is not empty and have valid tensors."""
  # If none or empty list.
  if tensor_fetches is None:
    raise RuntimeError('tensor_fetches provided to tensor_tracer cannot be '
                       'None.')
  if not isinstance(tensor_fetches, (list, tuple)):
    tensor_fetches = [tensor_fetches]
  elif not tensor_fetches:
    raise RuntimeError('tensor_fetches provided to tensor_tracer cannot be '
                       'empty list.')
  fetches = []
  for fetch in tensor_fetches:
    if isinstance(fetch, ops.Tensor):
      fetches.append(fetch)
    else:
      raise RuntimeError('Given tensor_fetch:%s is not a tensor.' % fetch)
  return fetches


def _process_op_fetches(op_fetches):
  """Check that op_fetches have valid ops."""
  if op_fetches is None:
    return []

  if not isinstance(op_fetches, (list, tuple)):
    op_fetches = [op_fetches]

  fetches = []
  for fetch in op_fetches:
    if isinstance(fetch, ops.Operation):
      fetches.append(fetch)
    elif isinstance(fetch, ops.Tensor):
      fetches.append(fetch.op)
    else:
      logging.warning('Ignoring the given op_fetch:%s, which is not an op.' %
                      fetch)
  return fetches


def _get_op_control_flow_context(op):
  """Returns the control flow of the given op.

  Args:
    op: tf.Operation for which the control flow context is requested.
  Returns:
    op_control_flow_context: which the is control flow context of the given
    op. If the operation type is LoopExit, returns the outer control flow
    context.
  """
  # pylint: disable=protected-access
  op_control_flow_context = op._control_flow_context
  # pylint: enable=protected-access
  if control_flow_util.IsLoopExit(op):
    op_control_flow_context = op_control_flow_context.outer_context
  return op_control_flow_context


def get_execution_ops(node, operations=None):
  return _filter_execution_path_operations(get_all_fetches(node), operations=operations)


def _filter_execution_path_operations(fetches, operations=None):
  """Returns the set of ops in the execution path to compute given fetches."""
  if operations is None:
    operations = tf.get_default_graph().get_operations()

  # If no fetch provided, then return all operations.
  if fetches is None:
    return list(operations)
  # Convert to list, if a single element is provided.
  if not isinstance(fetches, (list, tuple)):
    fetches = [fetches]
  # If a tensor is given as fetch, convert it to op.
  op_fetches = []
  for fetch in fetches:
    if isinstance(fetch, ops.Operation):
      op_fetches.append(fetch)
    elif isinstance(fetch, ops.Tensor):
      op_fetches.append(fetch.op)
    else:
      raise RuntimeError('Given fetch:%s is neither a tensor nor an op.'
                         %fetch)

  execution_path_operations_ordered = list(op_fetches)
  execution_path_operations = set(op_fetches)
  traverse_stack = list(op_fetches)
  while True:
    if not traverse_stack:
      break
    head_op = traverse_stack.pop()
    input_ops = [tensor_input.op for tensor_input in head_op.inputs]
    input_ops.extend(head_op.control_inputs)

    for input_op in input_ops:
      if input_op not in execution_path_operations:
        # Filter out loop condition operations, tracing them causes a cycle.
        # Trace only the loop-body.
        if loop_cond_op(input_op):
          continue
        execution_path_operations.add(input_op)
        execution_path_operations_ordered.append(input_op)
        traverse_stack.append(input_op)
  return execution_path_operations_ordered



def get_all_fetches(tensor_fetches, op_fetches=None):
    """Convert all non-operations (tensors, etc) into fetch operations.

    Args:
      tensor_fetches: a (list,tuple,or a single object) of tensor fetches
        returned by model_fn given to session.run. Function must be provided
        with as least one tensor to fetch.
      op_fetches: A list of op fetches returned by model_fn given to
        session.run. op_fetches and tensor_fetches are used to determine the
        nodes that will be executed. Can be None.

    Returns:
      tensor_fetches: an exact copy of tensor_fetches that has additional
                      dependencies.
    Raises:
      RuntimeError: If tensor_fetches is None or empty.
    """
    processed_t_fetches = _process_tensor_fetches(tensor_fetches)
    op_fetches = _process_op_fetches(op_fetches)
    all_fetches = op_fetches + [tensor.op for tensor in processed_t_fetches]
    return all_fetches

import os


def tf_trim_traceback(tb):
  core = os.path.join('site-packages', 'tensorflow_core')
  site = os.path.sep + 'site-packages' + os.path.sep
  frames = []
  is_core = []
  for frame in tb:
    file, line, function, code = frame
    is_core.append(core in file)
    if core in file:
      file = '@tensorflow_core' + file.split(core, 1)[1]
    if site in file:
      file = '@' + file.split(site, 1)[1]
    frames.append((file, line, function, code))
  last_frame = None
  while len(frames) > 0 and is_core[-1]:
    last_frame = frames.pop()
    is_core.pop()
  if last_frame is not None:
    frames.append(last_frame)
  return frames


import json


def escape(s):
  return json.dumps(s)


import cachetools


@cachetools.cached(cachetools.TTLCache(maxsize=128, ttl=2))
def getcwd():
  return os.getcwd()


def pretty_traceback_frame(frame):
  file, line, function, code = frame
  cwd = getcwd()
  if file.startswith(cwd):
    file = file[len(cwd)+1:]
  #return '\n  File {file}, line {line}, in {function}\n    {code}'.format(
  return '{file}:{line} ({function}):\n    {code}'.format(
      file=file, line=line, function=function, code=code)


def pretty_traceback(tb):
  return [pretty_traceback_frame(frame) for frame in tb][::-1]


def tf_traceback(node, pretty=True, trim=True):
  #fetches = get_all_fetches(node)
  # assert len(fetches) > 0
  # tb = fetches[0].traceback
  #if isinstance(node, tf.Tensor):
  if not isinstance(node, tf.Operation) and hasattr(node, 'op'):
    node = node.op
  tb = node.traceback
  if trim:
    tb = tf_trim_traceback(tb)
  if pretty:
    tb = pretty_traceback(tb)
  return tb

def tf_traceback_message(node):
  return '\n'.join(['-------', repr(node)] + tf_traceback(node))

def tf_tracebacks(nodes):
  return '\n'.join([tf_traceback_message(node) for node in nodes])
