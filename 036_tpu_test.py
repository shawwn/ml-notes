from pprint import pprint as pp
import tqdm

import tensorflow as tf
tf1 = tf.compat.v1

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

from tensorflow.python import tf2
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import one_device_strategy as one_device_lib
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_keras_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_keras_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_keras_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_keras_v2
from tensorflow.python.keras.optimizer_v2 import ftrl as ftrl_keras_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_keras_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_keras_v2
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.training import adagrad
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import rmsprop

from tensorflow.python.distribute import distribution_strategy_context

#from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver

import os

from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

#import tpu_cluster_resolver
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver

class TPUClusterResolver(tpu_cluster_resolver.TPUClusterResolver):
  def _start_local_server(self):
    #addr = self._request_compute_metadata('instance/network-interfaces/0/ip')
    addr = '127.0.0.1'
    address = compat.as_text(addr)
    self._server = server_lib.Server(
        {
            'local': ['0.0.0.0:0']
        }, protocol='grpc', config=None, start=True)
    # self._server.target is of the form: grpc://ipaddress:port
    target = compat.as_bytes(self._server.target)
    splits = target.split(compat.as_bytes(':'))
    assert len(splits) == 3, self._server.target
    assert splits[0] == compat.as_bytes('grpc'), self._server.target
    self._coordinator_port = compat.as_text(splits[2])
    self._coordinator_address = '%s:%s' % (
        address, compat.as_text(self._coordinator_port))

TPUClusterResolver = tpu_cluster_resolver.TPUClusterResolver
import tflex
TPUClusterResolver = tflex.TPUClusterResolver

# pylint: disable=missing-docstring
def _get_tpu_strategy_creator(steps_per_run, use_single_core=False, **kwargs):
  def _create_tpu_strategy():
    #resolver = TPUClusterResolver(os.environ['TPU_NAME'], coordinator_name='coordinator')
    resolver = TPUClusterResolver(os.environ['TPU_NAME'])
    topology = tpu_strategy_util.initialize_tpu_system(resolver)
    device_assignment = None
    if use_single_core:
      device_assignment = device_assignment_lib.DeviceAssignment(
          topology, core_assignment=device_assignment_lib.
          SINGLE_CORE_ASSIGNMENT)

    # Steps per run is only supported in TF 1.x
    if tf2.enabled():
      return tpu_lib.TPUStrategy(resolver, device_assignment, **kwargs)
    else:
      return tpu_lib.TPUStrategyV1(resolver, steps_per_run,
                                   device_assignment, **kwargs)
  return _create_tpu_strategy

tpu_strategy = combinations.NamedDistribution("TPU", _get_tpu_strategy_creator(steps_per_run=2), required_tpu=True)

from tensorflow.python.ops import variable_scope

class MockModel(object):

  def __init__(self, two_variables=False):
    self.variables = []
    self.variables.append(variable_scope.variable(1.25, name="dummy_var1"))
    if two_variables:
      self.variables.append(variable_scope.variable(2.0, name="dummy_var2"))

  def __call__(self, factor=2):
    x = factor * self.variables[0]
    if len(self.variables) > 1:
      x += self.variables[1]
    return x

from tensorflow.python.keras.engine import training as keras_training

class MiniModel(keras_training.Model):
  """Minimal model for mnist.

  Useful for testing and debugging on slow TPU simulators.
  """

  def __init__(self):
    super(MiniModel, self).__init__(name="")
    self.fc = keras_core.Dense(1, name="fc", kernel_initializer="ones",
                               bias_initializer="ones")

  def call(self, inputs, training=True):
    inputs = array_ops.ones([1, 10])
    return self.fc(inputs)


from tensorflow.python.eager import test


class TwoDeviceDistributionTestBase(test.TestCase):
  """Some tests that should work with any two-device DistributionStrategy."""

  def _test_run(self, strategy):
    out1 = strategy.experimental_run_v2(
        lambda: ds_context.get_replica_context().replica_id_in_sync_group + 1)
    self.assertAllEqual([1, 2], self.evaluate(strategy.unwrap(out1)))

    out2 = strategy.experimental_run_v2(
        lambda x: {"a": x * 2, "b": x * x}, args=(out1,))
    out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
    self.assertAllEqual([2, 4], out2_vals["a"])
    self.assertAllEqual([1, 4], out2_vals["b"])

    out3 = strategy.experimental_run_v2(lambda b, a: a + 2 * b + 2, kwargs=out2)
    self.assertAllEqual([6, 14], self.evaluate(strategy.unwrap(out3)))


from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables

def _replica_id():
  replica_id = ds_context.get_replica_context().replica_id_in_sync_group
  if not isinstance(replica_id, ops.Tensor):
    replica_id = constant_op.constant(replica_id)
  return replica_id


def _mimic_two_cpus():
  cpus = config.list_physical_devices("CPU")

  config.set_logical_device_configuration(cpus[0], [
      context.LogicalDeviceConfiguration(),
      context.LogicalDeviceConfiguration(),
  ])


import strategy_test_lib


class MirroredTwoDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    TwoDeviceDistributionTestBase,
    test.TestCase):
  def testRunRegroupError(self, distribution=tpu_strategy):
    def run_fn():
      replica_id = int(self.evaluate(_replica_id()))
      # Generates a list with different lengths on different devices.
      # Will fail in _regroup() (if more than one device).
      return list(range(replica_id))

    with distribution.scope(), self.assertRaises(AssertionError):
      distribution.extended.call_for_each_replica(run_fn)

class Namespace():
  pass

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

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

  assert_global_step(global_step_tensor)
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
  if context.executing_eagerly():
    with ops.device('cpu:0'):
      return variable_scope.get_variable(
          ops.GraphKeys.GLOBAL_STEP,
          shape=[],
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
          collections=[
              ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP
          ])
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


from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import step_fn
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def minimize_loss_example(optimizer, use_bias=False, use_callable_loss=True):
  """Example of non-distribution-aware legacy code."""

  def dataset_fn():
    dataset = dataset_ops.Dataset.from_tensors([[1.]]).repeat()
    # TODO(isaprykin): batch with drop_remainder causes shapes to be
    # fully defined for TPU.  Remove this when XLA supports dynamic shapes.
    return dataset.batch(1, drop_remainder=True)

  layer = core.Dense(1, use_bias=use_bias)

  def model_fn(x):
    """A very simple model written by the user."""

    def loss_fn():
      y = array_ops.reshape(layer(x), []) - constant_op.constant(1.)
      return y * y

    nonlocal optimizer
    if callable(optimizer):
      optimizer = optimizer()
    if isinstance(optimizer, optimizer_v2.OptimizerV2):
      return optimizer.minimize(loss_fn, lambda: layer.trainable_variables)
    elif use_callable_loss:
      return optimizer.minimize(loss_fn)
    else:
      return optimizer.minimize(loss_fn())

  return model_fn, dataset_fn, layer



from tensorflow.python.distribute.single_loss_example import batchnorm_example
#from tensorflow.python.distribute.single_loss_example import minimize_loss_example
from tensorflow.python.ops import variables as variables_lib


from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.lib.io import python_io
from tensorflow.python.platform import test
from tensorflow.python.tpu import datasets
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat



def main(sess):
  print(sess.run(tf.add(1, 2)))

  class Self():
    def __init__(self, sess):
      self._sess = sess
      self._devices = sess.list_devices()
      self._host_devices = [x.name for x in self._devices if ':CPU:' in x.name]
      self._tpu_devices = [x.name for x in self._devices if ':TPU:' in x.name]
      self._worker_device = self._host_devices[0]

    def cached_session(self):
      return self._sess

    def evaluate(self, *args, **kws):
      #import pdb; pdb.set_trace()
      return self._sess.run(*args, **kws)
    
    def _get_iterator(self, strategy, input_fn):
      with tf.device(self.host_devices[0]):
        def in_fn(ctx):
          #import pdb; pdb.set_trace()
          with tf.device(self.host_devices[0]):
            return input_fn()
        iterator = strategy.make_input_fn_iterator(in_fn)
        #self.evaluate(iterator.initialize())
      return iterator

    def _get_iterator(self, strategy, input_fn):
      iterator = strategy.make_input_fn_iterator(lambda _: input_fn())
      return iterator

  #def evaluate(*args, **kws):
  #  import pdb; pdb.set_trace()
    

  #self = Namespace()
  #self.evaluate = evaluate

  self = Self(sess)

  distribution = tpu_strategy.strategy

  #gs = tf.train.get_or_create_global_step()
  gs = get_or_create_global_step()
  sess.run(variables_lib.global_variables_initializer())

  def test1():
    print('test1')
    def run_fn(*args, **kws):
      pp({'args': args, 'kws': kws})
      #replica_id = tf.constant(4)
      replica_id = args[0]
      gs = tf1.train.get_or_create_global_step()
      #import pdb; pdb.set_trace()
      #replica_id = int(self.evaluate(_replica_id()))
      #print('replica_id', replica_id)
      # Generates a list with different lengths on different devices.
      # Will fail in _regroup() (if more than one device).
      return replica_id, tf1.assign_add(gs, 1)

    #import pdb; pdb.set_trace()
    #with distribution.scope():#, self.assertRaises(AssertionError):
    #  distribution.extended.call_for_each_replica(run_fn)
    print('step', sess.run(gs))
    for _ in tqdm.trange(2, desc="test1A"):
      outputs = distribution.experimental_run_v2(run_fn, args=(tf.constant(_),gs,))
      result = sess.run([distribution.unwrap(x) for x in outputs])
      print('step', sess.run(gs))
      pp(result)
    for _ in tqdm.trange(2, desc="test1B"):
      outputs = distribution.experimental_run_v2(run_fn, args=(tf.constant(_),gs,))
      result = sess.run([distribution.unwrap(x) for x in outputs])
      print('step', sess.run(gs))
      pp(result)

  #test1()

  def test4(temp_dir):
    filenames = []
    all_contents = []
    _NUM_FILES=10
    _NUM_ENTRIES=20
    for i in range(_NUM_FILES):
      filename = os.path.join(temp_dir, 'tf_record.%d' % i)
      filenames.append(filename)
      writer = python_io.TFRecordWriter(filename)
      for j in range(_NUM_ENTRIES):
        record = compat.as_bytes('Record %d of file %d' % (j, i))
        writer.write(record)
        all_contents.append(record)
      writer.close()

    import pdb; pdb.set_trace()

    filenames = dataset_ops.Dataset.from_tensor_slices(filenames)

    dataset = datasets.StreamingFilesDataset(filenames, filetype='tfrecord', file_reader_job='worker/replica:0/task:0/device:CPU:0')

    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = []
    for _ in range(4 * len(all_contents)):
      retrieved_values.append(compat.as_bytes(self._sess.run(get_next)))


  import tempfile
  with tempfile.TemporaryDirectory() as d:
    test4(d)

  def test3():
    
    def my_generator():
      yield (1, [1] * 10)

    def gen_dataset(dummy):
      return dataset_ops.Dataset.from_generator(
          my_generator, (dtypes.int64, dtypes.int64),
          (tensor_shape.TensorShape([]), tensor_shape.TensorShape([10])))

    import pdb; pdb.set_trace()

    #dataset = datasets.StreamingFilesDataset(dataset_ops.Dataset.range(10), filetype=gen_dataset)
    dataset = datasets.StreamingFilesDataset(dataset_ops.Dataset.range(10), filetype=gen_dataset, file_reader_job='worker/replica:0/task:0/device:CPU:0')

    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()

    retrieved_values = self._sess.run(get_next)

  test3()

  def test2():
    print('test2')
    optimizer_fn = tf1.train.AdamOptimizer
    #use_callable_loss = True
    use_callable_loss = False
    with distribution.scope():
      def optimizer_fn():
        #opt = tf1.train.AdamOptimizer()
        opt = tf1.train.GradientDescentOptimizer(0.01)
        if 'TPU_NAME' in os.environ:
          opt = tf1.tpu.CrossShardOptimizer(opt)
        return opt
      optimizer = optimizer_fn
      model_fn, dataset_fn, layer = minimize_loss_example(
          optimizer, use_bias=True, use_callable_loss=use_callable_loss)

      def step_fn(ctx, inputs):
        del ctx  # Unused
        return distribution.group(
            distribution.extended.call_for_each_replica(
                model_fn, args=(inputs,)))

      iterator = self._get_iterator(distribution, dataset_fn)

      def run_step():
        return distribution.extended.experimental_run_steps_on_iterator(
            step_fn, iterator, iterations=2).run_op

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          run_step = sess.make_callable(run_step())
      self.evaluate(variables_lib.global_variables_initializer())

      weights, biases = [], []
      for _ in tqdm.trange(5, desc="test2"):
        result = run_step()
        pp(result)
        weights.append(self.evaluate(layer.kernel))
        biases.append(self.evaluate(layer.bias))

      error = abs(numpy.add(numpy.squeeze(weights), numpy.squeeze(biases)) - 1)
      is_not_increasing = all(y <= x for x, y in zip(error, error[1:]))
      

  test2()
    
  import pdb; pdb.set_trace()
  
from tensorflow.core.protobuf import rewriter_config_pb2

def with_session(fn):
  timeout_in_ms=60000
  config = tf1.ConfigProto(operation_timeout_in_ms=timeout_in_ms,
                          graph_options=tf1.GraphOptions(
                            rewrite_options=rewriter_config_pb2.RewriterConfig(
                              disable_meta_optimizer=True)),
                          isolate_session_state=True)
  #cluster_resolver = TPUClusterResolver(os.environ['TPU_NAME'], coordinator_name='coordinator')
  cluster_resolver = TPUClusterResolver(os.environ['TPU_NAME'])
  cluster_spec = cluster_resolver.cluster_spec()
  if cluster_spec:
    config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
  
  with tf1.Session(cluster_resolver.get_master(), config=config).as_default() as sess:
    #sess.run(tf1.tpu.initialize_system())
    return main(sess)

if __name__ == "__main__":
  if 'TPU_NAME' not in os.environ:
    os.environ['TPU_NAME'] = 'grpc://0.tcp.ngrok.io:12621'
  #if 'TPU_NAME' in os.environ:
  #  if 'KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS' not in os.environ:
  #    os.environ['KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS'] = os.environ['TPU_NAME']
  #test.main()
  with_session(main)
  

