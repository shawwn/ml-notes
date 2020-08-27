acc = tf.raw_ops.ConditionalAccumulator(dtype=tf.float32, shape=(), container="tf.float32", shared_name="tf.float32", reduction_type="SUM", name='tf.float32')
sess.run(acc)
v = tf.Variable(1.0, dtype=tf.float32, collections=['local_variables'], name='tf.float32-inc')
sess.run(v.initializer)
i = tf.Variable(0, dtype=tf.int64, collections=['local_variables'], name='tf.float32-step')
i_incr = i.assign_add(1)
sess.run(i.initializer)
op = tf.raw_ops.AccumulatorApplyGradient(handle=acc, local_step=i.value(), gradient=v.value())
num = tf.raw_ops.AccumulatorNumAccumulated(handle=acc)

with tf.control_dependencies([i_incr]):
  nxt = tf.raw_ops.AccumulatorTakeGradient(handle=acc, num_required=1, dtype=tf.float32)


sess.run(op)
sess.run(nxt)
sess.run(op)
sess.run(nxt)
sess.run(num)
sess.run(op)
sess.run(num)


a = tf.placeholder(tf.float32, shape=[])
b = tf.placeholder(tf.float32, shape=[])
c = tf.placeholder(tf.float32, shape=[])
r1 = tf.add(a, b)
r2 = tf.multiply(r1, c)

h = sess.partial_run_setup([r1, r2], [a, b, c])
res1 = sess.partial_run(h, r1, feed_dict={a: 1, b: 2}); res1
res2 = sess.partial_run(h, r2, feed_dict={c: res1}); res2





import functools

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import queue_runner_impl


zero64 = constant_op.constant(0, dtype=dtypes.int64)
var0 = variables.VariableV1(zero64)
count_up_to_3 = var0.count_up_to(3)
var1 = variables.VariableV1(zero64)
count_up_to_30 = var1.count_up_to(30)
queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
qr = queue_runner_impl.QueueRunner(queue, [count_up_to_3, count_up_to_30])
threads = qr.create_threads(sess)






from tensorflow.python import summary
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import def_function
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_feed


def create_test_xla_compile_context():
  computation_name = ops.get_default_graph().unique_name('computation')
  pivot = control_flow_ops.no_op(name=computation_name + '/pivot')
  return xla.XLACompileContext(name=computation_name, pivot=pivot)





a = variable_scope.get_variable(name='variable_a', use_resource=True, initializer=1)

context = create_test_xla_compile_context()
context.Enter()
a.assign(2)
context.Exit()


@def_function.function
def func():
  context = create_test_xla_compile_context()
  context.Enter()
  o = a.assign(2)
  context.Exit()
  return o


op = lambda x: tpu_ops.tpu_ops.collective_permute(x, [[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]])
zz = tpu_ops.shard(op, outputs_from_all_shards=True, num_shards=8, inputs=[[tf.constant([x+1], dtype=tf.float32) for x in range(8)]]); sess.run(zz)


ops = []
for core in range(8):
  for step in range(8):
    with tf.control_dependencies(ops):
      ops.append(tpu_ops.tpu_ops.infeed_enqueue([tf.constant(step, tf.float32)], shape=[1], device_ordinal=core))


topology = tpu_topology
topology_rank = len(topology.mesh_shape)
mesh_shape = topology.mesh_shape

computation_shape = None
computation_stride = None
num_replicas = 3



if computation_shape is None:
  computation_shape = np.array([1] * topology_rank, dtype=np.int32)
else:
  computation_shape = np.asarray(computation_shape, dtype=np.int32)


if computation_stride is None:
  computation_stride = np.array([1] * topology_rank, dtype=np.int32)
else:
  computation_stride = np.asarray(computation_stride, dtype=np.int32)


if computation_shape.shape != (topology_rank,):
  raise ValueError("computation_shape must have shape [{}]; got {}".format(
      topology_rank, computation_shape.shape))


if computation_stride.shape != (topology_rank,):
  raise ValueError("computation_stride must have shape [{}]; got {}".format(
      topology_rank, computation_stride.shape))


if any(computation_shape < 1):
  raise ValueError(
      "computation_shape must be positive; got computation_shape={}".format(
          computation_shape))


if any(computation_stride < 1):
  raise ValueError(
      "computation_stride must be positive; got computation_stride={}".format(
          computation_stride))



# Computes the physical size of one computation instance.
computation_footprint = computation_shape * computation_stride
if any(computation_footprint > mesh_shape):
  raise ValueError(
      "computation footprint {} does not fit in TPU topology shape {}".format(
          computation_footprint, mesh_shape))


# Computes how many copies of the computation footprint fit in the mesh.
block_counts = mesh_shape // computation_footprint


replica_counts = block_counts * computation_stride
max_replicas = np.prod(replica_counts)


if num_replicas > max_replicas:
  raise ValueError(
      "requested {} replicas but only {} replicas with shape {} and "
      "computation_stride {} fit in a TPU mesh of shape {}".format(
          num_replicas, max_replicas, computation_shape, computation_stride,
          mesh_shape))


def ceil_of_ratio(n, m):
  return (n + m - 1) // m
      

def _invert_topology(self):
  """Inverts a [task,device,axis] topology to [x,y,z] -> task/device maps."""
  tasks = np.full(list(self.mesh_shape), -1, dtype=np.int32)
  devices = np.full(list(self.mesh_shape), -1, dtype=np.int32)
  for task in xrange(self.device_coordinates.shape[0]):
    for device in xrange(self.device_coordinates.shape[1]):
      x, y, z = self.device_coordinates[task, device, :]
      tasks[x, y, z] = task
      devices[x, y, z] = device
  return tasks, devices

topology._topology_tasks, topology._topology_devices = _invert_topology(topology)
topology._missing_devices = np.argwhere(topology._topology_tasks < 0)



if topology.missing_devices.size == 0:
  replica_shape = [0] * topology_rank
  if num_replicas > 0:
    remaining_replicas = num_replicas
    remaining_dims = topology_rank

    # Choose dimensions as close to an equal cube as possible,
    # in order of increasing dimension size. By visiting dimensions
    # in increasing size, we assign the most constrained dimension
    # first, so we won't make infeasible choices.
    #
    # As a secondary sort order, visit the dimensions in reverse
    # order. This means we try to use both cores on the same chip
    # in preference to two cores on different chips.

    for x, ni in sorted(((x, -i) for (i, x) in enumerate(replica_counts))):
      i = -ni
      target_size = int(math.ceil(remaining_replicas**(1.0 / remaining_dims)))
      replica_shape[i] = min(target_size, x)
      remaining_replicas = ceil_of_ratio(remaining_replicas, replica_shape[i])
      remaining_dims -= 1

    assert remaining_replicas == 1 and remaining_dims == 0

  # Assigns an offset to each replica such that no two replicas overlap.
  replica_offsets = np.full([num_replicas, topology_rank], -1, dtype=np.int32)

  # TODO(ylc): Revisit here when topology_rank > 3.
  enable_2d_tiling = (
      topology_rank == 3 and
      computation_shape[-1] == 2  # Only handle 2D case.
      and np.prod(computation_stride) == 1  # Ensure no stride.
      and num_replicas == max_replicas)  # Full replication.
  logging.info("enable_2d_tiling: {}".format(enable_2d_tiling))
  if enable_2d_tiling:
    assignment = []
    inner_ring = _ring_2d(computation_shape[0], computation_shape[1])
    outer_ring = _ring_2d(replica_shape[0], replica_shape[1])

    for replica in xrange(num_replicas):
      outer_x, outer_y = outer_ring[replica]
      per_replica_assignment = []
      for index in xrange(np.prod(computation_shape)):
        inner_x, inner_y = inner_ring[index // 2]
        px = outer_x * computation_shape[0] + inner_x
        py = outer_y * computation_shape[1] + inner_y
        pz = index % 2
        per_replica_assignment.append([px, py, pz])
      assignment.append(per_replica_assignment)
  else:
    for replica in xrange(num_replicas):
      # Chooses a replica number in each axis.
      t = replica
      pos = []
      for dim in replica_shape[::-1]:
        pos.append(t % dim)
        t //= dim
      replica_pos = np.array(pos[::-1], dtype=np.int32)

      # Determines where that replica starts in each axis.
      outer = replica_pos // computation_stride
      inner = replica_pos % computation_stride
      replica_offsets[replica, :] = outer * computation_footprint + inner

    # Computes a logical core -> physical core mapping for each replica.
    indices = [
        np.arange(0, computation_shape[i] * computation_stride[i],
                  computation_stride[i]) for i in xrange(topology_rank)
    ]
    indices = np.concatenate(
        [i[..., np.newaxis] for i in np.meshgrid(*indices, indexing="ij")],
        axis=-1)
    indices = indices.reshape((-1, topology_rank))
    assignment = indices + replica_offsets[:, np.newaxis, :]
else:
  # We have a slice with missing chips. We define a simple assignment by
  # ignoring computation stride. This assignment should enable a consistent
  # and correct device assignment on degraded slices. It is optimal when
  # weights are not sharded. But this device assignment may be sub-optimal for
  # other model parallelism scenarios.
  assert np.prod(computation_stride) == 1
  # Next, we check if we have sufficient devices.
  assert num_replicas * np.prod(computation_shape) <= topology.num_tasks * topology.num_tpus_per_task
  # Map replicas to physical devices in task order.
  device_coordinates = topology.device_coordinates
  assignment = []
  devices_per_replica = np.prod(computation_shape)
  for rindex in xrange(num_replicas):
    replica_assignment = []
    for index in xrange(devices_per_replica):
      logical_id = rindex * devices_per_replica + index
      # Pick logical cores in task order
      task = logical_id // topology.num_tpus_per_task
      device = logical_id % topology.num_tpus_per_task
      # Append physical cores to the replica assignment
      replica_assignment.append(device_coordinates[task, device, :])
    assignment.append(replica_assignment)



# replicating computations to specific groups of TPU cores:


# in terminal #1:

ids = [3, 7]
op = lambda x: tpu_ops.tpu_ops.infeed_dequeue(tf.float32, shape=(1,))
op2 = lambda x: tpu_ops.tpu_ops.collective_permute(op(x), [[0, 1], [1, 0]])
#op2 = lambda x: tpu_ops.tpu_ops.collective_permute(x, [[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]])

zz = tpu_ops.shard(op2, outputs_from_all_shards=True, num_shards=len(ids), inputs=[[tf.constant([x], dtype=tf.float32) for x in ids]], device_assignment=get_core_assignment(*ids))
sess.run(zz)


# in infeed terminal:
ids = [3, 7]
ops = []
for i in ids:
  ops.append(tpu_ops.tpu_ops.infeed_enqueue([tf.constant(i, tf.float32)], shape=[1], device_ordinal=i));


sess.run(ops)




def device_for_host(task=0, cpu=0, job="worker"):
  #return "/job:%s/task:%d/device:CPU:%d" % (job, task, cpu)
  return "/job:%s/replica:0/task:%d/device:CPU:%d" % (job, task, cpu)


def get_host_device_ids(device_assignment, job="worker"):
  host_device_ids = set()
  for replica_id in range(device_assignment.num_replicas):
    host_device = device_assignment.host_device(replica=replica_id, logical_core=0, job=job)
    # TODO(lehou): Get host_id in a better way.
    host_id = int(host_device.split('/task:')[1].split('/device:')[0])
    host_device_ids.add(host_id)
  return host_device_ids


def device_mapping(device_assignment, job='worker'):
  for replica in range(device_assignment.num_replicas):
    #for logical_core in range(device_assignment.num_cores_per_replica):
    for logical_core in range(device_assignment.num_cores_per_replica):
      #with tf.device(device_for_host(replica)):
      host_device = device_assignment.host_device(replica=replica, logical_core=logical_core, job=job)
      device_ordinal = device_assignment.tpu_ordinal(replica=replica, logical_core=logical_core)
      yield host_device, device_ordinal, logical_core, replica


def host_mapping(device_assignment, job='worker'):
  for host_device, device_ordinal, logical_core, replica in device_mapping(device_assignment, job=job):
    if logical_core == 0:
      yield host_device, device_ordinal, replica


# in terminal #1:



def prn(x): print(x); return x


def device_for_tpu_core(task=0, core=0, job="worker"):
  return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job, task, core)

#import tflex_tpu_device_assignment; import tflex_tpu_topology; topology = tflex_tpu_topology.get_topology(res); dev = tflex_tpu_device_assignment.device_assignment(tflex_tpu_topology.get_topology(res), [8,8,1], [1,1,1], 2)
import tflex_tpu_device_assignment; import tflex_tpu_topology; topology = tflex_tpu_topology.get_topology(res); dev = tflex_tpu_device_assignment.spatial_partition(topology, 4)


def alloc_op(gb=None):
  if gb is None:
    gb = 7 * dev.num_cores_per_replica
  def op():
    #with tf.device(dev.tpu_device(replica=0, job='worker')):
    vs = []
    rs = []
    num_cores = dev.num_cores_per_replica
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      for i in range(gb):
        core = i % num_cores
        with tf.device(device_for_tpu_core(core=core)):
          v = tf.get_local_variable('alloc_%dgb_%d' % (gb, i), shape=(16,1024,128,128), dtype=tf.float32, initializer=tf.ones_initializer())
          vs.append(v)
          r = tf.reduce_sum(v)
          rs.append(r)
      with tf.device(device_for_tpu_core()):
        #return tpu_ops.tpu_ops.infeed_dequeue(tf.float32, shape=(1,))
        #return tf.reduce_sum(v)
        #r = tf.add_n([tf.reduce_sum(v) for v in vs])
        r = tf.add_n(rs)
        print(r)
        return r
  return op

import time

zz = tpu_ops.shard(alloc_op(), outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[], device_assignment=dev)
#sess.run(tf.tpu.initialize_system())
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()]); 
now = time.time(); qq = sess.run(zz); elapsed = time.time() - now; print(qq); print(elapsed, 'seconds')


gb = 8

with tf.device(dev.tpu_device(0, 0)), tf.variable_scope('', reuse=tf.AUTO_REUSE):
  v = tf.get_local_variable('alloc_%dgb' % gb, shape=((gb*1024)//4,1024,1024), dtype=tf.float32, initializer=tf.ones_initializer())
  op = tf.reduce_sum(v)

sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()]); 

with tf.device(dev.host_device(replica=0, job='worker')): zz = tpu_ops.shard(alloc_op(), outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[], device_assignment=dev); qq = sess.run(zz); print(qq)

with tf.device(dev.host_device(replica=0, job='worker')): zz = tpu_ops.shard(alloc_op(), outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[], device_assignment=dev); qq = sess.run(zz); print(qq)



def op():
  #with tf.device(dev.tpu_device(replica=0, job='worker')):
  with tf.device(device_for_tpu_core()):
    return tpu_ops.tpu_ops.infeed_dequeue(tf.float32, shape=(1,))


#op2 = lambda x: tpu_ops.tpu_ops.collective_permute(op(x), [[0, 1], [1, 0]])
#op2 = lambda x: tpu_ops.tpu_ops.collective_permute(x, [[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]])


#with tf.device(device_for_host()):
with tf.device(dev.host_device(replica=0, job='worker')): zz = tpu_ops.shard(op, outputs_from_all_shards=True, num_shards=dev.num_replicas, inputs=[], device_assignment=dev); qq = sess.run(zz); print(qq)


# in infeed terminal:
ops = []
for replica, logical_core, host_device, device_ordinal in device_mapping(dev):
  if logical_core == 0:
    print(replica, host_device, device_ordinal)
    with tf.device(host_device):
      ops.append(tpu_ops.tpu_ops.infeed_enqueue([tf.constant(replica, tf.float32)], shape=[1], device_ordinal=device_ordinal));

sess.run(ops)













# Apply function (increments x_i) on elements for which a certain condition
# apply (x_i != -1 in this example).
x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
condition_mask=tf.not_equal(x,tf.constant(-1.))
partitioned_data = tf.dynamic_partition(x, tf.cast(condition_mask, tf.int32) , 2)
partitioned_data[1] = partitioned_data[1] + 1.0
condition_indices = tf.dynamic_partition(tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
x = tf.dynamic_stitch(condition_indices, partitioned_data)
# Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
# unchanged.




tf.raw_ops.EmptyTensorList(element_shape=(1,), max_num_elements=1, element_dtype=tf.int32)



@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.math.log(1 + e), grad


# x = tf.constant(100.)
# y = log1pexp(x)
# dy = tf.gradients(y, x) # Will be NaN when evaluated.










# simple example of outfeed enqueue/dequeue

enq = tpu.shard(lambda: tpu_ops.outfeed_enqueue_tuple([tf.constant([1], dtype=tf.int32)]), num_shards=8)

deq = [tpu_ops.outfeed_dequeue_tuple(dtypes=[tf.int32], shapes=[[1]], device_ordinal=i) for i in range(8)]



# pipeline



def tf_id():
  # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
  replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
  return replica_id

def tf_tpu_cpu(f, *args, **kws):
  return tpu.outside_compilation(f, *args, **kws)

def tf_now():
  return tf_tpu_cpu(lambda: tf.timestamp())

def tf_get(k):
  return tf_tpu_cpu(lambda: table.lookup(k))

#def tf_reenqueue(

def testify(x):
  def testifying(*args):
    if len(args) == 0:
      return x
    else:
      y = args[0]
      if x == y:
        return x
  return testifying

def count_tpu_cores(session=None):
  session = session or tf.get_default_session()
  return len([x for x in session.list_devices() if ':TPU:' in x.name])

from tensorflow.python.tpu import tpu
from tensorflow.python.tpu.ops import tpu_ops

def count_replica_cores():
  return 8

def tf_now():
  return tf.cast(tf_tpu_cpu(lambda: tf.identity(tf.timestamp(), name="timestamp")), tf.float32)

def tf_elapsed(since):
  result = tf_tpu_cpu(lambda x: tf.identity(tf.timestamp() - tf.cast(x, tf.float64), name="timestamp"),
      since)
  return tf.cast(result, tf.tloat32)

def tf_replica_id():
  # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
  return tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)

def shapesof(values, shapes=None):
  return shapes or [x.shape.as_list() for x in values]

# def tf_infeed_enqueue_tuple_from_core(values, shapes=None):
#   shapes = shapesof(values, shapes)
#   replica_id = tf_replica_id()
#   def on_cpu(core_id, values, shapes):
#     def case(i):
#       return lambda: tpu_ops.infeed_enqueue_tuple(values, shapes, device_ordinal=i) 
#     return tf.switch_case(core_id, [case(i) for i in range(count_replica_cores())])
#   return tpu.outside_compilation(on_cpu, replica_id, values, shapes)

def tf_infeed_enqueue_tuple_from_core(values, shapes=None):
  shapes = shapesof(values, shapes)
  replica_id = tf_replica_id()
  def on_cpu(core_id, values, shapes):
    def case(i):
      return lambda: tpu_ops.infeed_enqueue_tuple(values, shapes, device_ordinal=i) 
    return tf.switch_case(core_id, [case(i) for i in range(count_replica_cores())])
  return tpu.outside_compilation(on_cpu, replica_id, values, shapes)

def tpu_shard(op, num_shards=None, **kws):
  if num_shards is None:
    num_shards = count_replica_cores()
  return tpu.shard(op, num_shards=num_shards, **kws)

def on_core():
  start_time, i, token = tpu_ops.infeed_dequeue_tuple([tf.float32, tf.int64, tf.int64], shapes=[(1,), (1,), (1,)])
  now = tf_now()
  elapsed = now - start_time
  with tf.control_dependencies([tpu_ops.outfeed_enqueue_tuple([start_time, i, token])]):
    result = tf.identity(token * token)
    values = [now, i, result]
    reup = tf_infeed_enqueue_tuple_from_core(values, shapes=[(1,), (1,), (1,)])
    with tf.control_dependencies([reup]):
      return [i, replica_id, result]

def on_core():
  when, i, value = tpu_ops.infeed_dequeue_tuple([tf.float32, tf.int64, tf.int64], shapes=[(1,), (1,), (1,)])
  now = tf_now()
  elapsed = now - when
  values = [now, i + 1, value * value]
  reup = tf_infeed_enqueue_tuple_from_core(values, shapes=[(1,), (1,), (1,)])


# def tf_now():
#   return tf.cast(tf_tpu_cpu(lambda: tf.identity(tf.timestamp(), name="timestamp")), tf.float32)

# def tf_elapsed(since):
#   result = tf_tpu_cpu(lambda x: tf.identity(tf.timestamp() - tf.cast(x, tf.float64), name="timestamp"),
#       since)
#   return tf.cast(result, tf.tloat32)


# somehow this makes a persistent clock.

def kickstart():
  ops = [tpu_ops.infeed_enqueue_tuple([tf_now()], shapes=[()], device_ordinal=i) for i in range(count_replica_cores())]
  return ops

def on_core():
  start_time, = tpu_ops.infeed_dequeue_tuple([tf.float32], shapes=[(1,)])
  with tf.control_dependencies([start_time]):
    # elapsed = tf_elapsed(start_time)
    now = tf.cast(tf_now(), tf.float32)
    # #refeed = tf_infeed_enqueue_tuple_from_core([now])
    # # with tf.control_dependencies([refeed]):
    # #   return [elapsed, now]
    #return [elapsed, now]
    return [start_time, now]

core_op = tpu_shard(on_core)

def dequeue():
  return tpu_ops.infeed_dequeue_tuple([tf.float32], shapes=[(1,)])

deq = tpu_shard(dequeue)

kick = kickstart()


from tensorflow.python.ops import data_flow_ops

# fails
s = data_flow_ops.StagingArea([tf.float64])
def on_core():
  return s.get()

# fails
def on_core():
  s = data_flow_ops.StagingArea([tf.float64])
  return s.get()


s = data_flow_ops.StagingArea([tf.float64])

def on_core():
  v = tf_tpu_cpu(lambda: globals()['s'].get())
  return v

core_op = tpu_shard(on_core)



def feed_cores(dtype, shape=[]):
  num_cores = count_replica_cores()
  values_IN = tf.placeholder(dtype, [num_cores] + shape, name='infeed_enqueue_values_IN')
  ops = []
  for i in range(num_cores):
    index = tf.constant(i, tf.int64)
    #value = tf.cast(values_IN[i], tf.int32)
    value = values_IN[i]
    shapes = [index.shape.as_list(), value.shape.as_list()]
    values = [index, value]
    op = tpu_ops.infeed_enqueue_tuple(values, shapes=shapes, device_ordinal=i)
    ops.append(op)
  return values_IN, ops

values_IN, feed_op = feed_cores(tf.int32)

sess.run(feed_op, {values_IN: [i for i in range(count_replica_cores())]})

# def feed_cores():
#   num_cores = count_tpu_cores()
#   #assert isinstance(values, (list, tuple))
#   #assert len(values) % num_cores == 0
#   values_IN = tf.placeholder(tf.int32, [num_cores], name='infeed_enqueue_values_IN')
#   op = [tpu_ops.infeed_enqueue_tuple([tf.constant(i, tf.int64), tf.cast(values_IN[i], tf.int32)], shapes=[(), ()], device_ordinal=i) for i in range(num_cores)]
#   return values_IN, op

# values_IN, feed_op = feed_cores()


# def run_cores(on_core_fn);
#   op = tpu_shard(on_core_fn)
#   stop = False
#   def core_thunk():
#     while not stop:
#       sess.run(op)



import threading

# threading.Thread(target=



enq = tpu.shard(lambda: tpu_ops.outfeed_enqueue_tuple([tf.constant([1], dtype=tf.int32)]), num_shards=8)

deq = [tpu_ops.outfeed_dequeue_tuple(dtypes=[tf.int32], shapes=[[1]], device_ordinal=i) for i in range(8)]



# How to get a TPU's physical ID

from tensorflow.python.framework import errors_impl

def clone_session(session=None, graph=None, interactive=False, **kws):
  if session is None:
    session = tf.get_default_session()
  if graph is None:
    graph = session.graph
  config = session._config # is there a better way to do this?
  master = session.sess_str # is there a better way to do this?
  Session = (tf.compat.v1.InteractiveSession if interactive else tf.Session)
  return Session(master, graph=graph, config=config)

def get_tpu_id(session=None):
  if session is None:
    session = tf.get_default_session()
  try:
    with tf.Graph().as_default() as graph, clone_session(graph=graph) as throwaway_session:
      throwaway_session.run(gen_memory_stats_ops.bytes_in_use())
  except errors_impl.NotFoundError as e:
    return e.message.split(' in binary running on ')[-1].split('. ')[0]


from tensorflow.python.framework import device as pydev



class FakeOp(object):
  """A helper class to determine the current device.

  Supports only the type and device set/get methods needed to run the
  graph's _apply_device_function method.
  """

  def __init__(self):
    self._device = ""

  @property
  def type(self):
    return "FakeOp"

  @property
  def device(self):
    return self._device

  def _set_device(self, device):
    if isinstance(device, pydev.DeviceSpec):
      self._device = device.to_string()
    else:
      self._device = device

  def _set_device_from_string(self, device_str):
    self._device = device_str


def tf_determine_current_device_op(graph=None):
  if graph is None:
    graph = tf.get_default_graph()
  fake_op = FakeOp()
  graph._apply_device_functions(fake_op)  # pylint: disable=protected-access
  device = pydev.DeviceSpec.from_string(fake_op.device)
  
