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
ops = []
for i in [3, 7]:
  ops.append(tpu_ops.tpu_ops.infeed_enqueue([tf.constant(i, tf.float32)], shape=[1], device_ordinal=i));


sess.run(ops)
