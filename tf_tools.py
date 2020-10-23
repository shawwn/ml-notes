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
  return tpu.outside_compilation(f, *args, **kws)

def tpu_now():
  return tpu_cpu(lambda: tf.identity(tf.timestamp(), name="timestamp"))

def tpu_get(k):
  return tpu_cpu(lambda: table.lookup(k))



def enq(*values, name):
  return tft.tpu_cpu(lambda vs: tf.raw_ops.Stage(values=vs, container=name, shared_name=name), values)

def dtypes_of(xs):
  return [x.dtype if hasattr(x, 'dtype') else x for x in xs]

def deq(*dtypes, name):
  return tft.tpu_cpu(lambda vs: tf.raw_ops.Unstage(dtypes=dtypes_of(dtypes), container=name, shared_name=name), values)



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



