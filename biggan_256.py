import numpy as np
import tensorflow as tf

import graph_to_code

tf1 = tf.compat.v1


def absolute_name_scope(scope):
    return tf1.name_scope(scope + "/")


def absolute_variable_scope(scope, reuse=None, **kwargs):
    return tf1.variable_scope(tf1.VariableScope(name=scope, reuse=reuse, **kwargs), auxiliary_name_scope=False)


def absolute_scope(scope):
  return absolute_name_scope(scope), absolute_variable_scope(scope)


def assert_shape(lhs, shape):
  x = lhs
  if hasattr(lhs, 'shape'):
    lhs = lhs.shape
  if hasattr(lhs, 'as_list'):
    lhs = lhs.as_list()
  assert lhs == shape
  return x


def shapelist(x):
  if hasattr(x, 'shape'):
    x = x.shape
  return x.as_list()


def globalvar(name, **kws):
  shape = kws.pop('shape')
  initializer = kws.pop('initializer', None)
  if initializer is None:
    initializer = tf.initializers.zeros
  collections = kws.pop('collections', ['variables'])
  trainable = kws.pop('trainable', True)
  use_resource = kws.pop('use_resource', True)
  dtype = kws.pop('dtype', tf.float32)
  return tf1.get_variable(name, dtype=dtype, initializer=initializer, shape=shape, collections=collections, use_resource=use_resource, trainable=trainable, **kws)


def localvar(name, **kws):
  collections = kws.pop('collections', ['local_variables'])
  trainable = kws.pop('trainable', False)
  use_resource = kws.pop('use_resource', True)
  return globalvar(name, **kws, collections=collections, trainable=trainable, use_resource=use_resource)


def specnorm(w, epsilon=9.999999747378752e-05, return_norm=False):
  assert len(shapelist(w)) > 1
  shape = shapelist(w)
  ushape = [1, shape[-1]]
  w_reshaped = tf.reshape(w, [-1, shape[-1]])
  wshape = shapelist(w_reshaped)
  u0 = localvar('u0', dtype=w.dtype, shape=ushape, initializer=tf1.truncated_normal_initializer(mean=0.0, stddev=1.0), collections=['variables'])
  u1 = localvar('u1', dtype=w.dtype, shape=ushape, initializer=tf1.truncated_normal_initializer(mean=0.0, stddev=1.0), collections=['variables'])
  u2 = localvar('u2', dtype=w.dtype, shape=ushape, initializer=tf1.truncated_normal_initializer(mean=0.0, stddev=1.0), collections=['variables'])
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>]}],
  x0 = tf.matmul(u0, w_reshaped, transpose_b=True)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow/y:0' shape=() dtype=float32>]},
  #     {'value': 2.0}],
  # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow' type=Pow>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow:0' shape=(1, 20) dtype=float32>]}],
  x1 = tf.pow(x0, 2.0)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const:0' shape=(2,) dtype=int32>]},
  #     {'value': array([0, 1], dtype=int32)}],
  # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum' type=Sum>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const:0' shape=(2,) dtype=int32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum:0' shape=(1, 1) dtype=float32>]}],
  x2 = tf.reduce_sum(x1, axis=[0, 1], keepdims=True)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add/y:0' shape=() dtype=float32>]},
  #     {'value': 9.999999747378752e-05}],
  # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add' type=Add>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add:0' shape=(1, 1) dtype=float32>]}],
  x3 = tf.add(x2, epsilon)
  # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt' type=Rsqrt>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt:0' shape=(1, 1) dtype=float32>]}],
  x4 = tf.math.rsqrt(x3)
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>]}],
  x5 = tf.multiply(x0, x4)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_1' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>]}],
  x6 = tf.matmul(x5, w_reshaped)
  assert_shape(x6, ushape)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_1/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1/y:0' shape=() dtype=float32>]},
  #     {'value': 2.0}],
  # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_1' type=Pow>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1:0' shape=(1, 24576) dtype=float32>]}],
  x7 = tf.pow(x6, 2.0)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_1' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_1:0' shape=(2,) dtype=int32>]},
  #     {'value': array([0, 1], dtype=int32)}],
  # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_1' type=Sum>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_1:0' shape=(2,) dtype=int32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_1:0' shape=(1, 1) dtype=float32>]}],
  x8 = tf.reduce_sum(x7, axis=[0, 1], keepdims=True)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_1/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1/y:0' shape=() dtype=float32>]},
  #     {'value': 9.999999747378752e-05}],
  # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_1' type=Add>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_1:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1:0' shape=(1, 1) dtype=float32>]}],
  x9 = tf.add(x8, epsilon)
  # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_1' type=Rsqrt>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_1:0' shape=(1, 1) dtype=float32>]}],
  x10 = tf.math.rsqrt(x9)
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_1' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_1:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>]}],
  x11 = tf.multiply(x6, x10)
  # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1' type=ReadVariableOp>,
  #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1:0' shape=(1, 24576) dtype=float32>]}],
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_2' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>]}],
  x12 = tf.matmul(u1, w_reshaped, transpose_b=True)
  assert_shape(x12, [1, 20])
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_3' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_3:0' shape=(1, 1) dtype=float32>]}],
  x13 = tf.matmul(x5, x12, transpose_b=True)
  assert_shape(x13, [1, 1])
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_2' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_3:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_2:0' shape=(1, 20) dtype=float32>]}],
  x14 = tf.multiply(x13, x5)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_4' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_4:0' shape=(1, 1) dtype=float32>]}],
  x15 = tf.matmul(x5, x5, transpose_b=True)
  assert_shape(x15, [1, 1])
  # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv' type=RealDiv>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_2:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_4:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv:0' shape=(1, 20) dtype=float32>]}],
  x16 = tf.div(x14, x15)
  # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub' type=Sub>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>]}],
  x17 = tf.subtract(x12, x16)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_2/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2/y:0' shape=() dtype=float32>]},
  #     {'value': 2.0}],
  # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_2' type=Pow>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2:0' shape=(1, 20) dtype=float32>]}],
  x18 = tf.pow(x17, 2.0)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_2' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_2:0' shape=(2,) dtype=int32>]},
  #     {'value': array([0, 1], dtype=int32)}],
  # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_2' type=Sum>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_2:0' shape=(2,) dtype=int32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_2:0' shape=(1, 1) dtype=float32>]}],
  x19 = tf.reduce_sum(x18, axis=[0, 1], keepdims=True)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_2/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2/y:0' shape=() dtype=float32>]},
  #     {'value': 9.999999747378752e-05}],
  # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_2' type=Add>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_2:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2:0' shape=(1, 1) dtype=float32>]}],
  x20 = tf.add(x19, epsilon)
  # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_2' type=Rsqrt>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_2:0' shape=(1, 1) dtype=float32>]}],
  x21 = tf.math.rsqrt(x20)
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_3' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_2:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>]}],
  x22 = tf.multiply(x17, x21)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_5' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>]}],
  x23 = tf.matmul(x22, w_reshaped)
  assert_shape(x23, ushape)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_6' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_6:0' shape=(1, 1) dtype=float32>]}],
  x24 = tf.matmul(x11, x23, transpose_b=True)
  assert_shape(x24, [1, 1])
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_4' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_6:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_4:0' shape=(1, 24576) dtype=float32>]}],
  x25 = tf.multiply(x24, x11)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_7' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_7:0' shape=(1, 1) dtype=float32>]}],
  x26 = tf.matmul(x11, x11, transpose_b=True)
  assert_shape(x26, [1, 1])
  # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_1' type=RealDiv>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_4:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_7:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_1:0' shape=(1, 24576) dtype=float32>]}],
  x27 = tf.div(x25, x26)
  # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_1' type=Sub>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_1:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>]}],
  x28 = tf.subtract(x23, x27)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_3/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3/y:0' shape=() dtype=float32>]},
  #     {'value': 2.0}],
  # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_3' type=Pow>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3:0' shape=(1, 24576) dtype=float32>]}],
  x29 = tf.pow(x28, 2.0)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_3' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_3:0' shape=(2,) dtype=int32>]},
  #     {'value': array([0, 1], dtype=int32)}],
  # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_3' type=Sum>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_3:0' shape=(2,) dtype=int32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_3:0' shape=(1, 1) dtype=float32>]}],
  x30 = tf.reduce_sum(x29, axis=[0, 1], keepdims=True)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_3/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3/y:0' shape=() dtype=float32>]},
  #     {'value': 9.999999747378752e-05}],
  # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_3' type=Add>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_3:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3:0' shape=(1, 1) dtype=float32>]}],
  x31 = tf.add(x30, epsilon)
  # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_3' type=Rsqrt>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_3:0' shape=(1, 1) dtype=float32>]}],
  x32 = tf.math.rsqrt(x31)
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_5' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_3:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>]}],
  x33 = tf.multiply(x28, x32)
  # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2' type=ReadVariableOp>,
  #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2:0' shape=(1, 24576) dtype=float32>]}],
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_8' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>]}],
  x34 = tf.matmul(u2, w_reshaped, transpose_b=True)
  assert_shape(x34, [1, 20])
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_9' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_9:0' shape=(1, 1) dtype=float32>]}],
  x35 = tf.matmul(x5, x34, transpose_b=True)
  assert_shape(x35, [1, 1])
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_6' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_9:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_6:0' shape=(1, 20) dtype=float32>]}],
  x36 = tf.multiply(x35, x5)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_10' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_10:0' shape=(1, 1) dtype=float32>]}],
  x37 = tf.matmul(x5, x5, transpose_b=True)
  assert_shape(x37, [1, 1])
  # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_2' type=RealDiv>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_6:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_10:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_2:0' shape=(1, 20) dtype=float32>]}],
  x38 = tf.div(x36, x37)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_11' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_11:0' shape=(1, 1) dtype=float32>]}],
  x39 = tf.matmul(x22, x34, transpose_b=True)
  assert_shape(x39, [1, 1])
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_7' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_11:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_7:0' shape=(1, 20) dtype=float32>]}],
  x40 = tf.multiply(x39, x22)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_12' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_12:0' shape=(1, 1) dtype=float32>]}],
  x41 = tf.matmul(x22, x22, transpose_b=True)
  assert_shape(x41, [1, 1])
  # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_3' type=RealDiv>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_7:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_12:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_3:0' shape=(1, 20) dtype=float32>]}],
  x42 = tf.div(x40, x41)
  # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_4' type=Add>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_2:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_3:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_4:0' shape=(1, 20) dtype=float32>]}],
  x43 = tf.add(x38, x42)
  # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_2' type=Sub>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_4:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>]}],
  x44 = tf.subtract(x34, x43)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_4/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4/y:0' shape=() dtype=float32>]},
  #     {'value': 2.0}],
  # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_4' type=Pow>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4:0' shape=(1, 20) dtype=float32>]}],
  x45 = tf.pow(x44, 2.0)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_4' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_4:0' shape=(2,) dtype=int32>]},
  #     {'value': array([0, 1], dtype=int32)}],
  # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_4' type=Sum>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_4:0' shape=(2,) dtype=int32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_4:0' shape=(1, 1) dtype=float32>]}],
  x46 = tf.reduce_sum(x45, axis=[0, 1], keepdims=True)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_5/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5/y:0' shape=() dtype=float32>]},
  #     {'value': 9.999999747378752e-05}],
  # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_5' type=Add>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_4:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5:0' shape=(1, 1) dtype=float32>]}],
  x47 = tf.add(x46, epsilon)
  # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_4' type=Rsqrt>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_4:0' shape=(1, 1) dtype=float32>]}],
  x48 = tf.math.rsqrt(x47)
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_8' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_4:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>]}],
  x49 = tf.multiply(x44, x48)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_13' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>]}],
  x50 = tf.matmul(x49, w_reshaped)
  assert_shape(x50, ushape)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_14' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_14:0' shape=(1, 1) dtype=float32>]}],
  x51 = tf.matmul(x11, x50, transpose_b=True)
  assert_shape(x51, [1, 1])
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_9' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_14:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_9:0' shape=(1, 24576) dtype=float32>]}],
  x52 = tf.multiply(x51, x11)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_15' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_15:0' shape=(1, 1) dtype=float32>]}],
  x53 = tf.matmul(x11, x11, transpose_b=True)
  assert_shape(x53, [1, 1])
  # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_4' type=RealDiv>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_9:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_15:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_4:0' shape=(1, 24576) dtype=float32>]}],
  x54 = tf.div(x52, x53)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_16' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_16:0' shape=(1, 1) dtype=float32>]}],
  x55 = tf.matmul(x33, x50, transpose_b=True)
  assert_shape(x55, [1, 1])
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_10' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_16:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_10:0' shape=(1, 24576) dtype=float32>]}],
  x56 = tf.multiply(x55, x33)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_17' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_17:0' shape=(1, 1) dtype=float32>]}],
  x57 = tf.matmul(x33, x33, transpose_b=True)
  assert_shape(x57, [1, 1])
  # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_5' type=RealDiv>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_10:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_17:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_5:0' shape=(1, 24576) dtype=float32>]}],
  x58 = tf.div(x56, x57)
  # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_6' type=Add>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_4:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_5:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_6:0' shape=(1, 24576) dtype=float32>]}],
  x59 = tf.add(x54, x58)
  # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_3' type=Sub>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_6:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>]}],
  x60 = tf.subtract(x50, x59)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_5/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5/y:0' shape=() dtype=float32>]},
  #     {'value': 2.0}],
  # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_5' type=Pow>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5:0' shape=(1, 24576) dtype=float32>]}],
  x61 = tf.pow(x60, 2.0)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_5' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_5:0' shape=(2,) dtype=int32>]},
  #     {'value': array([0, 1], dtype=int32)}],
  # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_5' type=Sum>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_5:0' shape=(2,) dtype=int32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_5:0' shape=(1, 1) dtype=float32>]}],
  x62 = tf.reduce_sum(x61, axis=[0, 1], keepdims=True)
  # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_7/y' type=Const>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7/y:0' shape=() dtype=float32>]},
  #     {'value': 9.999999747378752e-05}],
  # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_7' type=Add>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_5:0' shape=(1, 1) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7/y:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7:0' shape=(1, 1) dtype=float32>]}],
  x63 = tf.add(x62, epsilon)
  # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_5' type=Rsqrt>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_5:0' shape=(1, 1) dtype=float32>]}],
  x64 = tf.math.rsqrt(x63)
  # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_11' type=Mul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_5:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>]}],
  x65 = tf.multiply(x60, x64)
  # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient' type=StopGradient>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>]}],
  x66 = tf.stop_gradient(x11)
  # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient_1' type=StopGradient>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_1:0' shape=(1, 20) dtype=float32>]}],
  x67 = tf.stop_gradient(x5)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_18' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_1:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_18:0' shape=(1, 24576) dtype=float32>]}],
  x68 = tf.matmul(x67, w_reshaped)
  assert_shape(x68, ushape)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_19' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_18:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_19:0' shape=(1, 1) dtype=float32>]}],
  x69nice = tf.matmul(x68, x66, transpose_b=True)
  assert_shape(x69nice, [1, 1])
  # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze' type=Squeeze>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_19:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze:0' shape=() dtype=float32>]}],
  x70 = tf.squeeze(x69nice)
  assert_shape(x70, [])
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_20' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_20:0' shape=(1, 24576) dtype=float32>]}],
  x71 = tf.matmul(x22, w_reshaped)
  assert_shape(x71, ushape)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_21' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_20:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_21:0' shape=(1, 1) dtype=float32>]}],
  x72 = tf.matmul(x71, x33, transpose_b=True)
  assert_shape(x72, [1, 1])
  # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze_1' type=Squeeze>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_21:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze_1:0' shape=() dtype=float32>]}],
  x73 = tf.squeeze(x72)
  assert_shape(x73, [])
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_22' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_22:0' shape=(1, 24576) dtype=float32>]}],
  x74 = tf.matmul(x49, w_reshaped)
  assert_shape(x74, ushape)
  # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_23' type=MatMul>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_22:0' shape=(1, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_23:0' shape=(1, 1) dtype=float32>]}],
  x75 = tf.matmul(x74, x65, transpose_b=True)
  assert_shape(x75, [1, 1])
  # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze_2' type=Squeeze>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_23:0' shape=(1, 1) dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze_2:0' shape=() dtype=float32>]}],
  x76 = tf.squeeze(x75)
  assert_shape(x76, [])
  with tf.name_scope('norm'):
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/mul' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/mul:0' shape=(20, 24576) dtype=float32>]}],
    x77 = tf.multiply(w_reshaped, w_reshaped)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Const' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Const:0' shape=(2,) dtype=int32>]},
    #     {'value': array([0, 1], dtype=int32)}],
    # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Sum' type=Sum>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/mul:0' shape=(20, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Const:0' shape=(2,) dtype=int32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sum:0' shape=(1, 1) dtype=float32>]}],
    x78 = tf.reduce_sum(x77, axis=[0, 1], keepdims=True)
    # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Sqrt' type=Sqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sum:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sqrt:0' shape=(1, 1) dtype=float32>]}],
    x79 = tf.math.sqrt(x78)
    # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Squeeze' type=Squeeze>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sqrt:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Squeeze:0' shape=() dtype=float32>]}],
    x80 = tf.squeeze(x79)
    assert_shape(x80, [])
  # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sqrt' type=Sqrt>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Squeeze:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sqrt:0' shape=() dtype=float32>]}],
  x81 = tf.math.sqrt(x80)
  # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient_2' type=StopGradient>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sqrt:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_2:0' shape=() dtype=float32>]}],
  x82 = tf.stop_gradient(x81)
  with tf.name_scope('norm'):
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/mul' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/mul:0' shape=(20, 24576) dtype=float32>]}],
    x83 = tf.multiply(w_reshaped, w_reshaped)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Const' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Const:0' shape=(2,) dtype=int32>]},
    #     {'value': array([0, 1], dtype=int32)}],
    # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Sum' type=Sum>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/mul:0' shape=(20, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Const:0' shape=(2,) dtype=int32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sum:0' shape=(1, 1) dtype=float32>]}],
    x84 = tf.reduce_sum(x83, axis=[0, 1], keepdims=True)
    # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt' type=Sqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sum:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt:0' shape=(1, 1) dtype=float32>]}],
    x85 = tf.math.sqrt(x84)
    # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Squeeze' type=Squeeze>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Squeeze:0' shape=() dtype=float32>]}],
    x86 = tf.squeeze(x85)
  # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_6' type=RealDiv>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze:0' shape=() dtype=float32>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_6:0' shape=(20, 24576) dtype=float32>]}],
  x87 = tf.div(w_reshaped, x70)
  # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp' type=AssignVariableOp>,
  #     <tf.Tensor 'module/Generator/G_Z/G_linear/u0:0' shape=() dtype=resource>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': []}],
  x88 = u0.assign(x66, read_value=False)
  # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_3' type=ReadVariableOp>,
  #     <tf.Tensor 'module/Generator/G_Z/G_linear/u0:0' shape=() dtype=resource>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_3:0' shape=(1, 24576) dtype=float32>]}],
  with tf.control_dependencies([x88]):
    x89 = u0.read_value()
  # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp_1' type=AssignVariableOp>,
  #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': []}],
  x90 = u1.assign(x33, read_value=False)
  # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_4' type=ReadVariableOp>,
  #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_4:0' shape=(1, 24576) dtype=float32>]}],
  with tf.control_dependencies([x90]):
    x91 = u1.read_value()
  # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp_2' type=AssignVariableOp>,
  #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
  #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>,
  #     {'outputs': []}],
  x92 = u2.assign(x65, read_value=False)
  # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_5' type=ReadVariableOp>,
  #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
  #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_5:0' shape=(1, 24576) dtype=float32>]}],
  with tf.control_dependencies([x92]):
    x93 = u2.read_value()
  with tf.control_dependencies([x88, x90, x92]):
    norm = tf.identity(x70, "norm")
    w_normalized = tf.div(w_reshaped, norm)
    w_normalized = tf.reshape(w_normalized, shape)
    if return_norm:
      return w_normalized, norm
    return w_normalized


def make_biggan():
  with tf1.variable_scope('module', reuse=tf1.AUTO_REUSE):
    dummy_z = np.array([[-0.27682826, -0.58815104,  1.7036977 , -0.2206092 , -0.14080267,
          -1.4358327 , -0.29529673, -0.21998306,  2.3495033 , -0.2704561 ,
           0.67489153, -1.7079376 , -0.8530004 ,  0.47657555, -1.0244914 ,
          -0.5066494 ,  0.40413463, -2.650805  ,  0.20753726, -0.45942673,
           0.34236595,  0.78934395,  1.2019389 , -1.255674  , -0.07768833,
           0.7577431 , -0.38986343,  0.03649916,  1.328297  , -0.1437277 ,
           0.76792073, -1.2927496 ,  2.3878598 ,  0.6853071 ,  0.11304516,
          -0.9645209 , -2.4931862 , -0.12763861,  0.7414209 ,  0.6020558 ,
          -0.6050938 , -0.5639263 , -0.00988291, -1.5184546 , -0.38591796,
          -0.20601207,  0.18363006,  1.962519  ,  1.0850583 , -0.8571455 ,
          -0.01302644,  0.6277824 , -0.08292245,  0.92597395, -0.11542881,
           0.38168597, -2.4266438 , -1.566245  , -1.039471  , -0.940249  ,
           0.02636161,  0.06007156,  1.2789264 , -0.07752071,  0.44986045,
           0.41236845,  1.3643209 , -1.4008445 , -0.19861189,  0.46731564,
           0.06151719,  0.98628384, -0.20362067, -0.5842369 , -0.7696563 ,
           0.94691944,  2.3646383 , -1.1924875 ,  0.35439596,  1.2308508 ,
           0.17359956,  1.3657194 , -1.1731008 , -0.9649893 ,  0.87262   ,
          -0.3879596 ,  0.12370261,  0.9923666 , -1.6314132 , -2.173692  ,
           1.2991096 , -0.5108776 , -0.31982934, -1.463904  ,  0.00470991,
          -0.18117207, -0.04366804, -1.4812558 ,  1.1272283 ,  0.5390479 ,
          -0.03865089, -0.5393169 , -0.10081987,  0.69317263,  1.2149591 ,
           0.26094043,  0.71965116,  0.81613004,  1.4130529 ,  0.44307762,
          -0.2564097 , -0.06270383,  0.11339105,  1.2114154 ,  0.9871673 ,
          -0.67596656, -0.34136584, -0.40325257, -1.5253726 , -0.3829709 ,
          -1.3955748 ,  1.349158  ,  0.58127445, -0.8905083 ,  1.272159  ,
           0.8208986 , -0.5260699 , -1.075426  ,  0.29986796,  0.06508358,
           0.3826486 ,  1.5031533 ,  1.2863646 , -0.15485081, -0.06244287,
           1.1686682 ,  0.35917065,  2.2737215 ,  1.5198022 , -1.2142191 ]],
        dtype=np.float32)
    tf_dummy_z = tf.convert_to_tensor(dummy_z, name="dummy_z")
    dummy_y = np.zeros(shape=(1,1000), dtype=np.float32)
    dummy_y[0,0] = 1.0
    tf_dummy_y = tf.convert_to_tensor(dummy_y, name="dummy_y")
    with tf1.variable_scope('linear'):
      w = globalvar('w', shape=(1000, 128), initializer=tf1.truncated_normal_initializer(mean=0.0, stddev=0.03162277489900589))
      w = tf.matmul(tf_dummy_y, w)
    with tf1.variable_scope('Generator_1'):
      z_split = tf.split(tf_dummy_z, 7, axis=1)
      z_in = [tf.concat([z, w], axis=1) for z in z_split[1:]]
    with tf1.variable_scope('Generator'):
      with tf1.variable_scope('G_Z'):
        with tf1.variable_scope('G_linear'):
          w = globalvar('w', shape=(20, 24576), initializer=tf1.random_normal_initializer(mean=0.0, stddev=0.019999999552965164))
          w0, wnorm = specnorm(w, return_norm=True)
          w1 = tf.matmul(z_split[0], w0)
          assert_shape(w1, [1, 24576])
          b = globalvar('b', shape=(24576,))
          w2 = tf.add(w1, b)
          w3 = tf.reshape(w2, [-1, 4, 4, 1536])
          assert_shape(w3, [1, 4, 4, 1536])
          return wnorm
          # with absolute_name_scope('module/Generator_1/G_Z/G_linear'), absolute_variable_scope('module/Generator_1/G_Z/G_linear'):
          #   w = tf.reshape(w, [-1, 24576])
          #   w_reshaped = w
          # u0 = tf.get_variable('u0', dtype=tf.float32, shape=(1, 24576), initializer=tf1.truncated_normal_initializer(mean=0.0, stddev=1.0), use_resource=True)
          # u1 = tf.get_variable('u1', dtype=tf.float32, shape=(1, 24576), initializer=tf1.truncated_normal_initializer(mean=0.0, stddev=1.0), use_resource=True)
          # u2 = tf.get_variable('u2', dtype=tf.float32, shape=(1, 24576), initializer=tf1.truncated_normal_initializer(mean=0.0, stddev=1.0), use_resource=True)
          # with absolute_name_scope('module/Generator_1/G_Z/G_linear'), absolute_variable_scope('module/Generator_1/G_Z/G_linear'):
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>]}],
          #   x0 = tf.matmul(u0, w_reshaped, transpose_b=True)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow/y:0' shape=() dtype=float32>]},
          #   #     {'value': 2.0}],
          #   # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow' type=Pow>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow:0' shape=(1, 20) dtype=float32>]}],
          #   x1 = tf.pow(x0, 2.0)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const:0' shape=(2,) dtype=int32>]},
          #   #     {'value': array([0, 1], dtype=int32)}],
          #   # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum' type=Sum>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const:0' shape=(2,) dtype=int32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum:0' shape=(1, 1) dtype=float32>]}],
          #   x2 = tf.reduce_sum(x1, axis=[0, 1], keepdims=True)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add/y:0' shape=() dtype=float32>]},
          #   #     {'value': 9.999999747378752e-05}],
          #   # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add' type=Add>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add:0' shape=(1, 1) dtype=float32>]}],
          #   x3 = tf.add(x2, 9.999999747378752e-05)
          #   # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt' type=Rsqrt>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt:0' shape=(1, 1) dtype=float32>]}],
          #   x4 = tf.math.rsqrt(x3)
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>]}],
          #   x5 = tf.multiply(x0, x4)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_1' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>]}],
          #   x6 = tf.matmul(x5, w_reshaped)
          #   assert_shape(x6, [1, 24576])
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_1/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1/y:0' shape=() dtype=float32>]},
          #   #     {'value': 2.0}],
          #   # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_1' type=Pow>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1:0' shape=(1, 24576) dtype=float32>]}],
          #   x7 = tf.pow(x6, 2.0)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_1' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_1:0' shape=(2,) dtype=int32>]},
          #   #     {'value': array([0, 1], dtype=int32)}],
          #   # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_1' type=Sum>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_1:0' shape=(2,) dtype=int32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_1:0' shape=(1, 1) dtype=float32>]}],
          #   x8 = tf.reduce_sum(x7, axis=[0, 1], keepdims=True)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_1/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1/y:0' shape=() dtype=float32>]},
          #   #     {'value': 9.999999747378752e-05}],
          #   # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_1' type=Add>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_1:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1:0' shape=(1, 1) dtype=float32>]}],
          #   x9 = tf.add(x8, 9.999999747378752e-05)
          #   # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_1' type=Rsqrt>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_1:0' shape=(1, 1) dtype=float32>]}],
          #   x10 = tf.math.rsqrt(x9)
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_1' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_1:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>]}],
          #   x11 = tf.multiply(x6, x10)
          #   # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1' type=ReadVariableOp>,
          #   #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1:0' shape=(1, 24576) dtype=float32>]}],
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_2' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>]}],
          #   x12 = tf.matmul(u1, w_reshaped, transpose_b=True)
          #   assert_shape(x12, [1, 20])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_3' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_3:0' shape=(1, 1) dtype=float32>]}],
          #   x13 = tf.matmul(x5, x12, transpose_b=True)
          #   assert_shape(x13, [1, 1])
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_2' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_3:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_2:0' shape=(1, 20) dtype=float32>]}],
          #   x14 = tf.multiply(x13, x5)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_4' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_4:0' shape=(1, 1) dtype=float32>]}],
          #   x15 = tf.matmul(x5, x5, transpose_b=True)
          #   assert_shape(x15, [1, 1])
          #   # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv' type=RealDiv>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_2:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_4:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv:0' shape=(1, 20) dtype=float32>]}],
          #   x16 = tf.div(x14, x15)
          #   # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub' type=Sub>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>]}],
          #   x17 = tf.subtract(x12, x16)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_2/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2/y:0' shape=() dtype=float32>]},
          #   #     {'value': 2.0}],
          #   # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_2' type=Pow>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2:0' shape=(1, 20) dtype=float32>]}],
          #   x18 = tf.pow(x17, 2.0)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_2' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_2:0' shape=(2,) dtype=int32>]},
          #   #     {'value': array([0, 1], dtype=int32)}],
          #   # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_2' type=Sum>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_2:0' shape=(2,) dtype=int32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_2:0' shape=(1, 1) dtype=float32>]}],
          #   x19 = tf.reduce_sum(x18, axis=[0, 1], keepdims=True)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_2/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2/y:0' shape=() dtype=float32>]},
          #   #     {'value': 9.999999747378752e-05}],
          #   # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_2' type=Add>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_2:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2:0' shape=(1, 1) dtype=float32>]}],
          #   x20 = tf.add(x19, 9.999999747378752e-05)
          #   # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_2' type=Rsqrt>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_2:0' shape=(1, 1) dtype=float32>]}],
          #   x21 = tf.math.rsqrt(x20)
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_3' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_2:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>]}],
          #   x22 = tf.multiply(x17, x21)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_5' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>]}],
          #   x23 = tf.matmul(x22, w_reshaped)
          #   assert_shape(x23, [1, 24576])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_6' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_6:0' shape=(1, 1) dtype=float32>]}],
          #   x24 = tf.matmul(x11, x23, transpose_b=True)
          #   assert_shape(x24, [1, 1])
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_4' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_6:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_4:0' shape=(1, 24576) dtype=float32>]}],
          #   x25 = tf.multiply(x24, x11)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_7' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_7:0' shape=(1, 1) dtype=float32>]}],
          #   x26 = tf.matmul(x11, x11, transpose_b=True)
          #   assert_shape(x26, [1, 1])
          #   # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_1' type=RealDiv>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_4:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_7:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_1:0' shape=(1, 24576) dtype=float32>]}],
          #   x27 = tf.div(x25, x26)
          #   # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_1' type=Sub>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_1:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>]}],
          #   x28 = tf.subtract(x23, x27)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_3/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3/y:0' shape=() dtype=float32>]},
          #   #     {'value': 2.0}],
          #   # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_3' type=Pow>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3:0' shape=(1, 24576) dtype=float32>]}],
          #   x29 = tf.pow(x28, 2.0)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_3' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_3:0' shape=(2,) dtype=int32>]},
          #   #     {'value': array([0, 1], dtype=int32)}],
          #   # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_3' type=Sum>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_3:0' shape=(2,) dtype=int32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_3:0' shape=(1, 1) dtype=float32>]}],
          #   x30 = tf.reduce_sum(x29, axis=[0, 1], keepdims=True)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_3/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3/y:0' shape=() dtype=float32>]},
          #   #     {'value': 9.999999747378752e-05}],
          #   # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_3' type=Add>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_3:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3:0' shape=(1, 1) dtype=float32>]}],
          #   x31 = tf.add(x30, 9.999999747378752e-05)
          #   # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_3' type=Rsqrt>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_3:0' shape=(1, 1) dtype=float32>]}],
          #   x32 = tf.math.rsqrt(x31)
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_5' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_3:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>]}],
          #   x33 = tf.multiply(x28, x32)
          #   # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2' type=ReadVariableOp>,
          #   #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2:0' shape=(1, 24576) dtype=float32>]}],
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_8' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>]}],
          #   x34 = tf.matmul(u2, w_reshaped, transpose_b=True)
          #   assert_shape(x34, [1, 20])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_9' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_9:0' shape=(1, 1) dtype=float32>]}],
          #   x35 = tf.matmul(x5, x34, transpose_b=True)
          #   assert_shape(x35, [1, 1])
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_6' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_9:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_6:0' shape=(1, 20) dtype=float32>]}],
          #   x36 = tf.multiply(x35, x5)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_10' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_10:0' shape=(1, 1) dtype=float32>]}],
          #   x37 = tf.matmul(x5, x5, transpose_b=True)
          #   assert_shape(x37, [1, 1])
          #   # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_2' type=RealDiv>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_6:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_10:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_2:0' shape=(1, 20) dtype=float32>]}],
          #   x38 = tf.div(x36, x37)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_11' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_11:0' shape=(1, 1) dtype=float32>]}],
          #   x39 = tf.matmul(x22, x34, transpose_b=True)
          #   assert_shape(x39, [1, 1])
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_7' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_11:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_7:0' shape=(1, 20) dtype=float32>]}],
          #   x40 = tf.multiply(x39, x22)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_12' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_12:0' shape=(1, 1) dtype=float32>]}],
          #   x41 = tf.matmul(x22, x22, transpose_b=True)
          #   assert_shape(x41, [1, 1])
          #   # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_3' type=RealDiv>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_7:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_12:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_3:0' shape=(1, 20) dtype=float32>]}],
          #   x42 = tf.div(x40, x41)
          #   # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_4' type=Add>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_2:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_3:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_4:0' shape=(1, 20) dtype=float32>]}],
          #   x43 = tf.add(x38, x42)
          #   # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_2' type=Sub>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_4:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>]}],
          #   x44 = tf.subtract(x34, x43)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_4/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4/y:0' shape=() dtype=float32>]},
          #   #     {'value': 2.0}],
          #   # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_4' type=Pow>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4:0' shape=(1, 20) dtype=float32>]}],
          #   x45 = tf.pow(x44, 2.0)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_4' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_4:0' shape=(2,) dtype=int32>]},
          #   #     {'value': array([0, 1], dtype=int32)}],
          #   # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_4' type=Sum>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_4:0' shape=(2,) dtype=int32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_4:0' shape=(1, 1) dtype=float32>]}],
          #   x46 = tf.reduce_sum(x45, axis=[0, 1], keepdims=True)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_5/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5/y:0' shape=() dtype=float32>]},
          #   #     {'value': 9.999999747378752e-05}],
          #   # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_5' type=Add>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_4:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5:0' shape=(1, 1) dtype=float32>]}],
          #   x47 = tf.add(x46, 9.999999747378752e-05)
          #   # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_4' type=Rsqrt>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_4:0' shape=(1, 1) dtype=float32>]}],
          #   x48 = tf.math.rsqrt(x47)
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_8' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_4:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>]}],
          #   x49 = tf.multiply(x44, x48)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_13' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>]}],
          #   x50 = tf.matmul(x49, w_reshaped)
          #   assert_shape(x50, [1, 24576])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_14' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_14:0' shape=(1, 1) dtype=float32>]}],
          #   x51 = tf.matmul(x11, x50, transpose_b=True)
          #   assert_shape(x51, [1, 1])
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_9' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_14:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_9:0' shape=(1, 24576) dtype=float32>]}],
          #   x52 = tf.multiply(x51, x11)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_15' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_15:0' shape=(1, 1) dtype=float32>]}],
          #   x53 = tf.matmul(x11, x11, transpose_b=True)
          #   assert_shape(x53, [1, 1])
          #   # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_4' type=RealDiv>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_9:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_15:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_4:0' shape=(1, 24576) dtype=float32>]}],
          #   x54 = tf.div(x52, x53)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_16' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_16:0' shape=(1, 1) dtype=float32>]}],
          #   x55 = tf.matmul(x33, x50, transpose_b=True)
          #   assert_shape(x55, [1, 1])
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_10' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_16:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_10:0' shape=(1, 24576) dtype=float32>]}],
          #   x56 = tf.multiply(x55, x33)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_17' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_17:0' shape=(1, 1) dtype=float32>]}],
          #   x57 = tf.matmul(x33, x33, transpose_b=True)
          #   assert_shape(x57, [1, 1])
          #   # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_5' type=RealDiv>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_10:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_17:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_5:0' shape=(1, 24576) dtype=float32>]}],
          #   x58 = tf.div(x56, x57)
          #   # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_6' type=Add>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_4:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_5:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_6:0' shape=(1, 24576) dtype=float32>]}],
          #   x59 = tf.add(x54, x58)
          #   # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_3' type=Sub>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_6:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>]}],
          #   x60 = tf.subtract(x50, x59)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_5/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5/y:0' shape=() dtype=float32>]},
          #   #     {'value': 2.0}],
          #   # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_5' type=Pow>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5:0' shape=(1, 24576) dtype=float32>]}],
          #   x61 = tf.pow(x60, 2.0)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_5' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_5:0' shape=(2,) dtype=int32>]},
          #   #     {'value': array([0, 1], dtype=int32)}],
          #   # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_5' type=Sum>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_5:0' shape=(2,) dtype=int32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_5:0' shape=(1, 1) dtype=float32>]}],
          #   x62 = tf.reduce_sum(x61, axis=[0, 1], keepdims=True)
          #   # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_7/y' type=Const>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7/y:0' shape=() dtype=float32>]},
          #   #     {'value': 9.999999747378752e-05}],
          #   # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_7' type=Add>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_5:0' shape=(1, 1) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7/y:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7:0' shape=(1, 1) dtype=float32>]}],
          #   x63 = tf.add(x62, 9.999999747378752e-05)
          #   # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_5' type=Rsqrt>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_5:0' shape=(1, 1) dtype=float32>]}],
          #   x64 = tf.math.rsqrt(x63)
          #   # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_11' type=Mul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_5:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>]}],
          #   x65 = tf.multiply(x60, x64)
          #   # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient' type=StopGradient>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>]}],
          #   x66 = tf.stop_gradient(x11)
          #   # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient_1' type=StopGradient>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_1:0' shape=(1, 20) dtype=float32>]}],
          #   x67 = tf.stop_gradient(x5)
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_18' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_1:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_18:0' shape=(1, 24576) dtype=float32>]}],
          #   x68 = tf.matmul(x67, w_reshaped)
          #   assert_shape(x68, [1, 24576])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_19' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_18:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_19:0' shape=(1, 1) dtype=float32>]}],
          #   x69nice = tf.matmul(x68, x66, transpose_b=True)
          #   assert_shape(x69nice, [1, 1])
          #   # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze' type=Squeeze>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_19:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze:0' shape=() dtype=float32>]}],
          #   x70 = tf.squeeze(x69nice)
          #   assert_shape(x70, [])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_20' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_20:0' shape=(1, 24576) dtype=float32>]}],
          #   x71 = tf.matmul(x22, w_reshaped)
          #   assert_shape(x71, [1, 24576])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_21' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_20:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_21:0' shape=(1, 1) dtype=float32>]}],
          #   x72 = tf.matmul(x71, x33, transpose_b=True)
          #   assert_shape(x72, [1, 1])
          #   # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze_1' type=Squeeze>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_21:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze_1:0' shape=() dtype=float32>]}],
          #   x73 = tf.squeeze(x72)
          #   assert_shape(x73, [])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_22' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_22:0' shape=(1, 24576) dtype=float32>]}],
          #   x74 = tf.matmul(x49, w_reshaped)
          #   assert_shape(x74, [1, 24576])
          #   # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_23' type=MatMul>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_22:0' shape=(1, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_23:0' shape=(1, 1) dtype=float32>]}],
          #   x75 = tf.matmul(x74, x65, transpose_b=True)
          #   assert_shape(x75, [1, 1])
          #   # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze_2' type=Squeeze>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_23:0' shape=(1, 1) dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze_2:0' shape=() dtype=float32>]}],
          #   x76 = tf.squeeze(x75)
          #   assert_shape(x76, [])
          #   with absolute_name_scope('module/Generator_1/G_Z/G_linear/norm'), absolute_variable_scope('module/Generator_1/G_Z/G_linear/norm'):
          #     # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/mul' type=Mul>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/mul:0' shape=(20, 24576) dtype=float32>]}],
          #     x77 = tf.multiply(w_reshaped, w_reshaped)
          #     # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Const' type=Const>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Const:0' shape=(2,) dtype=int32>]},
          #     #     {'value': array([0, 1], dtype=int32)}],
          #     # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Sum' type=Sum>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/mul:0' shape=(20, 24576) dtype=float32>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Const:0' shape=(2,) dtype=int32>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sum:0' shape=(1, 1) dtype=float32>]}],
          #     x78 = tf.reduce_sum(x77, axis=[0, 1], keepdims=True)
          #     # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Sqrt' type=Sqrt>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sum:0' shape=(1, 1) dtype=float32>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sqrt:0' shape=(1, 1) dtype=float32>]}],
          #     x79 = tf.math.sqrt(x78)
          #     # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Squeeze' type=Squeeze>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sqrt:0' shape=(1, 1) dtype=float32>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Squeeze:0' shape=() dtype=float32>]}],
          #     x80 = tf.squeeze(x79)
          #     assert_shape(x80, [])
          #   # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sqrt' type=Sqrt>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Squeeze:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sqrt:0' shape=() dtype=float32>]}],
          #   x81 = tf.math.sqrt(x80)
          #   # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient_2' type=StopGradient>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sqrt:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_2:0' shape=() dtype=float32>]}],
          #   x82 = tf.stop_gradient(x81)
          #   with absolute_name_scope('module/Generator_1/G_Z/G_linear/norm_1'), absolute_variable_scope('module/Generator_1/G_Z/G_linear/norm_1'):
          #     # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/mul' type=Mul>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/mul:0' shape=(20, 24576) dtype=float32>]}],
          #     x83 = tf.multiply(w_reshaped, w_reshaped)
          #     # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Const' type=Const>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Const:0' shape=(2,) dtype=int32>]},
          #     #     {'value': array([0, 1], dtype=int32)}],
          #     # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Sum' type=Sum>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/mul:0' shape=(20, 24576) dtype=float32>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Const:0' shape=(2,) dtype=int32>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sum:0' shape=(1, 1) dtype=float32>]}],
          #     x84 = tf.reduce_sum(x83, axis=[0, 1], keepdims=True)
          #     # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt' type=Sqrt>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sum:0' shape=(1, 1) dtype=float32>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt:0' shape=(1, 1) dtype=float32>]}],
          #     x85 = tf.math.sqrt(x84)
          #     # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Squeeze' type=Squeeze>,
          #     #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt:0' shape=(1, 1) dtype=float32>,
          #     #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Squeeze:0' shape=() dtype=float32>]}],
          #     x86 = tf.squeeze(x85)
          #   # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_6' type=RealDiv>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze:0' shape=() dtype=float32>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_6:0' shape=(20, 24576) dtype=float32>]}],
          #   x87 = tf.div(w_reshaped, x70)
          #   # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp' type=AssignVariableOp>,
          #   #     <tf.Tensor 'module/Generator/G_Z/G_linear/u0:0' shape=() dtype=resource>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': []}],
          #   x88 = u0.assign(x66, read_value=False)
          #   # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_3' type=ReadVariableOp>,
          #   #     <tf.Tensor 'module/Generator/G_Z/G_linear/u0:0' shape=() dtype=resource>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_3:0' shape=(1, 24576) dtype=float32>]}],
          #   with tf.control_dependencies([x88]):
          #     x89 = u0.read_value()
          #   # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp_1' type=AssignVariableOp>,
          #   #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': []}],
          #   x90 = u1.assign(x33, read_value=False)
          #   # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_4' type=ReadVariableOp>,
          #   #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_4:0' shape=(1, 24576) dtype=float32>]}],
          #   with tf.control_dependencies([x90]):
          #     x91 = u1.read_value()
          #   # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp_2' type=AssignVariableOp>,
          #   #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
          #   #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>,
          #   #     {'outputs': []}],
          #   x92 = u2.assign(x65, read_value=False)
          #   # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_5' type=ReadVariableOp>,
          #   #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
          #   #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_5:0' shape=(1, 24576) dtype=float32>]}],
          #   with tf.control_dependencies([x92]):
          #     x93 = u2.read_value()
          #   with tf.control_dependencies([x88, x90, x92]):
          #     #return tf.identity(x70), x89, x91, x93, w_reshaped
          #     return tf.identity(x70)

    

def test_biggan(graph=None):
  if graph is None:
    graph = tf.Graph()
  offset = len(graph.get_operations())
  with graph.as_default():
    make_biggan()
  ops = [graph_to_code.PrettyOp(op) for op in graph.get_operations()[offset:]]
  return ops
