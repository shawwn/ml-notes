import tensorflow.compat.v1 as tf

from pprint import pformat as pf
from pprint import pprint as pp

import numpy as np


def getop(name, graph=None):
  if graph is None:
    graph = tf.get_default_graph()
  return graph._get_operation_by_name_unsafe(name)


def getopi(name, graph=None):
  if graph is None:
    graph = tf.get_default_graph()
  op = getop(name, graph=graph)
  if op is not None:
    ops = graph.get_operations()
    return ops.index(op)


def value_from_node_def(op):
  if hasattr(op, 'node_def'):
    op = op.node_def
  value = op.attr['value']
  if value.tensor.float_val:
    assert len(value.tensor.float_val) == 1
    return value.tensor.float_val[0]
  if value.tensor.int_val:
    assert len(value.tensor.int_val) == 1
    return value.tensor.int_val[0]
  if value.tensor.string_val:
    assert len(value.tensor.string_val) == 1
    return value.tensor.string_val[0].decode('utf8')
  import pdb; pdb.set_trace()
  raise NotImplementedError()


def multivalue_from_node_def(op):
  value = op.node_def.attr['value']
  out = op.outputs[0]
  tf_dtype = out.dtype
  np_dtype = tf_dtype.as_numpy_dtype
  if value.tensor.float_val:
    assert len(value.tensor.float_val) == 1
    return value.tensor.float_val[0]
  elif value.tensor.int_val:
    assert len(value.tensor.int_val) == 1
    return value.tensor.int_val[0]
  elif value.tensor.int64_val:
    assert len(value.tensor.int64_val) == 1
    return value.tensor.int64_val[0]
  elif value.tensor.string_val:
    result = [x.decode('utf8') for x in value.tensor.string_val]
    if len(result) == 1:
      result = result[0]
    return result
  elif value.tensor.tensor_content:
    return np.frombuffer(op.node_def.attr['value'].tensor.tensor_content, dtype=np_dtype).reshape(op.outputs[0].shape)
  elif out.shape.as_list() == [0]:
    return np.zeros(shape=out.shape, dtype=np_dtype)
  import pdb; pdb.set_trace()
  raise NotImplementedError()
    



def codeop(op):
  if hasattr(op, 'op'):
    op = op.op
  inputs = list(op.inputs)
  code = [op.type, op] + inputs
  code.append({'outputs': op.outputs})
  # if op.type == 'Const' and op.outputs[0].shape == []:
  #   value = value_from_node_def(op)
  #   code.append({'value': value})
  if op.type == 'Const':
    value = multivalue_from_node_def(op)
    code.append({'value': value})
  return code



def ppop(op):
  code = codeop(op)
  pp(code)
  return code



class PrettyOp:
  def __init__(self, op):
    if hasattr(op, 'op'):
      op = op.op
    self.op = op
  def __str__(self):
    return str(self.op)
  def __repr__(self):
    code = codeop(self.op)
    #code[0] = "{}".format(code[0].type)
    s = pf(code)
    lines = s.splitlines()
    r = [''.join(lines[0:2])]
    for line in lines[2:]:
      r.append('    ' + line)
    return '\n'.join(r)


if 'roots' not in globals():
  roots = {}

