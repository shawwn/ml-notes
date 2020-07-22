import time

import numpy as np

import tensorflow as tf

from absl import app

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_ops
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import server_lib

import train_runner
from train_flags import FLAGS

from pprint import pprint as pp

from model_fns import gpt2_model
from input_fns import gpt2_input

import json


def getval(name, default, dtype='str'):
  if name.upper() in os.environ:
    value = os.environ[name.upper()]
  else:
    value = params.get(name, default)
  if dtype == 'str':
    pass
  if dtype == 'int' or isinstance(default, int):
    value = int(value)
  if dtype == 'float' or isinstance(default, float):
    value = float(value)
  elif dtype == 'bool' or isinstance(default, bool):
    assert isinstance(value, str)
    if value == '1' or value.lower() == 'true':
      value = True
    else:
      value = False
  else:
    assert dtype is not None
    value = dtype(value)
  tf.logging.info('getval(%s, %s) = %s', repr(name), repr(default), repr(value))
  return value


def main(unused_argv):
  global params
  #FLAGS.iterations_per_loop = 100
  #params = {'batch_size': FLAGS.train_batch_size}
  #params = {'batch_size': 128, 'use_tpu': True, 'precision': 'float32'}
  with open(FLAGS.params) as f:
    params = json.load(f)
  params['use_tpu'] = getval('use_tpu', True)
  batch_per_core = getval('batch_per_core', 1)
  FLAGS.train_batch_size = FLAGS.num_cores * getval('batch_per_core', 1)
  FLAGS.iterations_per_loop = getval('iterations', 20)
  FLAGS.train_steps = getval('train_steps', int(2e6))
  params['batch_size'] = FLAGS.train_batch_size
  params['precision'] = getval('precision', 'float32')
  pp(params)
  trunner = train_runner.TrainRunner(
      iterations=FLAGS.iterations_per_loop, train_steps=FLAGS.train_steps)
  def input_fn(params):
    tokens = [[_ for _ in range(0, 1024)]] * params['batch_size']
    labels = [[_ for _ in range(1, 1025)]] * params['batch_size']
    t = tf.broadcast_to(tokens, [len(tokens), len(tokens[0])])
    l = tf.broadcast_to(labels, [len(labels), len(labels[0])])
    #dset1 = tf.data.Dataset.from_tensor_slices(t);
    #dset2 = tf.data.Dataset.from_tensor_slices(l);
    dset1 = tf.data.Dataset.from_tensors(t);
    dset2 = tf.data.Dataset.from_tensors(l);
    dset = tf.data.Dataset.zip((dset1, dset2))
    dset = dset.repeat()
    return dset
  def create_train_op(loss, params):
    return tf.identity(loss)
  def model_fn(features, labels, mode, params):
    pp(['features', features])
    pp(['labels', labels])
    pp(['mode', mode])
    pp(['params', params])
    loss = tf.constant(0.0)
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = create_train_op(loss, params)
      if params['use_tpu']:
        return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
      else:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  trunner.initialize(gpt2_input, gpt2_model, params)
  tf.logging.info('trunner.initialize(): Done. Training...')
  trunner.train()
  tf.logging.info('trunner.train(): Done. Shutting down...')
  trunner.shutdown()
  tf.logging.info('trunner.shutdown(): Done.')

if __name__ == "__main__":
  app.run(main)
