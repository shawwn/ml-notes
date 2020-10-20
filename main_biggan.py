import time
import os

import numpy as np

import tensorflow.compat.v1 as tf

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
from tensorflow.python.tpu import tpu

import train_runner
from train_flags import FLAGS

from pprint import pprint as pp

# from model_fns import gpt2_model, gpt2_rev_model
# from input_fns import gpt2_input

import json

from tfjpg_parser import ImageNet, iterate_dataset

import tflex

def parseval(value, dtype, default=None):
  if dtype == 'str' or isinstance(default, str):
    pass
  elif dtype == 'int' or isinstance(default, int):
    value = int(value)
  elif dtype == 'float' or isinstance(default, float):
    value = float(value)
  elif dtype == 'bool' or isinstance(default, bool):
    if value == '1' or value.lower() == 'true':
      value = True
    else:
      value = False
  else:
    assert dtype is not None
    value = dtype(value)
  return value


def getval(name, default, dtype=None):
  if name.upper() in os.environ:
    value = os.environ[name.upper()]
    value = parseval(value, dtype=dtype, default=default)
    tf.logging.info('getval(%s, %s) = os.environ[%s] = %s', repr(name), repr(default), repr(name.upper()), repr(value))
  else:
    value = params.get(name, default)
    tf.logging.info('getval(%s, %s) = params[%s] = %s', repr(name), repr(default), repr(name), repr(value))
  return value


def main(unused_argv):
  global params
  #FLAGS.iterations_per_loop = 100
  #params = {'batch_size': FLAGS.train_batch_size}
  #params = {'batch_size': 128, 'use_tpu': True, 'precision': 'float32'}
  with open(FLAGS.params) as f:
    params = json.load(f)
  params['use_tpu'] = getval('use_tpu', True)
  params['batch_per_core'] = getval('batch_per_core', 1)
  params['iterations'] = getval('iterations', 20)
  params['batch_size'] = FLAGS.num_cores * params['batch_per_core']
  params['n_ctx'] = getval('n_ctx', 1024)
  params['n_embd'] = getval('n_embd', 768)
  params['n_head'] = getval('n_head', 12)
  params['n_layer'] = getval('n_layer', 12)
  params['n_vocab'] = getval('n_vocab', 50257)
  params['opt_name'] = getval('opt_name', 'adam')
  params['beta1'] = getval('beta1', 0.9)
  params['beta2'] = getval('beta2', 0.999)
  params['epsilon'] = getval('epsilon', 1e-9)
  params['lr'] = getval('lr', 0.00025)
  FLAGS.train_batch_size = params['batch_size']
  FLAGS.iterations_per_loop = params['iterations']
  FLAGS.train_steps = getval('train_steps', int(2e6))
  params['precision'] = getval('precision', 'float32')
  params['model'] = getval('model', 'GPT2')
  assert params['model'] in ['GPT2', 'GPT2Rev']

  graph = tf.Graph()
  with graph.as_default():
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu or FLAGS.master or getval('TPU_NAME', 'unknown'),
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
    config = tf.ConfigProto(operation_timeout_in_ms=600 * 60 * 1000,
                                 # graph_options=tf.GraphOptions(
                                 #     rewrite_options=rewriter_config_pb2.RewriterConfig(
                                 #         disable_meta_optimizer=True,
                                 #         ),
                                 #     ),
                                 isolate_session_state=True)
    cluster_spec = cluster_resolver.cluster_spec()
    if cluster_spec:
      config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    sess = tf.InteractiveSession(cluster_resolver.get_master(), graph=graph, config=config)
    import pdb; pdb.set_trace()
    tf.logging.info("TrainRunner: initializing TPU session...")
    if not bool(int(os.environ.get('TPU_NO_INIT', '0'))):
      tflex.run(sess, tf.tpu.initialize_system())
    tf.logging.info("TrainRunner: initializing TPU session (done)")
    

    seed = 0
    dataset = ImageNet.make_dataset("gs://dota-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*", 0, 1, seed=seed)
    it = iterate_dataset(dataset)

    def go():
      zz = next(it)
      images = [zz['image']]
      labels = [zz['label']]

      #import IPython
      print('label', labels[0])
      #print(labels[0] - 1, imagenet_label_names[labels[0] - 1])
      print(images[0].shape)
      print('embedding', zz['parsed']['image/class/embedding'].values.shape)
      print('filename', zz['parsed']['image/filename'])
      op = tf.io.encode_jpeg(images[0])
      with open('test.png', 'wb') as f:
        f.write(sess.run(op))
    go()

    import pdb; pdb.set_trace()

    dataset = dataset
    
    # model = gpt2_rev_model if params['model'] == 'GPT2Rev' else gpt2_model
    # pp(params)
    # trunner = train_runner.TrainRunner(
    #     iterations=FLAGS.iterations_per_loop, train_steps=FLAGS.train_steps)
    # def input_fn(params):
    #   tokens = [[_ for _ in range(0, 1024)]] * params['batch_size']
    #   labels = [[_ for _ in range(1, 1025)]] * params['batch_size']
    #   t = tf.broadcast_to(tokens, [len(tokens), len(tokens[0])])
    #   l = tf.broadcast_to(labels, [len(labels), len(labels[0])])
    #   #dset1 = tf.data.Dataset.from_tensor_slices(t);
    #   #dset2 = tf.data.Dataset.from_tensor_slices(l);
    #   dset1 = tf.data.Dataset.from_tensors(t);
    #   dset2 = tf.data.Dataset.from_tensors(l);
    #   dset = tf.data.Dataset.zip((dset1, dset2))
    #   dset = dset.repeat()
    #   return dset
    # def create_train_op(loss, params):
    #   return tf.identity(loss)
    # def model_fn(features, labels, mode, params):
    #   pp(['features', features])
    #   pp(['labels', labels])
    #   pp(['mode', mode])
    #   pp(['params', params])
    #   loss = tf.constant(0.0)
    #   if mode == tf.estimator.ModeKeys.TRAIN:
    #     train_op = create_train_op(loss, params)
    #     if params['use_tpu']:
    #       return tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
    #     else:
    #       return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    # trunner.initialize(gpt2_input, model, params)
    # tf.logging.info('trunner.initialize(): Done. Training...')
    # trunner.train()
    # tf.logging.info('trunner.train(): Done. Shutting down...')
    # trunner.shutdown()
    # tf.logging.info('trunner.shutdown(): Done.')

if __name__ == "__main__":
  app.run(main)

