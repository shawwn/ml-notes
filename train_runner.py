# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bypass TPUEstimator for ResNet-50 Train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import threading
import time
import os

from absl import flags
import tensorflow as tf
import tflex

from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.framework import graph_io

FLAGS = flags.FLAGS

_INITIAL_LOSS = 1e7


def device_for_tpu_core(task=0, core=0):
  job_name = FLAGS.tpu_job_name or "worker" #"tpu_worker"
  return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job_name, task,
                                                            core)


def wrap_computation_in_while_loop(op_fn, n, parallel_iterations=1):
  """Wraps the ops generated by `op_fn` in tf.while_loop."""

  def computation(i):
    ops = op_fn()
    if not isinstance(ops, list):
      ops = [ops]
    with tf.control_dependencies(ops):
      return i + 1

  return tf.while_loop(
      lambda i: tf.constant(True) if n is None else tf.less(i, n),
      computation, [tf.constant(0)],
      parallel_iterations=parallel_iterations)


def tpu_ordinal_fn(shard_index_in_host):
  """Return the TPU ordinal associated with a shard.

  Required because the enqueue ops are placed on CPU.

  Args:
    shard_index_in_host: the shard index

  Returns:
    The ordinal of the TPU device the shard's infeed should be placed on.
  """
  return shard_index_in_host % FLAGS.tpu_cores_per_host


class TrainRunner(object):
  """Remove init overheads in TPU Estimator via direct session.run calls."""

  def __init__(self, iterations, train_steps=-1):
    tf.logging.info("TrainRunner: constructor")
    self.feature_structure = {}
    self.loss = None
    self.infeed_queue = []
    self.enqueue_ops = []
    self.dataset_initializer = []
    self.iterations = iterations
    self.sess = None
    self.input_sess = None
    self.infeed_thread = None
    if train_steps < 0:
      train_steps = None
    if train_steps is not None:
      if train_steps % iterations != 0:
        train_steps = iterations * int(math.ceil(train_steps / iterations))
    self.train_steps = train_steps
    self.input_graph = tf.Graph()
    with tf.Graph().as_default() as self.init_graph:
      self.tpu_init = tpu.initialize_system()
      self.tpu_shutdown = tpu.shutdown_system()
    #self.cluster_resolver = tflex.TPUClusterResolver(
    self.cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu or FLAGS.master,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
    self.config = tf.ConfigProto(operation_timeout_in_ms=600 * 60 * 1000,
                                 graph_options=tf.GraphOptions(
                                     rewrite_options=rewriter_config_pb2.RewriterConfig(
                                         disable_meta_optimizer=True)),
                                 isolate_session_state=True)
    cluster_spec = self.cluster_resolver.cluster_spec()
    if cluster_spec:
      self.config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.init_sess = tf.Session(self.cluster_resolver.get_master(), graph=self.init_graph, config=self.config)
    tf.logging.info("TrainRunner: initializing TPU session...")
    if not bool(int(os.environ.get('TPU_NO_INIT', '0'))):
      tflex.run(self.init_sess, self.tpu_init)
    tf.logging.info("TrainRunner: initializing TPU session (done)")

  def device_for_host(self, task=0, cpu=0):
    job_name = FLAGS.tpu_job_name or "worker" # "tpu_worker"
    #return "/job:%s/task:%d/device:CPU:%d" % (job_name, task, cpu)
    return "/job:%s/replica:0/task:%d/device:CPU:%d" % (job_name, task, cpu)

  def build_enqueue_ops(self, input_fn, params, host_id):
    """Build enqueue operations for the input pipeline in a given host.

    Args:
      input_fn: dataset input graph generation function
      params:  input function parameters
      host_id:  host identifier
    """

    iparams = {}
    for k, v in params.items():
      iparams[k] = v
    iparams["batch_size"] = params["batch_size"] // FLAGS.num_cores
    iparams["dataset_num_shards"] = FLAGS.num_cores // FLAGS.tpu_cores_per_host

    def get_enqueue_ops_fn():
      """Generate the enqueue ops graph function."""

      iparams["dataset_index"] = host_id
      dataset = input_fn(iparams)
      iterator = dataset.make_initializable_iterator()
      self.dataset_initializer.append(iterator.initializer)

      def enqueue_ops_fn():
        """Generate the infeed enqueue ops graph."""

        per_host_sharded_inputs = []
        control_deps = []
        with tf.device(self.device_for_host(task=host_id)):
          for _ in range(FLAGS.tpu_cores_per_host):
            with tf.control_dependencies(control_deps):
              features, labels = iterator.get_next()
            self.feature_structure["features"] = features
            self.feature_structure["labels"] = labels
            flattened_inputs = data_nest.flatten(self.feature_structure)
            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          infeed = tpu.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs, tpu_ordinal_function=tpu_ordinal_fn)

      return enqueue_ops_fn

    with self.input_graph.as_default():
      with tf.device(self.device_for_host(host_id)):
        self.enqueue_ops.append(
            wrap_computation_in_while_loop(
                get_enqueue_ops_fn(),
                n=self.train_steps,
                parallel_iterations=1))

  def initialize(self, input_fn, model_fn, params):
    """Build graphs for the TPU device and the input pipelines.

    Args:
      input_fn: Dataset input graph generation function
      model_fn: Model definition function
      params:  Parameters to input and model functions
    """

    tf.logging.info("TrainRunner: initialize method")

    with tf.device(self.device_for_host()):
      self.global_step = tflex.get_or_create_global_step()

    def infeed_thread_fn():
      """Build and infeed session.run calls in a background thread."""
      i = 1
      while i < FLAGS.num_cores // FLAGS.tpu_cores_per_host:
        self.build_enqueue_ops(input_fn, params, i)
        i += 1
      # Build infeed sesssion
      self.input_sess = tf.Session(
          self.cluster_resolver.get_master(),
          graph=self.input_graph,
          config=self.config)
      self.input_sess.run(self.dataset_initializer)
      tf.logging.info('Ensure infeed data has fully uploaded')
      tflex.flush(self.input_sess)
      tf.logging.info('Run infeed session.run calls')
      tflex.run(self.input_sess, [self.enqueue_ops])

    self.build_enqueue_ops(input_fn, params, 0)

    def get_tpu_step(mparams):
      """Get the TPU graph generation function."""

      def tpu_pre(loss):
        """Generate the TPU graph."""
        del loss
        values = self.infeed_queue[0].generate_dequeue_op(tpu_device=0)
        unflattened_inputs = data_nest.pack_sequence_as(self.feature_structure,
                                                        values)
        features = unflattened_inputs["features"]
        labels = unflattened_inputs["labels"]
        estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN,
                                  mparams)
        return estimator_spec

      def tpu_make(estimator_spec):
        loss, train_op = estimator_spec.loss, estimator_spec.train_op
        with tf.device(device_for_tpu_core()):
          with tf.control_dependencies([train_op]):
            return tf.identity(loss, name="tpu_loss_op")

      def tpu_step(loss):
        estimator_spec = tpu_pre(loss)
        return tpu_make(estimator_spec)

      return tpu_pre, tpu_make, tpu_step

    tpu_pre, tpu_make, tpu_step = get_tpu_step(params)

    if False:
      with tf.Graph().as_default() as self.tpu_graph:
        params['use_tpu'] = False
        self.tpu_global_step = tflex.get_or_create_global_step()
        self.tpu_spec = tpu_pre(_INITIAL_LOSS)
        self.tpu_sess = tf.Session(self.cluster_resolver.get_master(), config=self.config, graph=self.graph)
        import pdb; pdb.set_trace()
        self.tpu_op = tpu_make(self.tpu_spec)
        params['use_tpu'] = True

    @tpu_function.on_device_training_loop
    def tpu_loop():
      return tpu.repeat(self.iterations, tpu_step, [_INITIAL_LOSS])
      #return tpu_step(_INITIAL_LOSS)

    (self.loss,) = tpu.shard(
        tpu_loop,
        inputs=[],
        num_shards=FLAGS.num_cores,
        outputs_from_all_shards=False,
    )
    initializer = tf.global_variables_initializer()
    if FLAGS.restore_trainable_variables:
      self.var_list = tf.trainable_variables()
    else:
      self.var_list = tf.global_variables()
    self.saver = tf.train.Saver(var_list=self.var_list, keep_checkpoint_every_n_hours=0.5)
    graph_io.write_graph(tf.Graph().as_graph_def(add_shapes=True),
                         FLAGS.model_dir, "graph.pbtxt")

    # Build tpu train model session and initialize graph
    self.sess = tf.Session(self.cluster_resolver.get_master(), config=self.config)
    tflex.run(self.sess, initializer)

    if FLAGS.restore_dir is not None:
      ckpt = tf.train.latest_checkpoint(FLAGS.restore_dir)
      if ckpt is not None:
        step = tflex.checkpoint_step(ckpt) or 0
        saver = tf.train.Saver(var_list=self.var_list, restore_sequentially=True)
        for x in var_list:
          tf.logging.info('\t%s', repr(x))
        tf.logging.info('Restoring %s step %d', ckpt, step)
        saver.restore(self.sess, ckpt)
        tf.logging.info('Setting step %d', step)
        self.global_step.load(step, self.sess)
        tf.logging.info('Restoring %s step %d (done)', ckpt, step)
    self.cur_step = tflex.run(self.sess, self.global_step)

    # Complete infeed graph generation and session.run calls
    self.infeed_thread = threading.Thread(target=infeed_thread_fn, daemon=True)
    self.infeed_thread.start()

  def train(self, num_threads=1, output_summaries=True):
    """Run the Train steps on the TPU device.

    Args:
      num_threads: number of outstanding checkpointing threads

    """
    if output_summaries:
      output_dir = os.path.join(FLAGS.model_dir, "eval")
      tf.gfile.MakeDirs(output_dir)
      # Summary writer writes out eval metrics.
      summary_writer = tf.compat.v1.summary.FileWriter(output_dir)

    def checkpoint_thread_fn(saver, sess):
      step = self.cur_step
      path = FLAGS.model_dir + "/model.ckpt-%d" % step
      tf.logging.info('step %d: Saving checkpoint %s...', step, path)
      now = time.time()
      saver.save(sess, path, write_meta_graph=False, global_step=step)
      elapsed = time.time() - now
      tf.logging.info('step %d: Saved checkpoint %s in %.2fs', step, path, elapsed)

    @tflex.register_command
    def save():
      checkpoint_thread_fn(self.saver, self.sess)

    thread_id = 0
    checkpoint_threads = []
    need_final_checkpoint = False
    tf.logging.info("TrainRunner: step %d", self.cur_step)
    #tflex.run(sess, self.global_step.initializer, dict([(self.global_step.initializer.inputs[1], self.cur_step)]))
    for i in range(num_threads):
      checkpoint_threads.append(None)
    end_step = None if self.train_steps is None else (self.cur_step + self.train_steps)
    while True if end_step is None else (self.cur_step < end_step):
      tflex.check_commands()
      if tflex.should_quit():
        tf.logging.info("TrainRunner: quitting")
        break
      start = time.time()
      tf.logging.info("TrainRunner: start next %d steps", self.iterations)
      self.cur_step += self.iterations
      loss = tflex.run(self.sess, [self.loss])
      thread = checkpoint_threads[thread_id]
      if checkpoint_threads[thread_id] is not None and checkpoint_threads[thread_id].is_alive():
        tf.logging.info("TrainRunner: checkpoint thread still active; skipping")
        need_final_checkpoint = True
      else:
        tf.logging.info("TrainRunner: starting checkpoint thread...")
        if checkpoint_threads[thread_id] is not None:
          checkpoint_threads[thread_id].join()
        checkpoint_threads[thread_id] = threading.Thread(
            target=checkpoint_thread_fn, args=(self.saver, self.sess), daemon=True)
        checkpoint_threads[thread_id].start()
        need_final_checkpoint = False
      thread_id += 1
      if thread_id >= num_threads:
        thread_id = 0
      end = time.time()
      tf.logging.info("TrainRunner: fetching global_step...")
      gs = tflex.run(self.sess, self.global_step)
      step_sec = end - start
      gs_sec = self.iterations / step_sec
      ex_sec = self.iterations * FLAGS.train_batch_size / (end - start)
      # Write out summary to tensorboard.
      if output_summaries:
        tf.logging.info("TrainRunner: writing summaries...")
        with tf.Graph().as_default():
          eval_results = {
              'loss': loss,
              'iterations_per_step': self.iterations,
              'seconds_per_step': step_sec,
              'global_step_per_second': gs_sec,
              'examples_per_second': ex_sec,
              'train_batch_size_per_core': FLAGS.train_batch_size // FLAGS.num_cores,
              'num_cores': FLAGS.num_cores,
              }
          for metric in eval_results:
            values = eval_results[metric]
            if not isinstance(values, list):
              values = [values]
            for i, value in enumerate(values):
              tag = '{}_{:02d}'.format(metric, i) if i > 0 else metric
              step = self.cur_step - len(values) + i + 1
              summaries = []
              summaries.append(tf.Summary.Value(tag=tag, simple_value=value))
              tf_summary = tf.Summary(value=list(summaries))
              summary_writer.add_summary(tf_summary, step)
          tf.logging.info("TrainRunner: flushing summaries (%d)...", self.cur_step)
          def thunk(cur_step):
            summary_writer.flush()
            tf.logging.info("TrainRunner: flushing summaries (%d) (done)", cur_step)
          tflex.parallelize([self.cur_step], thunk)
      tf.logging.info(
          "TrainRunner: step={} global={} end={} loss={} step_time={:.2f}sec examples/sec={:.7f} global_step/sec={:.7f}"
          .format(self.cur_step, gs, end_step, loss, step_sec, ex_sec, gs_sec))
    if need_final_checkpoint:
      tf.logging.info("TrainRunner: starting final checkpoint thread...")
      checkpoint_threads.append(None)
      i = len(checkpoint_threads) - 1
      checkpoint_threads[i] = threading.Thread(
          target=checkpoint_thread_fn, args=(self.saver, self.sess), daemon=True)
      checkpoint_threads[i].start()
    tf.logging.info("TrainRunner: waiting for infeed thread...")
    self.infeed_thread.join()
    tf.logging.info("TrainRunner: waiting for checkpoint threads...")
    for i in range(num_threads):
      if checkpoint_threads[i] is not None:
        checkpoint_threads[i].join()
        checkpoint_threads[i] = None
    tf.logging.info("TrainRunner: waiting for checkpoint threads (done)")
    if output_summaries:
      tf.logging.info("TrainRunner: closing summary writer...")
      summary_writer.close()
      tf.logging.info("TrainRunner: closing summary writer (done)")

  def shutdown(self):
    tf.logging.info("TrainRunner: shutting down...")
    if not bool(int(os.environ.get('TPU_NO_INIT', '0'))):
      tflex.run(self.init_sess, self.tpu_shutdown)
      tf.logging.info("TrainRunner: shutting down (done)")

