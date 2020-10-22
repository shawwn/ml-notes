import time
import os

import numpy as np

# Required import to configure core TF classes and functions.
import gin
import gin.tf.external_configurables
import gin.tf.utils
import tensorflow as tf
#import tensorflow.compat.v1 as tf

from absl import app
from absl import logging

import train_runner
import train_flags

FLAGS = train_flags.FLAGS

from pprint import pprint as pp
from pprint import pformat as pps

# from model_fns import gpt2_model, gpt2_rev_model
# from input_fns import gpt2_input

import BigGAN
#from tfjpg_parser import ImageNet
import tfjpg_parser
import losses
import utils

import tflex


def main(unused_argv):
  logging.info("Gin config: %s\nGin bindings: %s",
               FLAGS.gin_config, FLAGS.gin_bindings)
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)

  cfg = train_flags.run_config()
  pp(cfg)
  trunner = train_runner.TrainRunner(
      iterations=cfg.iterations_per_loop, train_steps=cfg.train_steps)
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
  def input_fn(params):
    info = train_runner.get_input_info(params)
    pp(['input_fn.params', params])
    pp(['input_fn.info', info])
    seed = params.get('seed', None)
    # seed = 0
    # dataset = tfjpg_parser.ImageNet.make_dataset(FLAGS.dataset or "gs://dota-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*", 0, 1, seed=seed)
    #dset = tfjpg_parser.ImageNet.make_dataset(params['dataset'], info.current_host, info.num_hosts, seed=seed, shuffle_filenames=False)
    #import pdb; pdb.set_trace()
    # def filter_fn(input):
    #   pp(['filter_fn.input', input])
    #   return tf.mod(input['id'], 100) == 0
    filter_fn = None
    def parse_fn(input):
      pp(['parse_fn.input', input])
      target_image_resolution = train_flags.options().resolution
      target_image_shape = [target_image_resolution, target_image_resolution]
      image = ((input['image'] / 255) - 0.5) * 2.0
      image = tf.image.resize_image_with_pad(
        image, target_image_shape[1], target_image_shape[0],
        method=tf.image.ResizeMethod.AREA)
      features = image
      label = tf.mod(input['id'], 1000)
      return {'reals': (features, label)}
    dset = tfjpg_parser.ImageNet.make_dataset(
        params['dataset'],
        info.current_host,
        info.num_hosts,
        seed=seed,
        shuffle_filenames=False,
        #filter_fn=lambda dset: pp(dset) or True,
        #parse_fn=lambda dset: {'image': ((dset['image'] / 255) - 0.5) * 2.0},
        filter_fn=filter_fn,
        parse_fn=parse_fn,
        #batch_size=params['batch_size'],
        batch_size=params['batch_per_core'],
        cache_image_data=True,
        )
    pp(['training_dataset', dset])
    return dset
  def create_train_op(input, labels, params):
    assert labels is None
    reals, reals_class_id = input['reals']
    pp(['input', input])
    pp(['reals', reals])
    pp(['reals_class_id', reals_class_id])
    pp(['params', params])
    mdl = BigGAN.GAN()
    BigGAN.instance = mdl
    dim_z = mdl.gan.generator.dim_z
    nclasses = mdl.gan.discriminator.n_class
    N, H, W, C = reals.shape.as_list()
    fakes_z, fakes_class_id = utils.prepare_z_y(G_batch_size=N, dim_z=dim_z, nclasses=nclasses)
    reals_y = tf.one_hot(reals_class_id, nclasses)
    fakes_y = tf.one_hot(fakes_class_id, nclasses)
    fakes = mdl.gan.generator(fakes_z, fakes_y)
    reals_D = mdl.gan.discriminator(reals, reals_y)
    fakes_D = mdl.gan.discriminator(fakes, fakes_y)
    global_step = tflex.get_or_create_global_step()
    #inc_global_step = global_step.assign_add(1, read_value=False, name="inc_global_step")
    # G_vars = []
    # D_vars = []
    # for variable in tf.trainable_variables():
    #   if variable.name.startswith('Generator/'):
    #     G_vars.append(variable)
    #   elif variable.name.startswith('Discriminator/'):
    #     D_vars.append(variable)
    #   elif variable.name.startswith('linear/w'):
    #     G_vars.append(variable)
    #     D_vars.append(variable)
    #   else:
    #     import pdb; pdb.set_trace()
    #     assert False, "Unexpected trainable variable"
    T_vars = tf.trainable_variables()
    G_vars = [x for x in T_vars if x.name.startswith('Generator/') or x.name.startswith('linear/w:')]
    D_vars = [x for x in T_vars if x.name.startswith('Discriminator/') or x.name.startswith('linear/w:')]
    leftover_vars = [x for x in T_vars if x not in G_vars and x not in D_vars]
    if len(leftover_vars) > 0:
      import pdb; pdb.set_trace()
      raise ValueError("Unexpected trainable variables")
    # pp({
    #   "G_vars": G_vars,
    #   "D_vars": D_vars,
    #   "leftover_vars": leftover_vars,
    #   })
    if True:
      def should_train_variable(v): return True
      train_vars = [v for v in tf.trainable_variables() if should_train_variable(v)]
      non_train_vars = [v for v in tf.trainable_variables() if not should_train_variable(v)]
      other_vars = [v for v in tf.global_variables() if v not in train_vars and v not in non_train_vars]
      local_vars = [v for v in tf.local_variables()]

      paramcount = lambda vs: sum([np.prod(v.shape.as_list()) for v in vs])

      def logvars(variables, label, print_variables=False):
        if print_variables:
          tf.logging.info("%s (%s parameters): %s", label, paramcount(variables), pps(variables))
        else:
          tf.logging.info("%s (%s parameters)", label, paramcount(variables))
        return variables

      tf.logging.info("Training %d parameters (%.2fM) out of %d parameters (%.2fM)" % (
        paramcount(train_vars), paramcount(train_vars)/(1024.0*1024.0),
        paramcount(tf.trainable_variables()), paramcount(tf.trainable_variables())/(1024.0*1024.0),
        ))

      tf.logging.info("---------")
      tf.logging.info("Variable details:")
      logvars(train_vars, "trainable variables", print_variables=True)
      logvars(non_train_vars, "non-trainable variables", print_variables=True)
      logvars(other_vars, "other global variables", print_variables=True)
      logvars(local_vars, "other local variables", print_variables=True)

      tf.logging.info("---------")
      tf.logging.info("Variable summary:")
      logvars(train_vars, "trainable variables")
      logvars(non_train_vars, "non-trainable variables")
      logvars(other_vars, "other global variables")
      logvars(local_vars, "other local variables")
    
    G_loss = losses.generator_loss(fakes_D)
    D_loss_real, D_loss_fake = losses.discriminator_loss(reals_D, fakes_D)
    D_loss = D_loss_real + D_loss_fake
    #loss = tf.constant(0.0)
    loss = G_loss + D_loss
    optimizer = tf.train.AdamOptimizer()
    if params['use_tpu']:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)
    #import pdb; pdb.set_trace()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # To update batchnorm, if present
    pp(['tf.GraphKeys.UPDATE_OPS', update_ops])
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, var_list=T_vars, global_step=global_step)
      return train_op, loss #D_loss_real
  def model_fn(input, labels, mode, params):
    pp(['model_fn.mode', mode])
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op, loss = create_train_op(input, labels, params)
      if params['use_tpu']:
        return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
      else:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    else:
      import pdb; pdb.set_trace()
      raise NotImplementedError()
  params = train_flags.options()
  trunner.initialize(input_fn, model_fn, params)
  tf.logging.info('trunner.initialize(): Done. Training...')
  trunner.train()
  tf.logging.info('trunner.train(): Done. Shutting down...')
  trunner.shutdown()
  tf.logging.info('trunner.shutdown(): Done.')
  


if __name__ == "__main__":
  app.run(main)

