import time
import os

import numpy as np

# Required import to configure core TF classes and functions.
import gin
import gin.tf.external_configurables
import tensorflow as tf
#import tensorflow.compat.v1 as tf

from absl import app
from absl import logging

import train_runner
from train_flags import flags, FLAGS, run_config

from pprint import pprint as pp
from pprint import pformat as pf

# from model_fns import gpt2_model, gpt2_rev_model
# from input_fns import gpt2_input

import BigGAN
from tfjpg_parser import ImageNet, iterate_dataset

import tflex

flags.DEFINE_multi_string(
    "gin_config", [],
    "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_bindings", [],
    "Newline separated list of Gin parameter bindings.")


def main(unused_argv):
  logging.info("Gin config: %s\nGin bindings: %s",
               FLAGS.gin_config, FLAGS.gin_bindings)
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)

  cfg = run_config()
  pp(cfg)


if __name__ == "__main__":
  app.run(main)

