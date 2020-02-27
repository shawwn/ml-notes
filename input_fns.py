import tensorflow as tf
import tflex
from pprint import pprint as pp

def make_source_dataset(index, num_hosts, batch_size):
  pp({'op': 'make_source_dataset', 'index': index, 'num_hosts': num_hosts, 'batch_size': batch_size})
  tokens = [[_ for _ in range(0, 1024)]] * batch_size
  labels = [[_ for _ in range(1, 1025)]] * batch_size
  #with tflex.device('/tpu:%d' % index):
  #with tf.device('/job:worker/replica:0/task:0/device:CPU:0'):
  with tflex.nullcontext():
    t = tf.broadcast_to(tokens, [len(tokens), len(tokens[0])])
    l = tf.broadcast_to(labels, [len(labels), len(labels[0])])
    #dset1 = tf.data.Dataset.from_tensor_slices(t);
    #dset2 = tf.data.Dataset.from_tensor_slices(l);
    dset1 = tf.data.Dataset.from_tensors(t);
    dset2 = tf.data.Dataset.from_tensors(l);
    dset = tf.data.Dataset.zip((dset1, dset2))
    #dset = dset.shuffle()
    dset = dset.repeat()
    return dset


def gpt2_input(params):
  pp({'op': 'gpt2_input', 'params': params})
  batch_size = params['batch_size']
  # TODO(dehao): Replace the following with params['context'].current_host
  if 'context' in params:
    current_host = params['context'].current_input_fn_deployment()[1]
    num_hosts = params['context'].num_hosts
  else:
    if 'dataset_index' in params:
      current_host = params['dataset_index']
      num_hosts = params['dataset_num_shards']
    else:
      current_host = 0
      num_hosts = 1
  dset = make_source_dataset(current_host, num_hosts, batch_size)
  return dset

