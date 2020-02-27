import numpy as np
import tensorflow as tf
import tflex
from pprint import pprint as pp

from train_flags import FLAGS

# Sample 1024(+1) tokens from the stitched together text
def sample_text(x, amount):
  s = tf.size(x)
  r = tf.random.uniform([], maxval=s-(amount+1), dtype=tf.dtypes.int32)
  r1 = tf.range(r, r+amount)
  r2 = tf.range(r+1, (r+1)+amount)
  r1 = tf.reshape(r1, [amount]) # Somehow, this makes the compiler happy
  r2 = tf.reshape(r2, [amount]) # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
  vals1 = tf.gather(x, r1)
  vals2 = tf.gather(x, r2)
  features, labels = vals1, vals2
  return features, labels

def make_source_dataset(index, num_hosts, batch_size, n_ctx):
  pp({'op': 'make_source_dataset', 'index': index, 'num_hosts': num_hosts, 'batch_size': batch_size})
  tokens = [[(_ + 0) for _ in range(0, n_ctx)]] * batch_size
  labels = [[(_ + 1) for _ in range(0, n_ctx)]] * batch_size
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

def make_source_tokens(index, num_hosts, n_vocab):
  #tokens = [(_ + 0) for _ in range(0, n_ctx+1)]
  if FLAGS.dataset is not None:
    tokens = []
    npz = np.load(FLAGS.dataset)
    for item in npz.files:
      tokens.extend(npz[item])
  else:
    tokens = [(_ + 0) % n_vocab for _ in range(0, 100000)]
  tf.logging.info("Dataset has %d tokens", len(tokens))
  t = tf.broadcast_to(tokens, [len(tokens)])
  dset = tf.data.Dataset.from_tensors(t);
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
  if False:
    dset = make_source_dataset(current_host, num_hosts, batch_size, n_ctx=params['n_ctx'])
  else:
    dset = make_source_tokens(current_host, num_hosts, n_vocab=params['n_vocab'])
    batch=True
    def _sample_text(*args, **kws):
      return sample_text(*args, **kws, amount=params['n_ctx'])
    if batch:
      iterations = FLAGS.iterations_per_loop
      dset = dset.apply(tf.data.experimental.map_and_batch(
          map_func=_sample_text, batch_size=batch_size,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          drop_remainder=True))
      dset = dset.repeat().prefetch(iterations)
    else:
      dset = dset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
  return dset

