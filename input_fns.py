import numpy as np
import tensorflow as tf
import tflex
import os
import tqdm
from pprint import pprint as pp

from train_flags import FLAGS

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_examples, print_progress=True, progress_interval=10):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_examples  = expected_examples
        self.cur_examples       = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_examples)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_examples)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_tokens(self, tokens):
        if self.print_progress and self.cur_examples % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_examples, self.expected_examples), end='', flush=True)
        if len(self.tfr_writers) <= 0:
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            tfr_file = self.tfr_prefix + '.tfrecords'
            self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        for lod, tfr_writer in enumerate(self.tfr_writers):
            #import pdb; pdb.set_trace()
            data = np.array(tokens, dtype=np.int32)
            feature = {
                #"hash": _bytes_feature(hash.encode()),
                "text": _int64_feature(data)
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feature))
            s = ex.SerializeToString()
            tfr_writer.write(s)
        self.cur_examples += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()



# Sample 1024(+1) tokens from the stitched together text
def sample_text(x, amount, batch_size=None):
  if batch_size is not None:
    features, labels = [], []
    for i in range(batch_size):
      features1, labels1 = sample_text(x, amount)
      features.append(features1)
      labels.append(labels1)
    features = tf.stack(features)
    labels = tf.stack(labels)
    return features, labels
  s = tf.size(x, out_type=tf.dtypes.int64)
  r = tf.random.uniform([], maxval=s-(amount+1), dtype=tf.dtypes.int64)
  r1 = tf.range(r, r+amount)
  r2 = tf.range(r+1, (r+1)+amount)
  r1 = tf.reshape(r1, [amount]) # Somehow, this makes the compiler happy
  r2 = tf.reshape(r2, [amount]) # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
  vals1 = tf.gather(x, r1)
  vals2 = tf.gather(x, r2)
  vals1 = tf.cast(vals1, tf.dtypes.int32)
  vals2 = tf.cast(vals2, tf.dtypes.int32)
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

def export_source_tokens(tfrecord_dir, tokens):
  tf.logging.info("Exporting tokens to %s...", FLAGS.export_dataset)
  with TFRecordExporter(tfrecord_dir, 1) as tfr:
    tfr.add_tokens(tokens)
  tf.logging.info("Exported tokens to %s", FLAGS.export_dataset)

if 'api' not in globals():
  api = tflex.Dictator()
  api.tokens = None

def unload_source_tokens():
  api.tokens = None

def load_source_tokens(dataset, export_dataset=None, quit_after_exporting=True):
  if dataset is None:
    tf.logging.info("Generating random fake tokens")
    tokens = [(_ + 0) % n_vocab for _ in range(0, 100000)]
  elif dataset.endswith('.tok16'):
    tf.logging.info("Reading tokens from %s...", dataset)
    with tf.io.gfile.GFile(dataset, 'rb') as f:
      data = f.read()
      tf.logging.info("len(data)=%s; np.frombuffer(%s, dtype=np.uint16)...", len(data), repr(dataset))
      tokens = np.frombuffer(data, dtype=np.uint16)
  else:
    tf.logging.info("Loading tokens from %s...", dataset)
    tokens = []
    npz = np.load(dataset)
    for item in npz.files:
      tokens.extend(npz[item])
  tf.logging.info("Finished reading tokens.")
  if export_dataset:
    export_source_tokens(export_dataset, tokens)
    if quit_after_exporting:
      tf.logging.info("Tokens exported; quitting.")
      import posix
      posix._exit(0)
  return tokens

def get_source_tokens(dataset=None, reload=False, export_dataset=None):
  if dataset is None:
    dataset = FLAGS.dataset
  if export_dataset is None:
    export_dataset = FLAGS.export_dataset
  if api.tokens is None or reload:
    unload_source_tokens()
    api.tokens = load_source_tokens(dataset)
  return api.tokens

def make_source_tokens(index, num_hosts, n_vocab):
  tokens = get_source_tokens()
  n = len(tokens)
  k = n // num_hosts
  i = index * k
  j = (index + 1) * k
  tokens = tokens[i:j]
  tf.logging.info("Shard %d/%d has %d tokens", index, num_hosts, len(tokens))
  dset = None
  step = int(10e6)
  for offset in tqdm.trange(0, len(tokens), step):
    t = tokens[offset:offset+step]
    #t = tf.broadcast_to(tf.cast(t, tf.int32), [len(t)])
    t = tf.data.Dataset.from_tensors(t);
    dset = t if dset is None else dset.concatenate(t)
  if _loaded_dataset is not None:
    if index >= num_hosts - 1:
      tf.logging.info('Resetting tokens')
      if not isinstance(_loaded_dataset, np.ndarray):
        if isinstance(_loaded_dataset, list):
          while len(_loaded_dataset) > 0:
            _loaded_dataset.pop()
      _loaded_dataset = None
  return dset

def bpe_text(batch_size, files, iterations, stitch, amount=1024, batch=True):
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=True))

    def _parse_function(example_proto):
        features = {
            #"hash": tf.VarLenFeature(tf.string),
            "text": tf.VarLenFeature(tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["text"], parsed_features["text"].dense_shape[0]

    dataset = dataset.map(_parse_function, num_parallel_calls=1).shuffle(1000 * stitch)

    # Since samples can be less than the correct length, and TPUs don't like variable lengths, this function stitches together enough samples
    # to have a text at least 1024 tokens long. For this to work the stitch parameter must be correctly tuned so that
    # stitch * min(characters_in_text) >= amount
    def _stitch_text(x, y):
        x = tf.sparse.to_dense(x)

        def _get_x(i):
            return tf.gather(x[i], tf.range(y[i]))

        out = _get_x(0)
        for i in range(1, stitch):
            #out = tf.concat([out, [50256], _get_x(i)], axis=0) # text1<|endoftext|>text2
            out = tf.concat([out, _get_x(i)], axis=0) # text1+text2

        return out

    # Hack-y way to stitch together multiple texts
    dataset = dataset.batch(stitch, drop_remainder=True).map(_stitch_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Sample 1024(+1) tokens from the stitched together text
    def _sample_text(x):
        s = tf.size(x)
        r = tf.random.uniform([], maxval=s-(amount+1), dtype=tf.dtypes.int32)
        r1 = tf.range(r, r+amount)
        r2 = tf.range(r+1, (r+1)+amount)
        r1 = tf.reshape(r1, [amount]) # Somehow, this makes the compiler happy
        r2 = tf.reshape(r2, [amount]) # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
        vals1 = tf.gather(x, r1)
        vals2 = tf.gather(x, r2)
        vals1 = tf.cast(vals1, tf.dtypes.int32)
        vals2 = tf.cast(vals2, tf.dtypes.int32)
        return vals1, vals2

    if batch:
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=_sample_text, batch_size=batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            drop_remainder=True))
        dataset = dataset.repeat().prefetch(iterations)

    else:
        dataset = dataset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()

    return dataset


def gpt2_input(params):
  pp({'op': 'gpt2_input', 'params': params})
  batch_size = params['batch_size']
  iterations = FLAGS.iterations_per_loop
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
  elif FLAGS.dataset is not None and FLAGS.dataset.startswith('gs://') and '*' in FLAGS.dataset:
    files = []
    for fname in FLAGS.dataset.split(','):
      files.extend(sorted(tf.io.gfile.glob(fname)))
    assert len(files) > 0
    dset = bpe_text(batch_size, files, iterations=iterations, stitch=min(2, len(files)), amount=params['n_ctx'], batch=True)
  elif False:
    dset = make_source_tokens(current_host, num_hosts, n_vocab=params['n_vocab'])
    batch=True
    def _sample_text(*args, **kws):
      return sample_text(*args, **kws, amount=params['n_ctx'])
    if batch:
      dset = dset.apply(tf.data.experimental.map_and_batch(
          map_func=_sample_text, batch_size=batch_size,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          drop_remainder=True))
      dset = dset.repeat().prefetch(iterations)
    else:
      dset = dset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
  else:
    #dset = make_source_tokens(current_host, num_hosts, n_vocab=params['n_vocab'])
    tokens = get_source_tokens()
    assert tokens.ndim == 1
    tokens_count = len(tokens)
    n = len(tokens)
    k = n // num_hosts
    i = current_host * k
    j = (current_host + 1) * k
    tokens = tokens[i:j]
    tf.logging.info("Shard %d/%d has %s tokens out of %s total", current_host, num_hosts, tflex.num(len(tokens)), tflex.num(tokens_count))
    with tf.variable_scope('cpu%d' % current_host):
      tokens_var = tf.get_local_variable('input_tokens', dtype=tf.uint16, shape=[tokens_count], use_resource=True)
    def sample_fn():
      return sample_text(tokens_var, amount=params['n_ctx'], batch_size=batch_size)
    def init_fn():
      return tokens_var.initializer
    def upload_fn(session=None):
      if session is None:
        session = tf.get_default_session()
      tf.logging.info('Loading %s tokens to TPU...', tflex.num(tokens_count))
      assert session is not None
      with tflex.with_elapsed(tflex.assign_values, [tokens_var], [tokens], session=session) as (elapsed, result):
        tf.logging.info('Loaded %s tokens to TPU in %.2fs', tflex.num(tokens_count), elapsed)
    dset = tflex.make_dataset_function(sample_fn=sample_fn, init_fn=init_fn, upload_fn=upload_fn)
    return dset
  return dset

