
import tensorflow as tf

tf1 = tf.compat.v1


def make_session(graph):
  session_config = config_pb2.ConfigProto(allow_soft_placement=True, isolate_session_state=False)
  master = None
  res = None
  cluster_spec = None
  cluster_def = None
  job_names = None
  master_job = 'worker'
  if 'TPU_NAME' in os.environ:
    res = TPUClusterResolver(os.environ['TPU_NAME'])
    master = res.get_master()
    cluster_spec = res.cluster_spec()
    if cluster_spec:
      cluster_def = cluster_spec.as_cluster_def()
      session_config.cluster_def.CopyFrom(cluster_def)
      job_names = set([job.name for job in cluster_def.job])
      assert len(job_names) == 1
      master_job = cluster_def.job[0].name
  elif 'TPU_IP' in os.environ:
    master = os.environ['TPU_IP'].replace('grpc://', '')
    if ':' not in master:
      master = master + ':8470'
    master = 'grpc://' + master
  sess = tf.Session(master, graph=graph, config=session_config)
  return sess
  

def tf_format(template, *inputs):
  op = tf.strings.format(template, inputs)
  # strip quotes introduced by formatting.
  op = tf.strings.regex_replace(op, """["']""", "")
  return op


def tf_cat(*args):
  return tf_format('{}' * len(args), *args)


def tf_io_encode_raw(x):
  x = tf.convert_to_tensor(x)
  unit_size = x.dtype.size
  total_size = tf.size(x, out_type=tf.int64) * unit_size
  serialized = tf.serialize_tensor(x)
  serialized_size = tf.size(tf.strings.bytes_split(serialized), out_type=tf.int64)
  offset = serialized_size - total_size
  return tf.strings.substr(serialized, offset, -1)


def tf_write(filename, contents):
  if contents.dtype != tf.string:
    contents = tf_io_encode_raw(contents)
  if isinstance(filename, (list, tuple)):
    filename = tf_cat(*filename)
  return tf.raw_ops.WriteFile(filename=filename, contents=contents)


class Swarm:
  def __init__(self, graph=None, session=None):
    if graph is None:
      graph = tf.Graph()
    if session is None:
      session = make_session(graph=graph)
    self.session = session
    self.graph = graph
    with self.graph.as_default():
      self.ph_filename = tf.placeholder(tf.string, shape=[], name='filename')
      self.ph_contents = tf.placeholder(tf.string, shape=[], name='contents')
      self.write_file_op = tf.raw_ops.WriteFile(filename=self.ph_filename, contents=self.ph_contents)


import memory_saving_gradients as memg

#from models.gpt2 import gpt2, gpt2_rev, sample, encoder
from models.gpt2 import gpt2, sample, encoder

from importlib import reload

tf1 = tf.compat.v1

reload(gpt2)
reload(gpt2_rev)
reload(sample)
reload(encoder)

params = gpt2.default_hparams()
context_fixed = tf1.placeholder(tf.int32, shape=[1, params['n_ctx']+1], name="tokens")
reload(gpt2); output_fixed = gpt2.model(params=params, X=context_fixed[:, :-1], labels=context_fixed[:, 1:], reuse=tf1.AUTO_REUSE)



def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits
    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

context = tf1.placeholder(tf.int32, shape=[1, None], name="tokens")
reload(gpt2); fwd = gpt2.model(params=params, X=context, reuse=tf1.AUTO_REUSE)
temp = tf1.placeholder(tf.float32, shape=[], name="temperature")
logits = fwd['logits'][:, -1, :]
logits = logits / temp
logits = top_k_logits(logits, 40)
token = tf.random.categorical(logits, 1, dtype=tf.int32)
completion = tf.concat([context, token], axis=-1)



import tflex
gs = tflex.get_or_create_global_step()

train_vars = [x for x in tf1.trainable_variables() if x.name.startswith(params['scope'] + '/')]
saver = tf1.train.Saver(var_list=train_vars)
saver.restore(sess, 'gs://tpu-usc1/models/gpt-2/117M/model.ckpt')

from transformers import GPT2TokenizerFast
enc = GPT2TokenizerFast.from_pretrained('gpt2')

def restore(path, step=None):
  global ckpt
  if step is None:
    ckpt = tf.train.latest_checkpoint(path)
  else:
    ckpt = os.path.join(path, 'model.ckpt-' + str(step))
  saver.restore(sess, ckpt)
  step = int(ckpt.split('-')[-1])
  gs.load(step)
  return step, ckpt


def complete(txt, n=16, temperature=0.7):
  toks = [enc.encode(txt)]
  for i in range(n):
    #toks = r(completion, {context: toks, temp: temperature})
    token = r(completion_op, {context: toks, temp: temperature})[0][0][0]
    toks[0].append(token)
    print(toks, token)
    # print(toks, token)
    # toks = [toks[0] + [token]]
  print(toks)
  return enc.decode(toks[0])

def complete(txt, n=16, temperature=0.7):
  toks = [enc.encode(txt)]
  for i in range(n):
    toks = r(completion, {context: toks, temp: temperature})
  return enc.decode(toks[0])


def thunk(context):
  fwd = gpt2.model(params=params, X=context, reuse=tf1.AUTO_REUSE)
  logits = fwd['logits'][:, -1, :]
  #logits = logits / temp
  logits = top_k_logits(logits, 40)
  token = tf.random.categorical(logits, 1, dtype=tf.int32)
  return token

completion_op = tft.tpu_shard(thunk, inputs=[[context[0]]*8])
r(completion_op, {context: [enc.encode("Hello, my name is")]})



def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=tf.int64)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]



from tensorflow.python.ops import gen_array_ops


length = tf1.placeholder(tf.int32, shape=[], name="length")

def thunk2(context, length):
  context = context[0]
  length = length[0]
  print(context, length)
  batch_size, seq_length = shape_list(context)
  print(batch_size, seq_length)
  #context = tf.concat([context, tf.fill([batch_size, length], -1)], axis=-1)
  #output = tf.fill([length, batch_size], -1)
  fixed_length = 1024
  output = tf.fill([fixed_length, batch_size], -1)
  print(output)
  def body(i, output):
    inds = tf.range(i)
    print('BODY', i, output, inds)
    tokens = tf.gather(output, inds)
    tokens = tf.transpose(tokens, [1, 0])
    tokens = tf.concat([context, tokens], axis=-1)
    #token = tf.concat([context, output[0:i]], axis=-1)
    token = thunk(tokens)
    #import pdb; pdb.set_trace()
    #context2 = tf.concat([context, token], axis=-1)
    #return context2
    #token = tf.expand_dims(token, axis=-1)
    print('TOKEN', token)
    output = gen_array_ops.inplace_update(output, [i], [token])
    return [i + 1, output]
  _, final = tf.while_loop(
      #cond=lambda *args: True,
      cond=lambda i, output: tf.less(i, length),
      #body=lambda context: tf.concat([context, thunk(context)], axis=-1),
      body=body,
      #maximum_iterations=length,
      maximum_iterations=16,
      loop_vars=[0, output],
      shape_invariants=[
        tf.TensorShape([]),
        tf.TensorShape([fixed_length, batch_size]),
      ],
      back_prop=False,
      )
  return [final]
  #return tf.concat([context, final], axis=-1)
  #token = thunk(context)
  #return token

sample_op = tft.tpu_shard(thunk2, inputs=[[context]*8, [length]*8])
r(sample_op, {context: [enc.encode("Hello, my name is")], length: 1},  options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))



from transformers import GPT2TokenizerFast
enc = GPT2TokenizerFast.from_pretrained('gpt2')

enc = encoder.get_encoder()

params = gpt2.default_hparams()
context = tf.placeholder(tf.int32, shape=[1, None], name="tokens")
fwd = gpt2_rev.model(params=params, X=context, reuse=tf1.AUTO_REUSE)



params = gpt2_rev.default_hparams()
context_fixed = tf1.placeholder(tf.int32, shape=[1, params['n_ctx']+1], name="tokens")
output_fixed = gpt2_rev.model_grad(params=params, X=context_fixed[:, :-1], labels=context_fixed[:, 1:], reuse=tf1.AUTO_REUSE)
context = tf1.placeholder(tf.int32, shape=[1, None], name="tokens")
fwd = gpt2_rev.model(params=params, X=context, reuse=tf1.AUTO_REUSE)
length = tf1.placeholder(tf.int32, shape=[], name="length")
samp = sample.sample_sequence(params=params, length=length, context=context, batch_size=1, temperature=0.7, top_k=40)
#output = gpt2_rev.model_grad(params=params, X=context[..., :-1], labels=context[..., 1:], reuse=tf1.AUTO_REUSE)
#loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output["logits"], labels=labels)


import tflex
gs = tflex.get_or_create_global_step()

train_vars = [x for x in tf1.trainable_variables() if x.name.startswith(params['scope'] + '/')]
saver = tf1.train.Saver(var_list=train_vars)
saver.restore(sess, 'gs://tpu-usc1/models/gpt-2/117M/model.ckpt')

def restore(path, step=None):
  global ckpt
  if step is None:
    ckpt = tf.train.latest_checkpoint(path)
  else:
    ckpt = os.path.join(path, 'model.ckpt-' + str(step))
  saver.restore(sess, ckpt)
  step = int(ckpt.split('-')[-1])
  gs.load(step)
  return step, ckpt


model_dir = 'gs://tpu-usc1/runs/gpt-2/revnet01/e3'
restore(model_dir)

# ckpt = tf.train.latest_checkpoint('gs://tpu-usc1/runs/gpt-2/revnet01/e3')
# saver.restore(sess, ckpt)

steps = [#610,
    1280, 1940, 2500, 2510, 2520, 2530, 2540, 3160, 3830, 4490, 5160, 5800, 6470, 7130, 7790, 8450, 9110, 9770, 10420, 11080, 11720, 12150, 12160, 12170, 12180, 12190, 12850, 13510, 14170, 14510, 14520, 14530, 14540, 14550, 14610, 14620, 14630, 14640, 14650, 15270, 15930, 15950, 15960, 15970, 15980,
    #15990,
    16000]

import tqdm

for i in tqdm.tqdm(steps):
  restore('gs://tpu-usc1/runs/gpt-2/revnet01/e3', step=i)
  r(hists)
  r(flush)
  



print(enc.decode(r(samp, {context: [enc.encode("Hello, my name is")], length: 16})[0]))


lr = tf.constant(0.00002, name="learning_rate")

opt = tf.train.AdamOptimizer(learning_rate=lr)

# #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=context[:, 1:], logits=output['logits'][:, :-1]))

# opt_grads = tf.gradients(loss, train_vars)
# #opt_grads = memg.gradients(loss, train_vars, checkpoints=output['activations'])

# opt_grad_vars = list(zip(opt_grads, train_vars))

opt_grad_vars = output['grads_and_vars']
opt_apply = opt.apply_gradients(opt_grad_vars, global_step=gs, name='train_op')


# >>> r(tf.global_variables_initializer())
# >>> r(opt_apply, {context: [[1,2]]})
# >>> r(opt_apply, {context: [[1,2]]})
# >>> r(loss, {context: [[1,2]]})
# 2.636035
# >>> r(loss, {context: [[1,2]]})
# 2.636035
# >>> r(opt_apply, {context: [[1,2]]})
# >>> r(loss, {context: [[1,2]]})
# 1.0684566
# >>> r(opt_apply, {context: [[1,2]]})
# >>> r(loss, {context: [[1,2]]})
# 0.42788827




sw2 = tf.compat.v2.summary.create_file_writer('gs://swarm-usc1/tmp/logtest');
#sw2 = tf.compat.v2.summary.create_file_writer('gs://swarm-usc1/tmp/logtest-tokens');
sw2.set_as_default()
r(sw2.init())
flush = sw2.flush()

from tensorboard.plugins.histogram import summary_v2 as histogram_summary_v2
from tensorboard.plugins.image import summary_v2 as image_summary_v2
from tensorboard.plugins.scalar import summary_v2 as scalar_summary_v2


hist_step = tf.placeholder(tf.int64, (), name='hist_step')
hist_name = tf.placeholder(tf.string, (), name='hist_name')
hist_data = tf.placeholder(tf.float64, [None, None], name='hist_data')

#hist = histogram_summary.summary_v2.histogram('name', [hist_data], step=hist_step)

import tflex

def histo(name, data, step=None, buckets=None):
  if step is None:
    step = tflex.get_or_create_global_step()
  #return histogram_summary.summary_v2.histogram(name, data, step=step)
  return histogram_summary_v2.histogram(name, data, step=step, buckets=buckets)


def histv(variable, step=None):
  return histo(variable.name.split(':')[0].lstrip('model/'), variable, step=step)


import numpy as np

np.set_printoptions(suppress=True)


colormap = turbo_colormap_data = np.array([
[0.18995,0.07176,0.23217],
[0.19483,0.08339,0.26149],
[0.19956,0.09498,0.29024],
[0.20415,0.10652,0.31844],
[0.20860,0.11802,0.34607],
[0.21291,0.12947,0.37314],
[0.21708,0.14087,0.39964],
[0.22111,0.15223,0.42558],
[0.22500,0.16354,0.45096],
[0.22875,0.17481,0.47578],
[0.23236,0.18603,0.50004],
[0.23582,0.19720,0.52373],
[0.23915,0.20833,0.54686],
[0.24234,0.21941,0.56942],
[0.24539,0.23044,0.59142],
[0.24830,0.24143,0.61286],
[0.25107,0.25237,0.63374],
[0.25369,0.26327,0.65406],
[0.25618,0.27412,0.67381],
[0.25853,0.28492,0.69300],
[0.26074,0.29568,0.71162],
[0.26280,0.30639,0.72968],
[0.26473,0.31706,0.74718],
[0.26652,0.32768,0.76412],
[0.26816,0.33825,0.78050],
[0.26967,0.34878,0.79631],
[0.27103,0.35926,0.81156],
[0.27226,0.36970,0.82624],
[0.27334,0.38008,0.84037],
[0.27429,0.39043,0.85393],
[0.27509,0.40072,0.86692],
[0.27576,0.41097,0.87936],
[0.27628,0.42118,0.89123],
[0.27667,0.43134,0.90254],
[0.27691,0.44145,0.91328],
[0.27701,0.45152,0.92347],
[0.27698,0.46153,0.93309],
[0.27680,0.47151,0.94214],
[0.27648,0.48144,0.95064],
[0.27603,0.49132,0.95857],
[0.27543,0.50115,0.96594],
[0.27469,0.51094,0.97275],
[0.27381,0.52069,0.97899],
[0.27273,0.53040,0.98461],
[0.27106,0.54015,0.98930],
[0.26878,0.54995,0.99303],
[0.26592,0.55979,0.99583],
[0.26252,0.56967,0.99773],
[0.25862,0.57958,0.99876],
[0.25425,0.58950,0.99896],
[0.24946,0.59943,0.99835],
[0.24427,0.60937,0.99697],
[0.23874,0.61931,0.99485],
[0.23288,0.62923,0.99202],
[0.22676,0.63913,0.98851],
[0.22039,0.64901,0.98436],
[0.21382,0.65886,0.97959],
[0.20708,0.66866,0.97423],
[0.20021,0.67842,0.96833],
[0.19326,0.68812,0.96190],
[0.18625,0.69775,0.95498],
[0.17923,0.70732,0.94761],
[0.17223,0.71680,0.93981],
[0.16529,0.72620,0.93161],
[0.15844,0.73551,0.92305],
[0.15173,0.74472,0.91416],
[0.14519,0.75381,0.90496],
[0.13886,0.76279,0.89550],
[0.13278,0.77165,0.88580],
[0.12698,0.78037,0.87590],
[0.12151,0.78896,0.86581],
[0.11639,0.79740,0.85559],
[0.11167,0.80569,0.84525],
[0.10738,0.81381,0.83484],
[0.10357,0.82177,0.82437],
[0.10026,0.82955,0.81389],
[0.09750,0.83714,0.80342],
[0.09532,0.84455,0.79299],
[0.09377,0.85175,0.78264],
[0.09287,0.85875,0.77240],
[0.09267,0.86554,0.76230],
[0.09320,0.87211,0.75237],
[0.09451,0.87844,0.74265],
[0.09662,0.88454,0.73316],
[0.09958,0.89040,0.72393],
[0.10342,0.89600,0.71500],
[0.10815,0.90142,0.70599],
[0.11374,0.90673,0.69651],
[0.12014,0.91193,0.68660],
[0.12733,0.91701,0.67627],
[0.13526,0.92197,0.66556],
[0.14391,0.92680,0.65448],
[0.15323,0.93151,0.64308],
[0.16319,0.93609,0.63137],
[0.17377,0.94053,0.61938],
[0.18491,0.94484,0.60713],
[0.19659,0.94901,0.59466],
[0.20877,0.95304,0.58199],
[0.22142,0.95692,0.56914],
[0.23449,0.96065,0.55614],
[0.24797,0.96423,0.54303],
[0.26180,0.96765,0.52981],
[0.27597,0.97092,0.51653],
[0.29042,0.97403,0.50321],
[0.30513,0.97697,0.48987],
[0.32006,0.97974,0.47654],
[0.33517,0.98234,0.46325],
[0.35043,0.98477,0.45002],
[0.36581,0.98702,0.43688],
[0.38127,0.98909,0.42386],
[0.39678,0.99098,0.41098],
[0.41229,0.99268,0.39826],
[0.42778,0.99419,0.38575],
[0.44321,0.99551,0.37345],
[0.45854,0.99663,0.36140],
[0.47375,0.99755,0.34963],
[0.48879,0.99828,0.33816],
[0.50362,0.99879,0.32701],
[0.51822,0.99910,0.31622],
[0.53255,0.99919,0.30581],
[0.54658,0.99907,0.29581],
[0.56026,0.99873,0.28623],
[0.57357,0.99817,0.27712],
[0.58646,0.99739,0.26849],
[0.59891,0.99638,0.26038],
[0.61088,0.99514,0.25280],
[0.62233,0.99366,0.24579],
[0.63323,0.99195,0.23937],
[0.64362,0.98999,0.23356],
[0.65394,0.98775,0.22835],
[0.66428,0.98524,0.22370],
[0.67462,0.98246,0.21960],
[0.68494,0.97941,0.21602],
[0.69525,0.97610,0.21294],
[0.70553,0.97255,0.21032],
[0.71577,0.96875,0.20815],
[0.72596,0.96470,0.20640],
[0.73610,0.96043,0.20504],
[0.74617,0.95593,0.20406],
[0.75617,0.95121,0.20343],
[0.76608,0.94627,0.20311],
[0.77591,0.94113,0.20310],
[0.78563,0.93579,0.20336],
[0.79524,0.93025,0.20386],
[0.80473,0.92452,0.20459],
[0.81410,0.91861,0.20552],
[0.82333,0.91253,0.20663],
[0.83241,0.90627,0.20788],
[0.84133,0.89986,0.20926],
[0.85010,0.89328,0.21074],
[0.85868,0.88655,0.21230],
[0.86709,0.87968,0.21391],
[0.87530,0.87267,0.21555],
[0.88331,0.86553,0.21719],
[0.89112,0.85826,0.21880],
[0.89870,0.85087,0.22038],
[0.90605,0.84337,0.22188],
[0.91317,0.83576,0.22328],
[0.92004,0.82806,0.22456],
[0.92666,0.82025,0.22570],
[0.93301,0.81236,0.22667],
[0.93909,0.80439,0.22744],
[0.94489,0.79634,0.22800],
[0.95039,0.78823,0.22831],
[0.95560,0.78005,0.22836],
[0.96049,0.77181,0.22811],
[0.96507,0.76352,0.22754],
[0.96931,0.75519,0.22663],
[0.97323,0.74682,0.22536],
[0.97679,0.73842,0.22369],
[0.98000,0.73000,0.22161],
[0.98289,0.72140,0.21918],
[0.98549,0.71250,0.21650],
[0.98781,0.70330,0.21358],
[0.98986,0.69382,0.21043],
[0.99163,0.68408,0.20706],
[0.99314,0.67408,0.20348],
[0.99438,0.66386,0.19971],
[0.99535,0.65341,0.19577],
[0.99607,0.64277,0.19165],
[0.99654,0.63193,0.18738],
[0.99675,0.62093,0.18297],
[0.99672,0.60977,0.17842],
[0.99644,0.59846,0.17376],
[0.99593,0.58703,0.16899],
[0.99517,0.57549,0.16412],
[0.99419,0.56386,0.15918],
[0.99297,0.55214,0.15417],
[0.99153,0.54036,0.14910],
[0.98987,0.52854,0.14398],
[0.98799,0.51667,0.13883],
[0.98590,0.50479,0.13367],
[0.98360,0.49291,0.12849],
[0.98108,0.48104,0.12332],
[0.97837,0.46920,0.11817],
[0.97545,0.45740,0.11305],
[0.97234,0.44565,0.10797],
[0.96904,0.43399,0.10294],
[0.96555,0.42241,0.09798],
[0.96187,0.41093,0.09310],
[0.95801,0.39958,0.08831],
[0.95398,0.38836,0.08362],
[0.94977,0.37729,0.07905],
[0.94538,0.36638,0.07461],
[0.94084,0.35566,0.07031],
[0.93612,0.34513,0.06616],
[0.93125,0.33482,0.06218],
[0.92623,0.32473,0.05837],
[0.92105,0.31489,0.05475],
[0.91572,0.30530,0.05134],
[0.91024,0.29599,0.04814],
[0.90463,0.28696,0.04516],
[0.89888,0.27824,0.04243],
[0.89298,0.26981,0.03993],
[0.88691,0.26152,0.03753],
[0.88066,0.25334,0.03521],
[0.87422,0.24526,0.03297],
[0.86760,0.23730,0.03082],
[0.86079,0.22945,0.02875],
[0.85380,0.22170,0.02677],
[0.84662,0.21407,0.02487],
[0.83926,0.20654,0.02305],
[0.83172,0.19912,0.02131],
[0.82399,0.19182,0.01966],
[0.81608,0.18462,0.01809],
[0.80799,0.17753,0.01660],
[0.79971,0.17055,0.01520],
[0.79125,0.16368,0.01387],
[0.78260,0.15693,0.01264],
[0.77377,0.15028,0.01148],
[0.76476,0.14374,0.01041],
[0.75556,0.13731,0.00942],
[0.74617,0.13098,0.00851],
[0.73661,0.12477,0.00769],
[0.72686,0.11867,0.00695],
[0.71692,0.11268,0.00629],
[0.70680,0.10680,0.00571],
[0.69650,0.10102,0.00522],
[0.68602,0.09536,0.00481],
[0.67535,0.08980,0.00449],
[0.66449,0.08436,0.00424],
[0.65345,0.07902,0.00408],
[0.64223,0.07380,0.00401],
[0.63082,0.06868,0.00401],
[0.61923,0.06367,0.00410],
[0.60746,0.05878,0.00427],
[0.59550,0.05399,0.00453],
[0.58336,0.04931,0.00486],
[0.57103,0.04474,0.00529],
[0.55852,0.04028,0.00579],
[0.54583,0.03593,0.00638],
[0.53295,0.03169,0.00705],
[0.51989,0.02756,0.00780],
[0.50664,0.02354,0.00863],
[0.49321,0.01963,0.00955],
[0.47960,0.01583,0.01055]
], dtype=np.float32)


tf_turbo_data = tf.constant(turbo_colormap_data, dtype=tf.float32, name='turbo_cmap')

# The look-up table contains 256 entries. Each entry is a floating point sRGB triplet.
# To use it with matplotlib, pass cmap=ListedColormap(turbo_colormap_data) as an arg to imshow() (don't forget "from matplotlib.colors import ListedColormap").
# If you have a typical 8-bit greyscale image, you can use the 8-bit value to index into this LUT directly.
# The floating point color values can be converted to 8-bit sRGB via multiplying by 255 and casting/flooring to an integer. Saturation should not be required for IEEE-754 compliant arithmetic.
# If you have a floating point value in the range [0,1], you can use interpolate() to linearly interpolate between the entries.
# If you have 16-bit or 32-bit integer values, convert them to floating point values on the [0,1] range and then use interpolate(). Doing the interpolation in floating point will reduce banding.
# If some of your values may lie outside the [0,1] range, use interpolate_or_clip() to highlight them.



def lerp(a, b, t):
  return (b - a) * t + a


def rerange(x, to_lo, to_hi, from_lo=None, from_hi=None):
  if from_lo is None:
    from_lo = x.min()
  if from_hi is None:
    from_hi = x.max()
  t = (x - from_lo) / (from_hi - from_lo)
  return lerp(to_lo, to_hi, t)


def interpolate(colormap, x):
  x = max(0.0, min(1.0, x))
  a = int(x*255.0)
  b = min(255, a + 1)
  f = x*255.0 - a
  return [colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f,
          colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f,
          colormap[a][2] + (colormap[b][2] - colormap[a][2]) * f]

def interpolate_or_clip(colormap, x):
  if   x < 0.0: return [0.0, 0.0, 0.0]
  elif x > 1.0: return [1.0, 1.0, 1.0]
  else: return interpolate(colormap, x)


def remapv(x, bot=None, top=None, epsilon=1e-4):
  if bot is None:
    bot = tf.reduce_min(x) if isinstance(x, tf.Tensor) else np.array(x).min()
  if top is None:
    top = tf.reduce_max(x) if isinstance(x, tf.Tensor) else np.array(x).max()
  return (x - bot) / (top - bot + epsilon)

def clamp(v, min=0., max=1.):
  #return tf.minimum(tf.maximum(v, min), max)
  return tf.clip_by_value(v, min, max)


def iround(u):
  return tf.to_int32(tf.floordiv(tf.to_float(u), 1.0))



def cmapv(x, remap=True):
  if remap:
    x = remapv(x)
  indices = iround(clamp(x*256.0, 0.0, 255.0))
  indices = tf.expand_dims(indices, axis=-1)
  rgb = tf.gather_nd(tf_turbo_data, indices)
  return rgb


def img_f32_to_u8(x):
  return tf.cast(iround(clamp(x*256.0, 0.0, 255.0)), tf.uint8)


def imgv(x, remap=True, cmap=True):
  #if len(x.shape) >= 3:
  if x.shape[0] == 1:
    x = tf.squeeze(x, axis=0)
  if x.shape[-1] == 1:
    x = tf.squeeze(x, axis=-1)
  if cmap:
    rgb = cmapv(x, remap=remap)
  else:
    rgb = tf.expand_dims(x, axis=-1)
  rgb8 = img_f32_to_u8(rgb)
  if len(rgb8.shape) <= 2:
    rgb8 = tf.tile(tf.expand_dims(rgb8, axis=0), [32, 1, 1])
  rgb8 = tf.transpose(rgb8, [1,0,2]) # [width, height, channels] to [height, width, channels]
  return tf.image.encode_png(rgb8)


def imgpath(x):
  if hasattr(x, 'name'):
    x = x.name
  return os.path.join('media', tputil.tf_sanitize_op_name(x).rsplit('_0', 1)[0] + '.png')


def imgdisk(x, path='test.png', remap=True, cmap=True):
  if path is None:
    path = imgpath(x)
  data = r(imgv(x, remap=remap, cmap=cmap))
  with open(path, 'wb') as f:
    f.write(data)


# rgb = cmapv(train_vars[-2])

# with open('test.png', 'wb') as f: f.write(r(tf.image.encode_png(tf.cast(iround(tf.tile(tf.expand_dims(rgbval, axis=0), [32, 1, 1])*255), tf.uint8))))


import sys
import os

sys.path.append('src')

import model, sample, encoder

from transformers import GPT2TokenizerFast; enc=GPT2TokenizerFast.from_pretrained('gpt2')

hparams = model.default_hparams()

context = tf.placeholder(tf.int32, shape=[1, None], name="tokens")
length = tf.placeholder(tf.int32, shape=[], name="length")
#fwd = model.model(params=params, X=context, reuse=tf.AUTO_REUSE)


output = sample.sample_sequence(hparams=hparams, length=length, batch_size=1, context=context, temperature=0.7, top_k=40)[0]

saver = tf.train.Saver()
ckpt = tf.train.latest_checkpoint('models/117M')
saver.restore(sess, ckpt)


print(enc.decode(r(output, {length: 8, context: [enc.encode("Hello, my name is")]})))


tf.global_variables()[0].load(wpe2)



reg2 = KNeighborsRegressor()

reg2.fit(colormap, np.linspace(0.0, 1.0, 256))

wpe_rgb = img2rgb('wpe_openai.png')
wte_rgb = img2rgb('wte_openai.png')


#PIL.Image.open('wpe_openai.png').convert('RGB').save('wpe_openai_90.jpg', quality=90)


wpe2 = rerange(reg2.predict(wpe_rgb.reshape([-1, 3]) / 255).reshape(wpe_rgb.shape[0:2]), wpe.min(), wpe.max())

wpe_jpg = img2rgb('wpe_openai_100.jpg')
wpe3 = rerange(reg2.predict(wpe_jpg.reshape([-1, 3]) / 255).reshape(wpe_jpg.shape[0:2]), wpe.min(), wpe.max())
np.square(wpe - wpe3).mean()

wte2 = rerange(reg2.predict(wte_rgb.reshape([-1, 3]) / 255).reshape(wte_rgb.shape[0:2]), wte.min(), wte.max())


# >>> np.square(wte - wte2).mean()
# 4.7295091816307575e-05

# >>> np.square(wpe - wpe2).mean()
# 0.00022108801059667483


from importlib import reload
import tf_tools as tft
reload(tft)


from tensorflow.python.ops import data_flow_ops
barrier = data_flow_ops.Barrier((tf.string, tf.int32), shapes=((), ()))
barrier.insert_many(0, keys=["k1", "k2"], values=["a", "b"]).run()
barrier.insert_many(1, keys=["k1"], values=[1]).run()
barrier.insert_many(0, keys=["k3"], values=["c"]).run()
barrier.insert_many(1, keys=["k3"], values=[3]).run()
barrier.insert_many(1, keys=["k2"], values=[2]).run()
r(barrier.take_many(2))



#acc1 = tf.SparseConditionalAccumulator(dtype=tf.float32)
acc1 = tft.SparseSum()
r(acc1.apply_grad([42, 69], [42.0, 420.69]))
r(acc1.apply_grad([42, 69, 128], [42.0, 420.69, 4.0]))
r(acc1.take())

acc2 = tf.SparseConditionalAccumulator(dtype=tf.float32, reduction_type='SUM')
r(acc2.apply_grad([42, 69], [42.0, 420.69]))
r(acc2.apply_grad([42, 69, 128], [42.0, 420.69, 4.0]))
r(acc2.take_grad(1))


>>> fetch_function = lambda squared_tensor:([squared_tensor.sq], lambda val: val[0])
>>> feed_function = lambda feed, feed_val: [(feed.sq, feed_val)]
>>> feed_function_for_partial_run = lambda feed: [feed.sq]
>>> tf_session_lib.register_session_run_conversion_functions(SquaredTensor, fetch_function=fetch_function, feed_function=feed_function, feed_function_for_partial_run=feed_function_for_partial_run)
>>> sq1 = SquaredTensor(tf.identity(42.0))
>>> r(tf.add(1,2))
3
>>> r(sq1)
1764.0
>>> sq1



from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

from tensorflow.python.ops import linalg

def thunk():
  a_indices = np.array([[0, 0], [2, 3], [2, 4], [3, 0]])
  a_values = np.array([1.0, 5.0, -1.0, -2.0], np.float32)
  a_dense_shape = [4, 5]
  b_indices = np.array([[0, 0], [3, 0], [3, 1]])
  b_values = np.array([2.0, 7.0, 8.0], np.float32)
  b_dense_shape = [5, 3]
  # Define (COO format) Sparse Tensors over Numpy arrays
  a_st = tf.sparse.SparseTensor(a_indices, a_values, a_dense_shape)
  b_st = tf.sparse.SparseTensor(b_indices, b_values, b_dense_shape)
  # Convert SparseTensors to CSR SparseMatrix
  a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(a_st.indices, a_st.values, a_st.dense_shape)
  b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(b_st.indices, b_st.values, b_st.dense_shape)
  # Compute the CSR SparseMatrix matrix multiplication
  c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(a=a_sm, b=b_sm, type=tf.float32)
  # Convert the CSR SparseMatrix product to a dense Tensor
  c_sm_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(c_sm, tf.float32)
  # Evaluate the dense Tensor value
  return c_sm_dense


r(tft.tpu_shard(thunk))


#c_sm_dense_value = sess.run(c_sm_dense)



def shape_to_indices(tensor):
  tensor = tf.convert_to_tensor(tensor)
  rank = len(tensor.shape)
  grid = tf.meshgrid(*[tf.range(0, x, dtype=tf.int64) for x in tensor.shape])
  indices = tf.stack(grid, axis=-1)
  return indices


def mask_to_indices(mask):
  indices = shape_to_indices(mask)
  return tf.boolean_mask(indices, mask)



block_size = 32
hidden_size = 4096

size = (hidden_size//block_size, hidden_size//block_size)

sparsity = np.random.randint(2, size=size)


indices = mask_to_indices(sparsity)



from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops


vals = [[1.0, 1.0, 1.0]]

st = tf.sparse.eye(hidden_size)
st = tf.sparse.reshape(st, [block_size, block_size, size[0], size[1]])
st = tf.sparse.transpose(st, [0,2,1,3])

s = tf.sparse.eye(16)
s = tf.sparse.reshape(s, [4, 4, 4, 4])
s = tf.sparse.transpose(s, [0,2,1,3])
qq = r(tf.sparse.to_dense(s))

s1 = tf.sparse.reshape(s, [-1, 4, 4])
sm = sparse_csr_matrix_ops.CSRSparseMatrix(s1)

mat = sparse_csr_matrix_ops.CSRSparseMatrix(vals, indices=indices)










from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

def thunk():
  a_indices = np.array([[0, 0], [2, 3], [2, 4], [3, 0]])
  a_values = np.array([1.0, 5.0, -1.0, -2.0], np.float32)
  a_dense_shape = [4, 5]
  b_indices = np.array([[0, 0], [3, 0], [3, 1]])
  b_values = np.array([2.0, 7.0, 8.0], np.float32)
  b_dense_shape = [5, 3]
  # Define (COO format) Sparse Tensors over Numpy arrays
  a_st = tf.sparse.SparseTensor(a_indices, a_values, a_dense_shape)
  b_st = tf.sparse.SparseTensor(b_indices, b_values, b_dense_shape)
  # Convert SparseTensors to CSR SparseMatrix
  a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(a_st.indices, a_st.values, a_st.dense_shape)
  b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(b_st.indices, b_st.values, b_st.dense_shape)
  # Compute the CSR SparseMatrix matrix multiplication
  c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(a=a_sm, b=b_sm, type=tf.float32)
  # Convert the CSR SparseMatrix product to a dense Tensor
  c_sm_dense = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(c_sm, tf.float32)
  return c_sm_dense


r(tft.tpu_shard(thunk))









from scipy import sparse

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging




def dense_to_csr_sparse_matrix(dense):
  dense_t = ops.convert_to_tensor(dense)
  locs = array_ops.stop_gradient(array_ops.where(math_ops.abs(dense_t) > 0))
  return sparse_csr_matrix_ops.dense_to_csr_sparse_matrix(dense_t, locs)





def thunk():
  sparsify = lambda m: m * (m > 0)
  dense_shape = [53, 65, 127]
  logits = sparsify(np.random.randn(*dense_shape))
  logits_with_ninf = np.copy(logits)
  logits_with_ninf[logits == 0] = -np.inf
  data_types = [tf.float32]
  dtype = data_types[0]
  logits_t = math_ops.cast(logits, dtype)
  logits_t_with_ninf = math_ops.cast(logits_with_ninf, dtype)
  expected = nn_ops.softmax(logits_t_with_ninf)
  sparse_logits_t = dense_to_csr_sparse_matrix(logits_t)
  softmax_sparse_logits_t = sparse_csr_matrix_ops.sparse_matrix_softmax( sparse_logits_t, type=dtype)
  dense_softmax = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense( softmax_sparse_logits_t, dtype)
  return dense_softmax

#qq = r(dense_softmax)

qq = r(tft.tpu_shard(thunk))


# https://www.tensorflow.org/api_docs/python/tf/sparse/sparse_dense_matmul


# tensorflow/python/sparse_tensor_dense_matmul_op_test --benchmarks
# A sparse [m, k] with % nonzero values between 1% and 80%
# B dense [k, n]

def sparse_test(nnz, sparse_shape, dense_shape):
  indices = tf.where(tf.random.uniform(sparse_shape, dtype=tf.float32) <= nnz)
  values = tf.ones_like(indices[..., 0], dtype=tf.float32)
  sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=sparse_shape)
  dense = tf.random.uniform(dense_shape, dtype=tf.float32)
  result = tf.sparse.sparse_dense_matmul(sparse, dense)
  #dense = tf.sparse.to_dense(sparse)
  #return sparse
  return result

def sparse_bench():
  #pp([(p, n,m,k) for p in [0.01, 0.2, 0.8] for n in [1,10,25] for m in [100, 1000] for k in [100, 1000]])
  pp([(p, [m, k], [k, n]) for p in [0.01, 0.2, 0.8] for n in [1,10,25] for m in [100, 1000] for k in [100, 1000]])







sqsh = lambda x: tf.reshape(x, [-1, shape_list(x)[-1]])

swap_last = lambda *args: (*args[:-2], args[-1], args[-2])

swp = lambda v: tf.transpose(v, swap_last(*range(len(shape_list(v)))))


