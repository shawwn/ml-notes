import tpu_normalization as tpun
import tftorch as nn
F = nn
init = nn
import tf_tools as tft
import math
import tensorflow.compat.v1 as tf
from functools import partial


class TestModule(nn.Module):
  def __init__(self, scope=None):
    super().__init__(scope=scope)
    #self.v = tf.Variable(tf.zeros([8]), name='v', use_resource=True, trainable=False, collections=['local_variables'])
    with self.scope():
      #self.v = tft.localvar('v', shape=[8])
      self.v = tft.globalvar('v', shape=[8])
      #self.v = self.v.assign(nn.kaiming_uniform_(self.v, a=math.sqrt(5)))
      with self.scope('bn1'):
        self.bn1 = tpun.CrossReplicaBatchNormalization()
      with self.scope('bn2'):
        self.bn2 = tpun.CrossReplicaBatchNormalization()
  def forward(self, input):
    #op = tpun.cross_replica_batch_normalization(v, training=self.training, fused=False)
    x = input
    # x = self.bn1.apply(x, training=self.training)
    # x = self.bn2.apply(x, training=self.training)
    output = x
    return output



def train_op(input, lr=1e-4, use_tpu=True):
  mdl = TestModule()
  mdl.train()
  output = mdl(input)
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  if use_tpu:
    opt = tf.tpu.CrossShardOptimizer(opt)
  train_vars = tf.trainable_variables()
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # To update batchnorm, if present
  op = opt.minimize(var_list=train_vars)
  return tf.group([op, update_ops])
  




#with tf.variable_scope('model', reuse=tf.AUTO_REUSE): train_op0123 = tft.tpu_shard(lambda: with_updates(tpun.cross_replica_batch_normalization(v, training=True, fused=False)), device_assignment=get_core_assignment(0, 1, 2, 3))




class SelfAttention(nn.Module):
  """ Self Attention Layer"""

  def __init__(self, in_dim, activation=F.relu, scope='attention', **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.channel_in = in_dim
      self.activation = activation

      self.theta = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False, scope='theta'))
      self.phi = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False, scope='phi'))
      self.pool = nn.MaxPool2d(2, 2)
      #self.pool = partial(tf.layers.max_pooling2d, pool_size=[2, 2], strides=2)
      
      self.g = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1, bias=False, scope='g'))
      self.o_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1, bias=False, scope='o_conv'))
      self.gamma = self.globalvar('gamma', shape=[1])

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x, y=None): # ignore y (class embedding)
    with self.scope():
      m_batchsize, width, height, C = nn.size(x)
      N = height * width

      g = nn.view(self.pool(self.g(x)), m_batchsize, N // 4, -1)

      theta = self.theta(x)
      phi = self.phi(x)
      phi = self.pool(phi)
      phi = nn.view(phi, m_batchsize, N // 4, -1)
      theta = nn.view(theta, m_batchsize, N, -1)
      attention = self.softmax(nn.bmm(theta, phi, transpose_b=True))
      g0 = nn.bmm(attention, g)
      attn_g = nn.view(g0, m_batchsize, width, height, -1)
      out = self.o_conv(attn_g)
      
      out = self.gamma * out + x
      return out


def val(x):
  if hasattr(x, 'read_value'):
    return x.read_value()
  else:
    return tf.identity(x, name='read')


def batchnorm(input, mean, variance, scale, offset, epsilon=9.999999747378752e-05):
  return tf.nn.batch_normalization(input, mean, variance, scale=scale, offset=offset, variance_epsilon=epsilon)
  # return F.batch_norm(input, mean=mean, variance=variance, weight=scale, bias=offset, training=False, exponential_average_factor=0.1, variance_epsilon=epsilon)
  # mean_v = val(mean)
  # var_v = val(variance)
  # inv_var = tf.math.rsqrt(var_v + epsilon)
  # scale = val(scale) if scale is not None else 1.0
  # x0 = inv_var * scale
  # x1 = input * x0
  # x2 = mean_v * x1
  # offset = val(offset) if offset is not None else 0.0
  # x3 = offset - x2
  # x4 = x1 + x3
  # return x4


class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1, scope='HyperBN', bn_scope='CrossReplicaBN', gamma_scope='gamma', beta_scope='beta', **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.num_features = num_features
      self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False, scope=gamma_scope))
      self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False, scope=beta_scope))
    self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum, scope=bn_scope, **kwargs)

  def forward(self, x, y):
    with self.scope():
      scale = self.gamma_embed(y) + 1
      offset = self.beta_embed(y)
      out = batchnorm(x, mean=self.bn.accumulated_mean, variance=self.bn.accumulated_var, scale=scale, offset=offset)
      return out


class ScaledCrossReplicaBN(nn.Module):
  def __init__(self, num_features, eps=1e-4, momentum=0.1, scope='ScaledCrossReplicaBN', bn_scope_suffix='bn', scale_name='gamma', offset_name='beta', **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.num_features = num_features
      self.scale = self.globalvar(scale_name, shape=[1, 1, 1, self.num_features])
      self.offset = self.globalvar(offset_name, shape=[1, 1, 1, self.num_features])
    self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum, scope=scope+bn_scope_suffix, **kwargs)

  def forward(self, input):
    with self.scope():
      out = batchnorm(input, mean=self.bn.accumulated_mean, variance=self.bn.accumulated_var, scale=self.scale, offset=self.offset)
      return out


class GBlock(nn.Module):
  def __init__(
      self, 
      in_channel,
      out_channel,
      kernel_size=[3, 3],
      padding=1,
      stride=1,
      n_class=None,
      bn=True,
      activation=F.relu,
      upsample=True,
      downsample=False,
      z_dim=148,
      scope=None,
      **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      if bn:
        self.HyperBN = ConditionalBatchNorm2d(in_channel, z_dim, index=0)
      self.conv0 = SpectralNorm(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True, scope='conv0')
      )
      if bn:
        self.HyperBN_1 = ConditionalBatchNorm2d(out_channel, z_dim, index=1)
      self.conv1 = SpectralNorm(
        nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, bias=True if bn else True, scope='conv1')
      )

      self.skip_proj = False
      if in_channel != out_channel or upsample or downsample:
        self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, scope='conv_sc'))
        self.skip_proj = True
      
      self.upsample = upsample
      self.downsample = downsample
      self.activation = activation
      self.bn = bn

  def forward(self, input, condition=None):
    with self.scope():
      out = input

      if self.bn:
        out = self.HyperBN(out, condition)
      out = self.activation(out)
      if self.upsample:
        #out = F.interpolate(out, scale_factor=2)
        out = F.upsample(out)
      out = self.conv0(out)
      if self.bn:
        out = self.HyperBN_1(out, condition)
      out = self.activation(out)
      out = self.conv1(out)

      if self.downsample:
        #out = F.avg_pool2d(out, 2)
        out = F.downsample(out)

      if self.skip_proj:
        skip = input
        if self.upsample:
          #skip = F.interpolate(skip, scale_factor=2)
          skip = F.upsample(skip)
        skip = self.conv_sc(skip)
        if self.downsample:
          #skip = F.avg_pool2d(skip, 2)
          skip = F.downsample(skip)
      else:
        skip = input
      return out + skip
    
    


class Generator256(nn.Module):
  def __init__(self, code_dim=140, n_class=1000, chn=96, debug=False, scope='Generator', **kwargs):
    super().__init__(scope=scope, **kwargs)
    self.linear = nn.Linear(n_class, 128, bias=False)
    with self.scope():

      if debug:
        chn = 8

      self.first_view = 16 * chn

      with self.scope('G_Z'):
        self.G_linear = SpectralNorm(nn.Linear(20, 4 * 4 * 16 * chn, scope="G_linear"))

      self.GBlock = []
      self.GBlock += [GBlock(16 * chn, 16 * chn, n_class=n_class, index=0)]
      self.GBlock += [GBlock(16 * chn, 8 * chn, n_class=n_class, index=1)]
      self.GBlock += [GBlock(8 * chn, 8 * chn, n_class=n_class, index=2)]
      self.GBlock += [GBlock(8 * chn, 4 * chn, n_class=n_class, index=3)]
      self.GBlock += [GBlock(4 * chn, 2 * chn, n_class=n_class, index=4)]
      self.sa_id = len(self.GBlock)
      assert self.sa_id == 5
      self.attention = SelfAttention(2 * chn)
      self.GBlock += [GBlock(2 * chn, 1 * chn, n_class=n_class, index=5)]

      self.num_split = len(self.GBlock) + 1

      self.ScaledCrossReplicaBN = ScaledCrossReplicaBN(1 * chn, eps=1e-4)
      self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1, scope='conv_2d'))
      
      
  def forward(self, input, class_id):
    with self.scope():
      codes = tf.split(input, self.num_split, 1)
      class_emb = self.linear(class_id) # 128

      with self.scope('G_Z'):
        out = self.G_linear(codes[0])
      out = nn.view(out, -1, 4, 4, self.first_view)
      for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
        if i == self.sa_id:
          out = self.attention(out)
        condition = nn.cat([code, class_emb], 1)
        out = GBlock(out, condition)

      out = self.ScaledCrossReplicaBN(out)
      out = F.relu(out)
      out = self.colorize(out)

      return tf.tanh(out)
  


class Discriminator256(nn.Module):
  def __init__(self, n_class=1000, chn=96, debug=False, scope='Discriminator', **kwargs):
    super().__init__(scope=scope, **kwargs)

    with self.scope():

      def conv(in_channel, out_channel, downsample=True, **kwargs):
        return GBlock(in_channel, out_channel, bn=False, upsample=False, downsample=downsample, **kwargs)

      if debug:
        chn = 8
      self.debug = debug

      with self.scope('pre_conv'):
        self.pre_conv = nn.Sequential(
          SpectralNorm(nn.Conv2d(3, 1 * chn, 3, padding=1, index=0)),
          nn.ReLU(index=1),
          SpectralNorm(nn.Conv2d(1 * chn, 1 * chn, 3, padding=1, index=2)),
          nn.AvgPool2d(2, index=3),
        )
      self.pre_skip = SpectralNorm(nn.Conv2d(3, 1 * chn, 1, scope='pre_skip'))

      with self.scope('conv'):
        self.conv = nn.Sequential(
          conv(1 * chn, 2 * chn, downsample=True, index=0),
          SelfAttention(2 * chn, index=0),
          conv(2 * chn, 2 * chn, downsample=True, index=1),
          conv(2 * chn, 4 * chn, downsample=True, index=2),
          conv(4 * chn, 8 * chn, downsample=True, index=3),
          conv(8 * chn, 8 * chn, downsample=True, index=4),
          conv(8 * chn, 16 * chn, downsample=True, index=5),
          conv(16 * chn, 16 * chn, downsample=False, index=6),
        )

      self.linear = SpectralNorm(nn.Linear(16 * chn, 1, scope='linear'))

      self.embed = nn.Embedding(n_class, 16 * chn, scope='embed')
      #self.embed.weight.data.uniform_(-0.1, 0.1) # TODO
      self.embed = SpectralNorm(self.embed)

  def forward(self, input, class_id):
    with self.scope():

      out = self.pre_conv(input)
      out += self.pre_skip(F.avg_pool2d(input, 2))
      out = self.conv(out)
      out = F.relu(out)
      out = nn.view(out, nn.size(out, 0), -1, nn.size(out, -1))
      out = nn.sum(out, 1)
      out = self.linear(out)
      out_linear = nn.squeeze(out, 1)
      embed = self.embed(class_id)

      prod = nn.sum(out * embed, 1)

      return out_linear + prod



class Generator512(nn.Module):
  def __init__(self, code_dim=128, n_class=1000, chn=96, debug=False, scope='Generator', **kwargs):
    super().__init__(scope=scope, **kwargs)
    self.linear = nn.Linear(n_class, 128, bias=False)
    with self.scope():

      if debug:
        chn = 8

      self.first_view = 16 * chn

      with self.scope('G_Z'):
        self.G_linear = SpectralNorm(nn.Linear(16, 4 * 4 * 16 * chn, scope="G_linear"))

      z_dim = code_dim + 16

      self.GBlock = []
      self.GBlock += [GBlock(16 * chn, 16 * chn, n_class=n_class, z_dim=z_dim, index=0)]
      self.GBlock += [GBlock(16 * chn, 8 * chn, n_class=n_class, z_dim=z_dim, index=1)]
      self.GBlock += [GBlock(8 * chn, 8 * chn, n_class=n_class, z_dim=z_dim, index=2)]
      self.GBlock += [GBlock(8 * chn, 4 * chn, n_class=n_class, z_dim=z_dim, index=3)]
      self.sa_id = len(self.GBlock)
      assert self.sa_id == 4
      self.attention = SelfAttention(2 * chn)
      self.GBlock += [GBlock(4 * chn, 2 * chn, n_class=n_class, z_dim=z_dim, index=4)]
      self.GBlock += [GBlock(2 * chn, 1 * chn, n_class=n_class, z_dim=z_dim, index=5)]
      self.GBlock += [GBlock(1 * chn, 1 * chn, n_class=n_class, z_dim=z_dim, index=6)]

      self.num_split = len(self.GBlock) + 1

      self.ScaledCrossReplicaBN = ScaledCrossReplicaBN(1 * chn, eps=1e-4)
      self.colorize = SpectralNorm(nn.Conv2d(1 * chn, 3, [3, 3], padding=1, scope='conv_2d'))
      
      

  def forward(self, input, class_id):
    with self.scope():
      codes = tf.split(input, self.num_split, 1)
      class_emb = self.linear(class_id) # 128

      with self.scope('G_Z'):
        out = self.G_linear(codes[0])
      out = nn.view(out, -1, 4, 4, self.first_view)
      for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
        if i == self.sa_id:
          out = self.attention(out)
        condition = nn.cat([code, class_emb], 1)
        out = GBlock(out, condition)

      out = self.ScaledCrossReplicaBN(out)
      out = F.relu(out)
      out = self.colorize(out)

      return tf.tanh(out)
  


class Discriminator512(nn.Module):
  def __init__(self, n_class=1000, chn=96, debug=False, scope='Discriminator', **kwargs):
    super().__init__(scope=scope, **kwargs)

    with self.scope():

      def conv(in_channel, out_channel, downsample=True, **kwargs):
        return GBlock(in_channel, out_channel, bn=False, upsample=False, downsample=downsample, **kwargs)

      if debug:
        chn = 8
      self.debug = debug

      with self.scope('pre_conv'):
        self.pre_conv = nn.Sequential(
          SpectralNorm(nn.Conv2d(3, 1 * chn, 3, padding=1, index=0)),
          nn.ReLU(index=1),
          SpectralNorm(nn.Conv2d(1 * chn, 1 * chn, 3, padding=1, index=2)),
          nn.AvgPool2d(2, index=3),
        )
      self.pre_skip = SpectralNorm(nn.Conv2d(3, 1 * chn, 1, scope='pre_skip'))

      with self.scope('conv'):
        self.conv = nn.Sequential(
          conv(1 * chn, 1 * chn, downsample=True, index=0),
          conv(1 * chn, 2 * chn, downsample=True, index=1),
          SelfAttention(2 * chn, index=1),
          conv(2 * chn, 2 * chn, downsample=True, index=2),
          conv(2 * chn, 4 * chn, downsample=True, index=3),
          conv(4 * chn, 8 * chn, downsample=True, index=4),
          conv(8 * chn, 8 * chn, downsample=True, index=5),
          conv(8 * chn, 16 * chn, downsample=True, index=6),
          conv(16 * chn, 16 * chn, downsample=False, index=7),
        )

      self.linear = SpectralNorm(nn.Linear(16 * chn, 1, scope='linear'))

      self.embed = nn.Embedding(n_class, 16 * chn, scope='embed')
      #self.embed.weight.data.uniform_(-0.1, 0.1) # TODO
      self.embed = SpectralNorm(self.embed)


  def forward(self, input, class_id):
    with self.scope():

      out = self.pre_conv(input)
      out += self.pre_skip(F.avg_pool2d(input, 2))
      out = self.conv(out)
      out = F.relu(out)
      out = nn.view(out, nn.size(out, 0), -1, nn.size(out, -1))
      out = nn.sum(out, 1)
      out = self.linear(out)
      out_linear = nn.squeeze(out, 1)
      embed = self.embed(class_id)

      prod = nn.sum(out * embed, 1)

      return out_linear + prod



class BigGAN256(nn.Module):
  def __init__(self, scope='module', disc=False, **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.discriminator = Discriminator256() if disc else None
      self.generator = Generator256()
      def ema_getter(getter, name, *args, **kwargs):
        v = name.split('/')[-1]
        if v in ['w', 'b', 'beta', 'gamma']:
          name = name + '/ema_b999900'
        var = getter(name, *args, **kwargs)
        return var
      with tf.variable_scope("", reuse=True, custom_getter=ema_getter):
        self.ema_generator = Generator256()
      


class BigGAN512(nn.Module):
  def __init__(self, scope='module', disc=False, **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.discriminator = Discriminator512() if disc else None
      self.generator = Generator512()
      def ema_getter(getter, name, *args, **kwargs):
        v = name.split('/')[-1]
        if v in ['w', 'b', 'beta', 'gamma']:
          name = name + '/ema_b999900'
        var = getter(name, *args, **kwargs)
        return var
      with tf.variable_scope("", reuse=True, custom_getter=ema_getter):
        self.ema_generator = Generator512()






# BigGAN-deep: uses a different resblock and pattern


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.

class DeepGBlock(nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      z_dim,
      activation=F.relu,
      upsample=None,
      channel_ratio=4,
      scope='GBlock',
      **kwargs):
    super().__init__(scope=scope, **kwargs)
    if upsample is None and in_channels != out_channels:
      upsample = True
    with self.scope():
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.hidden_channels = self.in_channels // channel_ratio
      def conv(in_channel, out_channel, **kwargs):
        return SpectralNorm(nn.Conv2d(in_channel, out_channel, **kwargs))
      def bn(in_channel, **kwargs):
        return ConditionalBatchNorm2d(in_channel, z_dim, scope='BatchNorm', bn_scope='BatchNorm', gamma_scope='scale', beta_scope='offset', **kwargs)
      self.activation = activation
      self.bn1 = bn(self.in_channels, index=0)
      self.conv1 = conv(self.in_channels, self.hidden_channels, kernel_size=1, padding=0, scope='conv0')
      self.bn2 = bn(self.hidden_channels, index=1)
      self.conv2 = conv(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, scope='conv1')
      self.bn3 = bn(self.hidden_channels, index=2)
      self.conv3 = conv(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, scope='conv2')
      self.bn4 = bn(self.hidden_channels, index=3)
      self.conv4 = conv(self.hidden_channels, self.out_channels, kernel_size=1, padding=0, scope='conv3')
      # upsample layers
      if upsample is True:
        self.upsample = F.upsample
      else:
        self.upsample = upsample

  def forward(self, x, y):
    with self.scope():
      # Project down to channel ratio
      h = self.conv1(self.activation(self.bn1(x, y)))
      # Apply next BN-ReLU
      h = self.activation(self.bn2(h, y))
      if self.upsample:
        h = self.upsample(h)
      # 3x3 convs
      h = self.conv2(h)
      h = self.conv3(self.activation(self.bn3(h, y)))
      # Final 1x1 conv
      h = self.conv4(self.activation(self.bn4(h, y)))
      # Drop channels in x if necessary
      if self.in_channels != self.out_channels:
        x = x[:, :, :, :self.out_channels]
      if self.upsample:
        x = self.upsample(x)
      h = h + x
      return h


class DeepGenerator512(nn.Module):
  def __init__(self, dim_z=128, shared_dim=0, n_class=1000, chn=128, hier=True, debug=False, scope='Generator', **kwargs):
    super().__init__(scope=scope, **kwargs)
    self.hier = hier
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    self.linear = nn.Linear(n_class, 128, bias=False)
    with self.scope():

      if debug:
        chn = 8

      self.first_view = 16 * chn

      z_dim = dim_z + self.shared_dim

      with self.scope('GenZ'):
        self.G_linear = SpectralNorm(nn.Linear(z_dim, 4 * 4 * 16 * chn, scope="G_linear"))

      def conv(in_channel, out_channel, *, index, **kwargs):
        if index % 2 == 0:
          upsample = False
        else:
          upsample = True
        return DeepGBlock(in_channel, out_channel, z_dim=z_dim, upsample=upsample, index=index, **kwargs)


      self.conv = nn.Sequential(
        conv(16 * chn, 16 * chn, index=0),
        conv(16 * chn, 16 * chn, index=1),
        conv(16 * chn, 16 * chn, index=2),
        conv(16 * chn, 8 * chn, index=3),
        conv(8 * chn, 8 * chn, index=4),
        conv(8 * chn, 8 * chn, index=5),
        conv(8 * chn, 8 * chn, index=6),
        conv(8 * chn, 4 * chn, index=7),
        SelfAttention(4 * chn),
        conv(4 * chn, 4 * chn, index=8),
        conv(4 * chn, 2 * chn, index=9),
        conv(2 * chn, 2 * chn, index=10),
        conv(2 * chn, 1 * chn, index=11),
        conv(1 * chn, 1 * chn, index=12),
        conv(1 * chn, 1 * chn, index=13),
      )

      self.ScaledCrossReplicaBN = ScaledCrossReplicaBN(1 * chn, eps=1e-4, scope='BatchNorm', bn_scope_suffix='', scale_name='scale', offset_name='offset')
      self.colorize = SpectralNorm(nn.Conv2d(1 * chn, chn, [3, 3], padding=1, scope='conv_to_rgb'))
      
      

  def forward(self, z, y):
    with self.scope():
      y = self.linear(y) # 128

      # If hierarchical, concatenate zs and ys
      if self.hier:
        z = nn.cat([z, y], 1)
        y = z

      with self.scope('GenZ'):
        h = self.G_linear(z)
      h = nn.view(h, nn.size(h, 0), 4, 4, -1)

      h = self.conv(h, y)

      h = self.ScaledCrossReplicaBN(h)
      h = F.relu(h)
      h = self.colorize(h)

      # take rgb channels
      h = h[:, :, :, :3]

      return tf.tanh(h)




class BigGANDeep512(nn.Module):
  def __init__(self, scope='module', disc=False, ema=True, **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.discriminator = DeepDiscriminator512() if disc else None
      self.generator = DeepGenerator512()
      if ema:
        def ema_getter(getter, name, *args, **kwargs):
          v = name.split('/')[-1]
          if v in ['w', 'b', 'scale', 'offset', 'gamma']:
            name = name + '/ema_0.9999'
          var = getter(name, *args, **kwargs)
          return var
        with tf.variable_scope("", reuse=True, custom_getter=ema_getter):
          self.ema_generator = DeepGenerator512()
      else:
        self.ema_generator = self.generator





assert_shape = nn.assert_shape
shapelist = nn.shapelist


class SpectralNorm(nn.Module):
  def __init__(self, module, name='weight', epsilon=9.999999747378752e-05, update=None, scope=None, **kwargs):
    super().__init__(scope=scope, **kwargs)
    self.module = module
    self.name = name
    self.epsilon = epsilon
    self.update = update
    self.update = False
    self._make_params()

  def should_update(self):
    if self.update is False:
      return False
    if self.training:
      return True
    return False


  def _make_params(self):
    w = getattr(self.module, self.name)
    assert len(shapelist(w)) > 1
    shape = shapelist(w)
    ushape = [1, shape[-1]]
    with self.module.scope():
      self.u0 = self.module.localvar('u0', dtype=w.dtype, shape=ushape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0), collections=['variables'])
      self.u1 = self.module.localvar('u1', dtype=w.dtype, shape=ushape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0), collections=['variables'])
      self.u2 = self.module.localvar('u2', dtype=w.dtype, shape=ushape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0), collections=['variables'])
      return w

  def _update(self):
    if not hasattr(self, 'u0'):
      w = self._make_params()
    epsilon = self.epsilon
    w = getattr(self.module, self.name)
    w = val(w)
    assert len(shapelist(w)) > 1
    shape = shapelist(w)
    ushape = [1, shape[-1]]
    w_reshaped = tf.reshape(w, [-1, shape[-1]])
    wshape = shapelist(w_reshaped)
    vshape = [1, wshape[0]]
    u0 = self.u0
    u1 = self.u1
    u2 = self.u2
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>]}],
    x0 = tf.matmul(u0, w_reshaped, transpose_b=True)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow/y:0' shape=() dtype=float32>]},
    #     {'value': 2.0}],
    # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow' type=Pow>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow:0' shape=(1, 20) dtype=float32>]}],
    x1 = tf.pow(x0, 2.0)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const:0' shape=(2,) dtype=int32>]},
    #     {'value': array([0, 1], dtype=int32)}],
    # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum' type=Sum>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const:0' shape=(2,) dtype=int32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum:0' shape=(1, 1) dtype=float32>]}],
    x2 = tf.reduce_sum(x1, axis=[0, 1], keepdims=True)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add/y:0' shape=() dtype=float32>]},
    #     {'value': 9.999999747378752e-05}],
    # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add' type=Add>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add:0' shape=(1, 1) dtype=float32>]}],
    x3 = tf.add(x2, epsilon)
    # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt' type=Rsqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt:0' shape=(1, 1) dtype=float32>]}],
    x4 = tf.math.rsqrt(x3)
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>]}],
    x5 = tf.multiply(x0, x4)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_1' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>]}],
    x6 = tf.matmul(x5, w_reshaped)
    assert_shape(x6, ushape)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_1/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1/y:0' shape=() dtype=float32>]},
    #     {'value': 2.0}],
    # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_1' type=Pow>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1:0' shape=(1, 24576) dtype=float32>]}],
    x7 = tf.pow(x6, 2.0)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_1' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_1:0' shape=(2,) dtype=int32>]},
    #     {'value': array([0, 1], dtype=int32)}],
    # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_1' type=Sum>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_1:0' shape=(2,) dtype=int32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_1:0' shape=(1, 1) dtype=float32>]}],
    x8 = tf.reduce_sum(x7, axis=[0, 1], keepdims=True)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_1/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1/y:0' shape=() dtype=float32>]},
    #     {'value': 9.999999747378752e-05}],
    # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_1' type=Add>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_1:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1:0' shape=(1, 1) dtype=float32>]}],
    x9 = tf.add(x8, epsilon)
    # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_1' type=Rsqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_1:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_1:0' shape=(1, 1) dtype=float32>]}],
    x10 = tf.math.rsqrt(x9)
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_1' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_1:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>]}],
    x11 = tf.multiply(x6, x10)
    # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1' type=ReadVariableOp>,
    #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1:0' shape=(1, 24576) dtype=float32>]}],
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_2' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>]}],
    x12 = tf.matmul(u1, w_reshaped, transpose_b=True)
    assert_shape(x12, vshape)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_3' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_3:0' shape=(1, 1) dtype=float32>]}],
    x13 = tf.matmul(x5, x12, transpose_b=True)
    assert_shape(x13, [1, 1])
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_2' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_3:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_2:0' shape=(1, 20) dtype=float32>]}],
    x14 = tf.multiply(x13, x5)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_4' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_4:0' shape=(1, 1) dtype=float32>]}],
    x15 = tf.matmul(x5, x5, transpose_b=True)
    assert_shape(x15, [1, 1])
    # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv' type=RealDiv>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_2:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_4:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv:0' shape=(1, 20) dtype=float32>]}],
    x16 = tf.div(x14, x15)
    # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub' type=Sub>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_2:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>]}],
    x17 = tf.subtract(x12, x16)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_2/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2/y:0' shape=() dtype=float32>]},
    #     {'value': 2.0}],
    # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_2' type=Pow>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2:0' shape=(1, 20) dtype=float32>]}],
    x18 = tf.pow(x17, 2.0)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_2' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_2:0' shape=(2,) dtype=int32>]},
    #     {'value': array([0, 1], dtype=int32)}],
    # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_2' type=Sum>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_2:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_2:0' shape=(2,) dtype=int32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_2:0' shape=(1, 1) dtype=float32>]}],
    x19 = tf.reduce_sum(x18, axis=[0, 1], keepdims=True)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_2/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2/y:0' shape=() dtype=float32>]},
    #     {'value': 9.999999747378752e-05}],
    # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_2' type=Add>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_2:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2:0' shape=(1, 1) dtype=float32>]}],
    x20 = tf.add(x19, epsilon)
    # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_2' type=Rsqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_2:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_2:0' shape=(1, 1) dtype=float32>]}],
    x21 = tf.math.rsqrt(x20)
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_3' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_2:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>]}],
    x22 = tf.multiply(x17, x21)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_5' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>]}],
    x23 = tf.matmul(x22, w_reshaped)
    assert_shape(x23, ushape)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_6' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_6:0' shape=(1, 1) dtype=float32>]}],
    x24 = tf.matmul(x11, x23, transpose_b=True)
    assert_shape(x24, [1, 1])
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_4' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_6:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_4:0' shape=(1, 24576) dtype=float32>]}],
    x25 = tf.multiply(x24, x11)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_7' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_7:0' shape=(1, 1) dtype=float32>]}],
    x26 = tf.matmul(x11, x11, transpose_b=True)
    assert_shape(x26, [1, 1])
    # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_1' type=RealDiv>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_4:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_7:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_1:0' shape=(1, 24576) dtype=float32>]}],
    x27 = tf.div(x25, x26)
    # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_1' type=Sub>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_5:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_1:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>]}],
    x28 = tf.subtract(x23, x27)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_3/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3/y:0' shape=() dtype=float32>]},
    #     {'value': 2.0}],
    # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_3' type=Pow>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3:0' shape=(1, 24576) dtype=float32>]}],
    x29 = tf.pow(x28, 2.0)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_3' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_3:0' shape=(2,) dtype=int32>]},
    #     {'value': array([0, 1], dtype=int32)}],
    # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_3' type=Sum>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_3:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_3:0' shape=(2,) dtype=int32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_3:0' shape=(1, 1) dtype=float32>]}],
    x30 = tf.reduce_sum(x29, axis=[0, 1], keepdims=True)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_3/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3/y:0' shape=() dtype=float32>]},
    #     {'value': 9.999999747378752e-05}],
    # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_3' type=Add>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_3:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3:0' shape=(1, 1) dtype=float32>]}],
    x31 = tf.add(x30, epsilon)
    # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_3' type=Rsqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_3:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_3:0' shape=(1, 1) dtype=float32>]}],
    x32 = tf.math.rsqrt(x31)
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_5' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_3:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>]}],
    x33 = tf.multiply(x28, x32)
    # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2' type=ReadVariableOp>,
    #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2:0' shape=(1, 24576) dtype=float32>]}],
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_8' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_2:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>]}],
    x34 = tf.matmul(u2, w_reshaped, transpose_b=True)
    assert_shape(x34, vshape)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_9' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_9:0' shape=(1, 1) dtype=float32>]}],
    x35 = tf.matmul(x5, x34, transpose_b=True)
    assert_shape(x35, [1, 1])
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_6' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_9:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_6:0' shape=(1, 20) dtype=float32>]}],
    x36 = tf.multiply(x35, x5)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_10' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_10:0' shape=(1, 1) dtype=float32>]}],
    x37 = tf.matmul(x5, x5, transpose_b=True)
    assert_shape(x37, [1, 1])
    # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_2' type=RealDiv>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_6:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_10:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_2:0' shape=(1, 20) dtype=float32>]}],
    x38 = tf.div(x36, x37)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_11' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_11:0' shape=(1, 1) dtype=float32>]}],
    x39 = tf.matmul(x22, x34, transpose_b=True)
    assert_shape(x39, [1, 1])
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_7' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_11:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_7:0' shape=(1, 20) dtype=float32>]}],
    x40 = tf.multiply(x39, x22)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_12' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_12:0' shape=(1, 1) dtype=float32>]}],
    x41 = tf.matmul(x22, x22, transpose_b=True)
    assert_shape(x41, [1, 1])
    # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_3' type=RealDiv>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_7:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_12:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_3:0' shape=(1, 20) dtype=float32>]}],
    x42 = tf.div(x40, x41)
    # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_4' type=Add>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_2:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_3:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_4:0' shape=(1, 20) dtype=float32>]}],
    x43 = tf.add(x38, x42)
    # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_2' type=Sub>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_8:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_4:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>]}],
    x44 = tf.subtract(x34, x43)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_4/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4/y:0' shape=() dtype=float32>]},
    #     {'value': 2.0}],
    # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_4' type=Pow>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4:0' shape=(1, 20) dtype=float32>]}],
    x45 = tf.pow(x44, 2.0)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_4' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_4:0' shape=(2,) dtype=int32>]},
    #     {'value': array([0, 1], dtype=int32)}],
    # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_4' type=Sum>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_4:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_4:0' shape=(2,) dtype=int32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_4:0' shape=(1, 1) dtype=float32>]}],
    x46 = tf.reduce_sum(x45, axis=[0, 1], keepdims=True)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_5/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5/y:0' shape=() dtype=float32>]},
    #     {'value': 9.999999747378752e-05}],
    # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_5' type=Add>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_4:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5:0' shape=(1, 1) dtype=float32>]}],
    x47 = tf.add(x46, epsilon)
    # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_4' type=Rsqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_5:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_4:0' shape=(1, 1) dtype=float32>]}],
    x48 = tf.math.rsqrt(x47)
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_8' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_2:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_4:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>]}],
    x49 = tf.multiply(x44, x48)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_13' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>]}],
    x50 = tf.matmul(x49, w_reshaped)
    assert_shape(x50, ushape)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_14' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_14:0' shape=(1, 1) dtype=float32>]}],
    x51 = tf.matmul(x11, x50, transpose_b=True)
    assert_shape(x51, [1, 1])
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_9' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_14:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_9:0' shape=(1, 24576) dtype=float32>]}],
    x52 = tf.multiply(x51, x11)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_15' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_15:0' shape=(1, 1) dtype=float32>]}],
    x53 = tf.matmul(x11, x11, transpose_b=True)
    assert_shape(x53, [1, 1])
    # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_4' type=RealDiv>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_9:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_15:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_4:0' shape=(1, 24576) dtype=float32>]}],
    x54 = tf.div(x52, x53)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_16' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_16:0' shape=(1, 1) dtype=float32>]}],
    x55 = tf.matmul(x33, x50, transpose_b=True)
    assert_shape(x55, [1, 1])
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_10' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_16:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_10:0' shape=(1, 24576) dtype=float32>]}],
    x56 = tf.multiply(x55, x33)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_17' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_17:0' shape=(1, 1) dtype=float32>]}],
    x57 = tf.matmul(x33, x33, transpose_b=True)
    assert_shape(x57, [1, 1])
    # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_5' type=RealDiv>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_10:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_17:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_5:0' shape=(1, 24576) dtype=float32>]}],
    x58 = tf.div(x56, x57)
    # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_6' type=Add>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_4:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_5:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_6:0' shape=(1, 24576) dtype=float32>]}],
    x59 = tf.add(x54, x58)
    # ['Sub', <tf.Operation 'module/Generator_1/G_Z/G_linear/sub_3' type=Sub>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_13:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_6:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>]}],
    x60 = tf.subtract(x50, x59)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_5/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5/y:0' shape=() dtype=float32>]},
    #     {'value': 2.0}],
    # ['Pow', <tf.Operation 'module/Generator_1/G_Z/G_linear/Pow_5' type=Pow>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5:0' shape=(1, 24576) dtype=float32>]}],
    x61 = tf.pow(x60, 2.0)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/Const_5' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_5:0' shape=(2,) dtype=int32>]},
    #     {'value': array([0, 1], dtype=int32)}],
    # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sum_5' type=Sum>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Pow_5:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Const_5:0' shape=(2,) dtype=int32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_5:0' shape=(1, 1) dtype=float32>]}],
    x62 = tf.reduce_sum(x61, axis=[0, 1], keepdims=True)
    # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_7/y' type=Const>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7/y:0' shape=() dtype=float32>]},
    #     {'value': 9.999999747378752e-05}],
    # ['Add', <tf.Operation 'module/Generator_1/G_Z/G_linear/add_7' type=Add>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sum_5:0' shape=(1, 1) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7/y:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7:0' shape=(1, 1) dtype=float32>]}],
    x63 = tf.add(x62, epsilon)
    # ['Rsqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Rsqrt_5' type=Rsqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/add_7:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_5:0' shape=(1, 1) dtype=float32>]}],
    x64 = tf.math.rsqrt(x63)
    # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/mul_11' type=Mul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/sub_3:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Rsqrt_5:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>]}],
    x65 = tf.multiply(x60, x64)
    # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient' type=StopGradient>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_1:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>]}],
    x66 = tf.stop_gradient(x11)
    # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient_1' type=StopGradient>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul:0' shape=(1, 20) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_1:0' shape=(1, 20) dtype=float32>]}],
    x67 = tf.stop_gradient(x5)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_18' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_1:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_18:0' shape=(1, 24576) dtype=float32>]}],
    x68 = tf.matmul(x67, w_reshaped)
    assert_shape(x68, ushape)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_19' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_18:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_19:0' shape=(1, 1) dtype=float32>]}],
    x69nice = tf.matmul(x68, x66, transpose_b=True)
    assert_shape(x69nice, [1, 1])
    # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze' type=Squeeze>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_19:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze:0' shape=() dtype=float32>]}],
    x70 = tf.squeeze(x69nice)
    assert_shape(x70, [])
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_20' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_3:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_20:0' shape=(1, 24576) dtype=float32>]}],
    x71 = tf.matmul(x22, w_reshaped)
    assert_shape(x71, ushape)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_21' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_20:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_21:0' shape=(1, 1) dtype=float32>]}],
    x72 = tf.matmul(x71, x33, transpose_b=True)
    assert_shape(x72, [1, 1])
    # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze_1' type=Squeeze>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_21:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze_1:0' shape=() dtype=float32>]}],
    x73 = tf.squeeze(x72)
    assert_shape(x73, [])
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_22' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_8:0' shape=(1, 20) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_22:0' shape=(1, 24576) dtype=float32>]}],
    x74 = tf.matmul(x49, w_reshaped)
    assert_shape(x74, ushape)
    # ['MatMul', <tf.Operation 'module/Generator_1/G_Z/G_linear/MatMul_23' type=MatMul>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_22:0' shape=(1, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_23:0' shape=(1, 1) dtype=float32>]}],
    x75 = tf.matmul(x74, x65, transpose_b=True)
    assert_shape(x75, [1, 1])
    # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/Squeeze_2' type=Squeeze>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/MatMul_23:0' shape=(1, 1) dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze_2:0' shape=() dtype=float32>]}],
    x76 = tf.squeeze(x75)
    assert_shape(x76, [])
    with tf.name_scope('norm'):
      # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/mul' type=Mul>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/mul:0' shape=(20, 24576) dtype=float32>]}],
      x77 = tf.multiply(w_reshaped, w_reshaped)
      # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Const' type=Const>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Const:0' shape=(2,) dtype=int32>]},
      #     {'value': array([0, 1], dtype=int32)}],
      # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Sum' type=Sum>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/mul:0' shape=(20, 24576) dtype=float32>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Const:0' shape=(2,) dtype=int32>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sum:0' shape=(1, 1) dtype=float32>]}],
      x78 = tf.reduce_sum(x77, axis=[0, 1], keepdims=True)
      # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Sqrt' type=Sqrt>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sum:0' shape=(1, 1) dtype=float32>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sqrt:0' shape=(1, 1) dtype=float32>]}],
      x79 = tf.math.sqrt(x78)
      # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm/Squeeze' type=Squeeze>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Sqrt:0' shape=(1, 1) dtype=float32>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Squeeze:0' shape=() dtype=float32>]}],
      x80 = tf.squeeze(x79)
      assert_shape(x80, [])
    # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/Sqrt' type=Sqrt>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm/Squeeze:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/Sqrt:0' shape=() dtype=float32>]}],
    x81 = tf.math.sqrt(x80)
    # ['StopGradient', <tf.Operation 'module/Generator_1/G_Z/G_linear/StopGradient_2' type=StopGradient>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Sqrt:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient_2:0' shape=() dtype=float32>]}],
    x82 = tf.stop_gradient(x81)
    with tf.name_scope('norm'):
      # ['Mul', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/mul' type=Mul>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/mul:0' shape=(20, 24576) dtype=float32>]}],
      x83 = tf.multiply(w_reshaped, w_reshaped)
      # ['Const', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Const' type=Const>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Const:0' shape=(2,) dtype=int32>]},
      #     {'value': array([0, 1], dtype=int32)}],
      # ['Sum', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Sum' type=Sum>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/mul:0' shape=(20, 24576) dtype=float32>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Const:0' shape=(2,) dtype=int32>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sum:0' shape=(1, 1) dtype=float32>]}],
      x84 = tf.reduce_sum(x83, axis=[0, 1], keepdims=True)
      # ['Sqrt', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt' type=Sqrt>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sum:0' shape=(1, 1) dtype=float32>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt:0' shape=(1, 1) dtype=float32>]}],
      x85 = tf.math.sqrt(x84)
      # ['Squeeze', <tf.Operation 'module/Generator_1/G_Z/G_linear/norm_1/Squeeze' type=Squeeze>,
      #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Sqrt:0' shape=(1, 1) dtype=float32>,
      #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/norm_1/Squeeze:0' shape=() dtype=float32>]}],
      x86 = tf.squeeze(x85)
    # ['RealDiv', <tf.Operation 'module/Generator_1/G_Z/G_linear/truediv_6' type=RealDiv>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Reshape:0' shape=(20, 24576) dtype=float32>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/Squeeze:0' shape=() dtype=float32>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/truediv_6:0' shape=(20, 24576) dtype=float32>]}],
    x87 = tf.div(w_reshaped, x70)
    # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp' type=AssignVariableOp>,
    #     <tf.Tensor 'module/Generator/G_Z/G_linear/u0:0' shape=() dtype=resource>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/StopGradient:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': []}],
    x88 = u0.assign(x66, read_value=False) if self.should_update() else x66
    # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_3' type=ReadVariableOp>,
    #     <tf.Tensor 'module/Generator/G_Z/G_linear/u0:0' shape=() dtype=resource>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_3:0' shape=(1, 24576) dtype=float32>]}],
    with tf.control_dependencies([x88]):
      x89 = u0.read_value()
    # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp_1' type=AssignVariableOp>,
    #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_5:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': []}],
    x90 = u1.assign(x33, read_value=False) if self.should_update() else x33
    # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_4' type=ReadVariableOp>,
    #     <tf.Tensor 'module/Generator/G_Z/G_linear/u1:0' shape=() dtype=resource>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_4:0' shape=(1, 24576) dtype=float32>]}],
    with tf.control_dependencies([x90]):
      x91 = u1.read_value()
    # ['AssignVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/AssignVariableOp_2' type=AssignVariableOp>,
    #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
    #     <tf.Tensor 'module/Generator_1/G_Z/G_linear/mul_11:0' shape=(1, 24576) dtype=float32>,
    #     {'outputs': []}],
    x92 = u2.assign(x65, read_value=False) if self.should_update() else x65
    # ['ReadVariableOp', <tf.Operation 'module/Generator_1/G_Z/G_linear/ReadVariableOp_5' type=ReadVariableOp>,
    #     <tf.Tensor 'module/Generator/G_Z/G_linear/u2:0' shape=() dtype=resource>,
    #     {'outputs': [<tf.Tensor 'module/Generator_1/G_Z/G_linear/ReadVariableOp_5:0' shape=(1, 24576) dtype=float32>]}],
    with tf.control_dependencies([x92]):
      x93 = u2.read_value()
    with tf.control_dependencies([x88, x90, x92]):
      norm = tf.identity(x70, "norm")
    w_normalized = tf.div(w_reshaped, norm, name="truediv")
    w_normalized = tf.reshape(w_normalized, shape)
    # if return_norm:
    #   return w_normalized, norm
    # return w_normalized
    setattr(self.module, self.name, w_normalized)

  def forward(self, *args):
    #import pdb; pdb.set_trace()
    self._update()
    return self.module.forward(*args)
  
