
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint as pp
from contextlib import contextmanager

import sys
import os
import re
import six
from six.moves.urllib.error import URLError

from tensorflow.python import framework
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver
from tensorflow.contrib.cluster_resolver import TPUClusterResolver as BaseTPUClusterResolver
from tensorflow.python.eager.context import LogicalDevice
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

mock = test.mock

def reroute(addr, host=None):
  if host is None or host is False:
    return addr
  if addr.startswith('grpc://'):
    return 'grpc://' + reroute(addr[len('grpc://'):], host=host)
  if not re.match('[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+[:]8470', addr):
    return addr
  if not addr.endswith(':8470'):
    return addr
  a, b, c, d = [int(x) for x in addr.split(':')[0].split('.')]
  if a == 10 and b in [48, 49]:
    assert (d == 2)
    port = b * 1000 + c
  elif a == 10 and b in range(2, 66) and c == 0:
    port = b * 1000 + d
  else:
    return addr
  return host + ':' + str(port)


class TPUClusterResolver(BaseTPUClusterResolver):
  def __init__(self, *args, host=None, node_count=None, node_offset=None, **kws):
    super(TPUClusterResolver, self).__init__(*args, **kws)
    if host is None:
      host = _tpu_host()
    self._host = host
    if node_count is None:
      if 'TPU_NODE_COUNT' in os.environ:
        node_count = int(os.environ['TPU_NODE_COUNT'])
    self._node_count = node_count
    if node_offset is None:
      if 'TPU_NODE_OFFSET' in os.environ:
        node_offset = int(os.environ['TPU_NODE_OFFSET'])
    self._node_offset = node_offset

  def master(self, *args, **kws):
    ip = super(TPUClusterResolver, self).master(*args, **kws)
    return reroute(ip, host=self._host)

  def cluster_spec(self):
    spec = super(TPUClusterResolver, self).cluster_spec()
    r = dict()
    for k, v in spec.as_dict().items():
      r[k] = [reroute(ip, host=self._host) for ip in v]
    i = self._node_count or len(r['worker'])
    j = self._node_offset or 0
    r['worker'] = [r['worker'][0]] + r['worker'][(j+1):(j+1)+(i-1)]
    spec2 = server_lib.ClusterSpec(r)
    print(spec2.as_cluster_def())
    return spec2



_master = resolver.TPUClusterResolver.master

def _tpu_host():
  return os.environ.get('TPU_HOST', '10.255.128.3')

def mock_master(cls, *args, **kws):
  ip = _master(cls, *args, **kws)
  return reroute(ip, host=os.environ['TPU_HOST'])

_cluster_spec = resolver.TPUClusterResolver.cluster_spec

def cluster_spec(cls, *args, **kws):
  spec = _cluster_spec(cls, *args, **kws)
  r = dict()
  for k, v in spec.as_dict().items():
    r[k] = [reroute(ip, host=os.environ['TPU_HOST']) for ip in v]
  return server_lib.ClusterSpec(r)

__fetch_cloud_tpu_metadata = resolver.TPUClusterResolver._fetch_cloud_tpu_metadata

def _fetch_cloud_tpu_metadata(cls, *args, **kws):
  while True:
    try:
      return __fetch_cloud_tpu_metadata(cls, *args, **kws)
    except Exception as e:
      if '[Errno 111] Connection refused' in str(e):
        # retry
        import time
        time.sleep(1.0)
      else:
        raise e

@contextmanager
def patch_tensorflow():
  with mock.patch.object(resolver.TPUClusterResolver, 'master', mock_master):
    with mock.patch.object(resolver.TPUClusterResolver, 'cluster_spec', cluster_spec):
      with mock.patch.object(resolver.TPUClusterResolver, '_fetch_cloud_tpu_metadata', _fetch_cloud_tpu_metadata):
        result = yield
        return result

def patch_tensorflow_interactive():
  patch = patch_tensorflow()
  patch.__enter__()
  return patch


def interact():
    import code
    code.InteractiveConsole(locals=globals()).interact()

if __name__ == '__main__':
  _tf_patch = patch_tensorflow_interactive()
  if len(sys.argv) <= 1:
    from tensorflow.core.protobuf import config_pb2
    import tensorflow as tf
    import numpy as np
    session_config = config_pb2.ConfigProto(allow_soft_placement=True, isolate_session_state=True)
    master = None
    res = None
    cluster_spec = None
    cluster_def = None
    if 'TPU_NAME' in os.environ:
      res = resolver.TPUClusterResolver(os.environ['TPU_NAME'])
      master = res.get_master()
      cluster_spec = res.cluster_spec()
      if cluster_spec:
        cluster_def = cluster_spec.as_cluster_def()
        session_config.cluster_def.CopyFrom(cluster_def)
    graph = tf.Graph()
    sess = tf.compat.v1.InteractiveSession(master, graph=graph, config=session_config)
    devices = sess.list_devices()
    num_cores = len([x for x in devices if ':TPU:' in x.name])
    print(cluster_def)
    print('ip: %s', master, num_cores)
    r = sess.run
    from tensorflow.python.tpu import tpu as tpu_ops
    from tensorflow.compiler.tf2xla.python import xla
    from tensorflow.compiler.tf2xla.ops import gen_xla_ops
  else:
    filename = sys.argv[1]
    sys.argv = sys.argv[1:]
    with open(filename) as f:
      source = f.read()
    code = compile(source, filename, 'exec')
    exec(code, globals(), globals())

