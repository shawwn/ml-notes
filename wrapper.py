
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint as pp

import sys
import os
import re
import six
from six.moves.urllib.error import URLError

from tensorflow.python import framework
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver
from tensorflow.python.eager.context import LogicalDevice
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

mock = test.mock

def mock_request_compute_metadata(cls, *args, **kwargs):
  del cls, kwargs  # Unused.
  if args[0] == 'project/project-id':
    #return 'test-project'
    return 'gpt-2-15b-poetry'
  elif args[0] == 'instance/zone':
    #return 'projects/test-project/locations/us-central1-c'
    return 'projects/gpt-2-15b-poetry/locations/europe-west4-a'
  elif args[0] == 'instance/network-interfaces/0/ip':
    #return '10.128.1.2'
    return '127.0.0.1'
  return ''

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

if __name__ == '__main__':
  with mock.patch.object(resolver.TPUClusterResolver, 'master', mock_master):
    with mock.patch.object(resolver.TPUClusterResolver, 'cluster_spec', cluster_spec):
      with mock.patch.object(resolver.TPUClusterResolver, '_fetch_cloud_tpu_metadata', _fetch_cloud_tpu_metadata):
        if len(sys.argv) <= 1:
          from tensorflow.core.protobuf import config_pb2
          import tensorflow as tf
          import numpy as np
          session_config = config_pb2.ConfigProto(allow_soft_placement=True, isolate_session_state=True)
          res = resolver.TPUClusterResolver(os.environ['TPU_NAME'])
          cluster_spec = res.cluster_spec()
          if cluster_spec:
            cluster = cluster_spec.as_cluster_def()
            session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
          else:
            cluster = None
          master = res.get_master()
          print(master, cluster)
          graph = tf.Graph()
          sess = tf.compat.v1.InteractiveSession(master, graph=graph, config=session_config)
          import pdb
          pdb.set_trace()
        else:
          filename = sys.argv[1]
          sys.argv = sys.argv[1:]
          with open(filename) as f:
            source = f.read()
          code = compile(source, filename, 'exec')
          exec(code, globals(), globals())

