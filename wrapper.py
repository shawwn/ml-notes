
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint as pp
from pprint import pformat as pf
from contextlib import contextmanager

import sys
import os
import re
import six
import json
import base64
from six.moves.urllib.error import URLError

from tensorflow.python import framework
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver
from tensorflow.compat.v1.distribute.cluster_resolver import TPUClusterResolver as BaseTPUClusterResolver
from tensorflow.python.eager.context import LogicalDevice
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.tpu import topology as topology_lib

import gin

try:
  from cloud_tpu_client import client  # pylint: disable=g-import-not-at-top
except ImportError:
  try:
    logging.debug(
        'Falling back to TensorFlow client; we recommended you install the Cloud '
        'TPU client directly with pip install cloud-tpu-client.')
    from tensorflow.python.tpu.client import client  # pylint: disable=g-import-not-at-top
  except ImportError:
    client = None


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
    kws['project'] = kws.pop('project', 'gpt-2-15b-poetry')
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


from six.moves.urllib import request

def _as_text(s):
  if isinstance(s, bytes):
    return s.decode('utf-8')
  return s

def _request_compute_metadata(path):
  _GCE_METADATA_ENDPOINT = 'http://35.225.160.61'
  req = request.Request(
      '%s/computeMetadata/v1/%s' % (_GCE_METADATA_ENDPOINT, path),
      headers={'Metadata-Flavor': 'Google'})
  resp = request.urlopen(req)
  return _as_text(resp.read())

# cli = client.Client(tpu=os.environ['TPU_NAME'])
# service = cli._tpu_service()
# info = service.projects().locations().nodes().get(name=cli._full_name().replace(os.environ['TPU_NAME'], 'tpu-v2-8-usc1f-0')).execute()
# {'name': 'projects/gpt-2-15b-poetry/locations/us-central1-f/nodes/tpu-v2-8-usc1f-0', 'acceleratorType': 'v2-8', 'ipAddress': '10.48.0.2', 'state': 'READY', 'tensorflowVersion': '2.3', 'network': 'global/networks/tpu-usc1f', 'cidrBlock': '10.48.0.0/29', 'port': '8470', 'serviceAccount': 'service-41076153887@cloud-tpu.iam.gserviceaccount.com', 'createTime': '2020-09-18T07:21:45.237850246Z', 'schedulingConfig': {'preemptible': True}, 'networkEndpoints': [{'ipAddress': '10.48.0.2', 'port': 8470}], 'health': 'HEALTHY'}


_master = resolver.TPUClusterResolver.master

def _tpu_host():
  return os.environ.get('TPU_HOST', '10.255.128.3')

def mock_master(cls, *args, **kws):
  ip = _master(cls, *args, **kws)
  return reroute(ip, host=os.environ.get('TPU_HOST', None))

_cluster_spec = resolver.TPUClusterResolver.cluster_spec

def cluster_spec(cls, *args, **kws):
  spec = _cluster_spec(cls, *args, **kws)
  r = dict()
  for k, v in spec.as_dict().items():
    r[k] = [reroute(ip, host=os.environ.get('TPU_HOST', None)) for ip in v]
  return server_lib.ClusterSpec(r)


__fetch_cloud_tpu_metadata = (client.Client if client is not None else resolver.TPUClusterResolver)._fetch_cloud_tpu_metadata

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


__parse_topology = topology_lib.Topology._parse_topology

def _parse_topology(self, serialized):
    """Parses a serialized `TopologyProto` into `self`."""
    proto = topology_pb2.TopologyProto()
    proto.ParseFromString(serialized)

    self._mesh_shape = np.array(proto.mesh_shape, dtype=np.int32)
    if len(self._mesh_shape) != 4 or any(self._mesh_shape < 1):
      raise ValueError("`mesh_shape` must be a vector of size 4 with positive "
                       "entries; got {}".format(self._mesh_shape))

    if proto.num_tasks < 0:
      raise ValueError("`num_tasks` must be >= 0; got {}".format(
          proto.num_tasks))
    if proto.num_tpu_devices_per_task < 0:
      raise ValueError("`num_tpu_devices_per_task` must be >= 0; got {}".format(
          proto.num_tpu_devices_per_task))

    expected_coordinates_size = (
        proto.num_tasks * proto.num_tpu_devices_per_task * len(
            proto.mesh_shape))
    if len(proto.device_coordinates) != expected_coordinates_size:
      raise ValueError("`device_coordinates` must have shape num_tasks ({}) * "
                       "num_tpu_devices_per_task ({}) * len(mesh_shape) ({}); "
                       "got shape {}".format(proto.num_tasks,
                                             proto.num_tpu_devices_per_task,
                                             proto.mesh_shape,
                                             len(proto.device_coordinates)))

    coords = np.array(proto.device_coordinates, dtype=np.int32)
    if any(coords < 0):
      raise ValueError("`device_coordinates` must be >= 0")
    coords = coords.reshape((proto.num_tasks, proto.num_tpu_devices_per_task,
                             len(proto.mesh_shape)))
    self._device_coordinates = coords
  

__invert_topology = topology_lib.Topology._invert_topology

def _invert_topology(self):
  """Inverts a [task,device,axis] topology to [x,y,z] -> task/device maps."""
  tasks = np.full(list(self.mesh_shape), -1, dtype=np.int32)
  devices = np.full(list(self.mesh_shape), -1, dtype=np.int32)
  for task in range(self.device_coordinates.shape[0]):
    for device in range(self.device_coordinates.shape[1]):
      x, y, z, core = self.device_coordinates[task, device, :]
      tasks[x, y, z, core] = task
      devices[x, y, z, core] = device
  return tasks, devices


@contextmanager
def patch_tensorflow():
  with mock.patch.object(resolver.TPUClusterResolver, 'master', mock_master):
    with mock.patch.object(resolver.TPUClusterResolver, 'cluster_spec', cluster_spec):
      with mock.patch.object(client.Client if client is not None else resolver.TPUClusterResolver, '_fetch_cloud_tpu_metadata', _fetch_cloud_tpu_metadata):
        with mock.patch.object(topology_lib.Topology, '_parse_topology', _parse_topology):
          with mock.patch.object(topology_lib.Topology, '_invert_topology', _invert_topology):
            result = yield
            return result


def patch_tensorflow_interactive():
  patch = patch_tensorflow()
  patch.__enter__()
  gin.enter_interactive_mode()
  return patch

def interact():
    import code
    code.InteractiveConsole(locals=globals()).interact()


def clone_session(session=None, graph=None, interactive=False, **kws):
  if session is None:
    session = tf.get_default_session()
  if graph is None:
    graph = session.graph
  config = session._config # is there a better way to do this?
  master = session.sess_str # is there a better way to do this?
  Session = (tf.compat.v1.InteractiveSession if interactive else tf.Session)
  return Session(master, graph=graph, config=config, **kws)


def reset_session(session=None, graph=None, interactive=True, **kws):
  if session is None:
    session = tf.get_default_session()
  if graph is None:
    graph = tf.Graph()
  graph.as_default().__enter__()
  session2 = clone_session(session, graph=graph, interactive=interactive, **kws)
  session2.as_default().__enter__()
  if 'sess' in globals():
    globals()['sess'] = session2
  return session2

from tensorflow.python.distribute import values

def enclosing_tpu_context():
  return values._enclosing_tpu_context()


from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_gradients
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent



if __name__ == '__main__':
  _tf_patch = patch_tensorflow_interactive()
  if len(sys.argv) <= 1:
    from tensorflow.core.protobuf import config_pb2
    import tensorflow as tf
    tf1 = tf.compat.v1
    tf.compat.v1.logging.set_verbosity('DEBUG')
    import numpy as np
    #session_config = config_pb2.ConfigProto(allow_soft_placement=True, isolate_session_state=True)
    rpc_options = config_pb2.RPCOptions()
    # Setting cache_rpc_response to true will enable sender side caching of
    # response for RecvTensorAsync and RecvBufAsync to allow receiver to retry
    # requests . This is only necessary when the network fabric is experiencing a
    # significant error rate.  Without it we'll fail a step on an network error,
    # while with it we'll be able to complete long steps (like complex
    # initializations) in the face of some network errors during RecvTensor.
    rpc_options.cache_rpc_response = True

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True,
        disable_meta_optimizer=True,
        dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
        fail_on_optimizer_errors=True,
        )

    graph_options = config_pb2.GraphOptions(
        rewrite_options=rewriter_config,
        place_pruned_graph=True,
        infer_shapes=True,
        )

    session_config = config_pb2.ConfigProto(
        graph_options=graph_options,
        allow_soft_placement=True,
        isolate_session_state=False,
        )
    
    master = None
    res = None
    cluster_spec = None
    cluster_def = None
    job_names = None
    master_job = 'worker'
    try:
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
    except:
      import traceback
      traceback.print_exc()
    graph = tf.Graph()
    sess = tf.compat.v1.InteractiveSession(master, graph=graph, config=session_config)
    devices = sess.list_devices()
    cores = sorted([x.name for x in devices if ':TPU:' in x.name])
    num_cores = len(cores)
    print(cluster_def)
    print('cores: %d ip: %s' % (num_cores, master))
    r = sess.run
    from importlib import reload
    import tf_tools as tft
    from tensorflow.python.tpu import tpu as tpu_ops
    from tensorflow.compiler.tf2xla.python import xla
    from tensorflow.compiler.tf2xla.ops import gen_xla_ops
    from tensorflow.python.tpu import tpu_strategy_util
    from tensorflow.python.tpu import device_assignment as device_assignment_lib
    from tensorflow.python.tpu import topology as topology_lib
    tpu_topology = None
    topology_cache = {}
    try:
      with open('topology.cache', 'r') as f:
        topology_cache = json.load(f)
    except FileNotFoundError:
      pass
    def cached_topology(name=None):
      if name is None:
        name = os.environ.get('TPU_NAME', None)
      result = topology_cache.get(name, None)
      if result is not None:
        serialized = base64.b64decode(result)
        return topology_lib.Topology(serialized=serialized)
    def get_topology():
      global tpu_topology
      tpu_topology = cached_topology()
      if tpu_topology is None:
        tpu_topology = tpu_strategy_util.initialize_tpu_system(res)
        topology_cache.update({os.environ['TPU_NAME']: base64.b64encode(tpu_topology.serialized()).decode('utf8')})
        with open('topology.cache', 'w') as f:
          f.write(json.dumps(topology_cache))
      return tpu_topology
    def get_task_and_cores_to_replicas():
      return device_assignment_lib._compute_task_and_cores_to_replicas(tpu_topology.device_coordinates, tpu_topology)
    def get_core_assignment(*core_ids):
      return device_assignment_lib.DeviceAssignment(get_topology(), [[get_topology().device_coordinates[0][i]] for i in core_ids])
    def get_device_assignment(num_replicas, computation_shape=None, topology=None):
      if topology is None:
        topology = get_topology()
      if computation_shape is None:
        computation_shape = [1, 1, 1, 2]
      device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, computation_shape=computation_shape, num_replicas=num_replicas)
      return device_assignment
    tpu_topology = cached_topology()
  else:
    filename = sys.argv[1]
    sys.argv = sys.argv[1:]
    with open(filename) as f:
      source = f.read()
    code = compile(source, filename, 'exec')
    exec(code, globals(), globals())


