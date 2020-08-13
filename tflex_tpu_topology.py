import os
import json
import base64

from tensorflow.python.tpu import tpu as tpu_ops
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import topology as topology_lib
#from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tflex import TPUClusterResolver


_TOPOLOGY_CACHE_FILENAME = '.tpu_topology_cache.json'


class Context():
  pass


if 'api' not in globals():
  api = Context()
  api.topology = None
  api.topology_cache = {}
  try:
    with open(_TOPOLOGY_CACHE_FILENAME, 'r') as f:
      api.topology_cache = json.load(f)
  except FileNotFoundError:
    pass


def cached_topology(name=None):
  if name is None:
    name = os.environ.get('TPU_NAME', '')
  result = api.topology_cache.get(name, None)
  if result is not None:
    serialized = base64.b64decode(result)
    return topology_lib.Topology(serialized=serialized)


def get_cluster_resolver(cluster_resolver=None):
  if cluster_resolver is None:
    cluster_resolver = TPUClusterResolver(os.environ['TPU_NAME'])
  return cluster_resolver


def get_topology(cluster_resolver=None):
  api.topology = cached_topology()
  if api.topology is None:
    cluster_resolver = get_cluster_resolver(cluster_resolver)
    api.topology = tpu_strategy_util.initialize_tpu_system(cluster_resolver)
    api.topology_cache.update({os.environ['TPU_NAME']: base64.b64encode(api.topology.serialized()).decode('utf8')})
    with open(_TOPOLOGY_CACHE_FILENAME, 'w') as f:
      f.write(json.dumps(api.topology_cache))
  return api.topology


def get_task_and_cores_to_replicas():
  return device_assignment_lib._compute_task_and_cores_to_replicas(api.topology.device_coordinates, api.topology)


def get_core_assignment(*core_ids):
  return device_assignment_lib.DeviceAssignment(get_topology(), [[get_topology().device_coordinates[0][i]] for i in core_ids])


def get_metadata(cluster_resolver=None):
  cluster_resolver = get_cluster_resolver(cluster_resolver)
  meta = tpu_system_metadata_lib._query_tpu_system_metadata(cluster_resolver.get_master(), cluster_def=cluster_resolver.cluster_spec().as_cluster_def(), query_topology=True)
  return meta


api.topology = cached_topology()

