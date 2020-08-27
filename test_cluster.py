import os
import time

# os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"

# TF 2.3.0, you can use kindiana/tf2_test venv to try it out
import tensorflow as tf
from cloud_tpu_client import Client
import logging
# from tensorflow.python.eager import context

# tf.get_logger().setLevel(logging.INFO)
# tf.debugging.set_log_device_placement(True)

@tf.function
def red_sum(a, b, c, d):
    return tf.reduce_sum(a) + tf.reduce_sum(b) + tf.reduce_sum(c) + tf.reduce_sum(d)

c = Client("tpu-v2-8-usc1f-0")
c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')
resolver1 = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="tpu-v2-8-usc1f-0", job_name='tpu0')
tf.config.experimental_connect_to_cluster(resolver1)
tf.tpu.experimental.initialize_tpu_system(resolver1)

# Create the tensors before benchmarking
# looks like ~2GB tensors are the biggest you can send
with tf.device('/job:tpu0/replica:0/task:0/device:TPU:0'):
    tpu0_0 = tf.Variable(tf.fill([256, 1024, 1024], 1))
    tpu0_1 = tf.Variable(tf.fill([256, 1024, 1024], 1))
    tpu0_2 = tf.Variable(tf.fill([256, 1024, 1024], 1))
    tpu0_3 = tf.Variable(tf.fill([256, 1024, 1024], 1))
with tf.device('/job:tpu0/replica:0/task:0/device:CPU:0'):
    tpu0_cpu1 = tf.Variable(tf.fill([256, 1024, 1024], 1))
    tpu0_cpu2 = tf.Variable(tf.fill([256, 1024, 1024], 1))
    tpu0_cpu3 = tf.Variable(tf.fill([256, 1024, 1024], 1))
    tpu0_cpu4 = tf.Variable(tf.fill([256, 1024, 1024], 1))

with tf.device('/job:tpu0/replica:0/task:0/device:CPU:0'):
    c = red_sum(tpu0_0, tpu0_1, tpu0_2, tpu0_3) + red_sum(tpu0_cpu1, tpu0_cpu2, tpu0_cpu3, tpu0_cpu4)
c.numpy()

c = Client("tpu-v2-8-usc1f-1")
c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')
resolver2 = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="tpu-v2-8-usc1f-1", job_name='tpu1')
cluster = tf.distribute.cluster_resolver.UnionResolver(resolver1, resolver2)
tf.config.experimental_connect_to_cluster(cluster)
tf.tpu.experimental.initialize_tpu_system(resolver2)

print(f"total TPU devices: {len(tf.config.list_logical_devices('TPU'))}")

# Case 1: TPU to other TPU's CPU
with tf.device('/job:tpu1/replica:0/task:0/device:CPU:0'):
    c = red_sum(tpu0_0, tpu0_1, tpu0_2, tpu0_3)

start = time.time()
c.numpy()
t = time.time()-start
print(f"transferred 4GB tensors from TPU0:0 to TPU1:CPU in {t:.04} s, {32/t:.04}gbps")

# Case 2: TPU CPU to other TPU CPU
with tf.device('/job:tpu1/replica:0/task:0/device:CPU:0'):
    c = red_sum(tpu0_cpu1, tpu0_cpu2, tpu0_cpu3, tpu0_cpu4)

start = time.time()
c.numpy()
t = time.time()-start
print(f"transferred 4GB tensors from TPU0:CPU to TPU1:CPU in {t:.04} s, {32/t:.04}gbps")

# Case 3a: TPU HBM to other TPU HBM proxying through CPU (proxy on send side CPU)
with tf.device('/job:tpu0/replica:0/task:0/device:CPU:0'):
    send_proxy1 = tf.identity(tpu0_0)
    send_proxy2 = tf.identity(tpu0_1)
    send_proxy3 = tf.identity(tpu0_2)
    send_proxy4 = tf.identity(tpu0_3)

with tf.device('/job:tpu1/replica:0/task:0/device:TPU:0'):
    c = red_sum(send_proxy1, send_proxy2, send_proxy3, send_proxy4)

start = time.time()
c.numpy()
t = time.time()-start
print(f"transferred 4GB tensors from TPU0:0 to TPU1:0 (send side proxying) in {t:.04} s, {32/t:.04}gbps")

# Case 3b: TPU HBM to other TPU HBM proxying through CPU (proxy on recv side CPU)
with tf.device('/job:tpu1/replica:0/task:0/device:CPU:0'):
    recv_proxy1 = tf.identity(tpu0_0)
    recv_proxy2 = tf.identity(tpu0_1)
    recv_proxy3 = tf.identity(tpu0_2)
    recv_proxy4 = tf.identity(tpu0_3)

with tf.device('/job:tpu1/replica:0/task:0/device:TPU:0'):
    c = red_sum(recv_proxy1, recv_proxy2, recv_proxy3, recv_proxy4)

start = time.time()
c.numpy()
t = time.time()-start
print(f"transferred 4GB tensors from TPU0:0 to TPU1:0 (recv side proxying) in {t:.04} s, {32/t:.04}gbps")

# Case 4: TPU HBM to local CPU
with tf.device('/job:tpu0/replica:0/task:0/device:CPU:0'):
    c = red_sum(tpu0_0, tpu0_1, tpu0_2, tpu0_3)

start = time.time()
c.numpy()
t = time.time()-start
print(f"transferred 4GB tensors from TPU0:0 to TPU0:CPU (baseline) in {t:.04} s, {32/t:.04}gbps")

# testing what happens when you remove a node from the cluster
tf.config.experimental_connect_to_cluster(resolver1)
with tf.device('/job:tpu0/replica:0/task:0/device:CPU:0'):
    c = red_sum(tpu0_0, tpu0_1, tpu0_2, tpu0_3) + red_sum(tpu0_cpu1, tpu0_cpu2, tpu0_cpu3, tpu0_cpu4)
c.numpy()

# total TPU devices: 16
# transferred 4GB tensors from TPU0:0 to TPU1:CPU in 4.121 s, 7.765gbps
# transferred 4GB tensors from TPU0:CPU to TPU1:CPU in 3.091 s, 10.35gbps
# transferred 4GB tensors from TPU0:0 to TPU1:0 (send side proxying) in 6.511 s, 4.915gbps
# transferred 4GB tensors from TPU0:0 to TPU1:0 (recv side proxying) in 1.88 s, 17.02gbps
# transferred 4GB tensors from TPU0:0 to TPU0:CPU (baseline) in 2.263 s, 14.14gbps
