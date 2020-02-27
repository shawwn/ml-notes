import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_ops
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import server_lib

class GrpcServerTest(test.TestCase):

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(GrpcServerTest, self).__init__(methodName)
    self._cached_server = server_lib.Server.create_local_server()

  def testRunStep(self):
    server = self._cached_server

    with session.Session(server.target) as sess:
      c = constant_op.constant([[2, 1]])
      d = constant_op.constant([[1], [2]])
      e = math_ops.matmul(c, d)
      self.assertAllEqual([[4]], sess.run(e))
    # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  @test_util.run_v1_only("b/120545219")
  def testMultipleSessions(self):
    server = self._cached_server

    c = constant_op.constant([[2, 1]])
    d = constant_op.constant([[1], [2]])
    e = math_ops.matmul(c, d)

    sess_1 = session.Session(server.target)
    sess_2 = session.Session(server.target)

    self.assertAllEqual([[4]], sess_1.run(e))
    self.assertAllEqual([[4]], sess_2.run(e))

    sess_1.close()
    sess_2.close()
    # TODO(mrry): Add `server.stop()` and `server.join()` when these work.

  @test_util.run_v1_only("b/120545219")
  def testIsolateSessionState(self):
    server = self._cached_server

    init_value = array_ops.placeholder(dtypes.int32)
    v = variables.VariableV1(init_value, validate_shape=False, name="v")

    sharing_config = config_pb2.ConfigProto(isolate_session_state=False)
    sharing_sess_0 = session.Session(server.target, config=sharing_config)
    sharing_sess_1 = session.Session(server.target, config=sharing_config)

    isolate_config = config_pb2.ConfigProto(isolate_session_state=True)
    isolate_sess_0 = session.Session(server.target, config=isolate_config)
    isolate_sess_1 = session.Session(server.target, config=isolate_config)

    # Initially all variables are initialized.
    for sess in [sharing_sess_0, sharing_sess_1,
                 isolate_sess_0, isolate_sess_1]:
      with self.assertRaises(errors_impl.FailedPreconditionError):
        sess.run(v)

    # Shared sessions will see each other's updates, but isolated sessions
    # will not.
    sharing_sess_0.run(v.initializer, feed_dict={init_value: 86})
    self.assertAllEqual(86, sharing_sess_0.run(v))
    self.assertAllEqual(86, sharing_sess_1.run(v))
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_0.run(v)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_1.run(v)

    # Changing the shape works because `validate_shape` is False.
    sharing_sess_1.run(v.initializer, feed_dict={init_value: [86, 99]})
    self.assertAllEqual([86, 99], sharing_sess_0.run(v))
    self.assertAllEqual([86, 99], sharing_sess_1.run(v))
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_0.run(v)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_1.run(v)

    # Initializing in an isolated session will only affect the state in that
    # session.
    isolate_sess_0.run(v.initializer, feed_dict={init_value: 37})
    self.assertAllEqual([86, 99], sharing_sess_0.run(v))
    self.assertAllEqual([86, 99], sharing_sess_1.run(v))
    self.assertAllEqual(37, isolate_sess_0.run(v))
    with self.assertRaises(errors_impl.FailedPreconditionError):
      isolate_sess_1.run(v)

    # Isolated sessions can have different shapes for the same variable.
    isolate_sess_1.run(v.initializer, feed_dict={init_value: [19, 86]})
    self.assertAllEqual([86, 99], sharing_sess_0.run(v))
    self.assertAllEqual([86, 99], sharing_sess_1.run(v))
    self.assertAllEqual(37, isolate_sess_0.run(v))
    self.assertAllEqual([19, 86], isolate_sess_1.run(v))


if __name__ == "__main__":
  test.main()
