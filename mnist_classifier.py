"""MNIST classifier. See MNISTClassifier class for usage example."""
import tftorch as nn

class MNISTClassifier(nn.Sequential):
  """MNIST classifier.

  Example usage:

      >>> import mnist_classifier
      >>> mnist = mnist_classifier.MNISTClassifier()
      >>> mnist
      MNISTClassifier(
        (0): Conv2d(1, 10, kernel_size=(1, 1), stride=(1, 1), padding=VALID)
        (1): ResidualBlock(
          (0): ReLU()
          (1): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=SAME)
          (2): ReLU()
          (3): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=SAME)
        )
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): ResidualBlock(
          (0): ReLU()
          (1): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=SAME)
          (2): ReLU()
          (3): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=SAME)
        )
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Flatten(start_dim=1, end_dim=-1)
        (6): Linear(in_features=490, out_features=10, bias=True)
        (7): LogSoftmax(dim=-1)
      )
      >>> pp(list(mnist.parameters()))
      [<tf.Variable 'mnist/pre_conv/w:0' shape=(1, 1, 1, 10) dtype=float32>,
       <tf.Variable 'mnist/pre_conv/b:0' shape=(10,) dtype=float32>,
       <tf.Variable 'mnist/residual/conv_2d/w:0' shape=(3, 3, 10, 10) dtype=float32>,
       <tf.Variable 'mnist/residual/conv_2d/b:0' shape=(10,) dtype=float32>,
       <tf.Variable 'mnist/residual/conv_2d_1/w:0' shape=(3, 3, 10, 10) dtype=float32>,
       <tf.Variable 'mnist/residual/conv_2d_1/b:0' shape=(10,) dtype=float32>,
       <tf.Variable 'mnist/residual_1/conv_2d/w:0' shape=(3, 3, 10, 10) dtype=float32>,
       <tf.Variable 'mnist/residual_1/conv_2d/b:0' shape=(10,) dtype=float32>,
       <tf.Variable 'mnist/residual_1/conv_2d_1/w:0' shape=(3, 3, 10, 10) dtype=float32>,
       <tf.Variable 'mnist/residual_1/conv_2d_1/b:0' shape=(10,) dtype=float32>,
       <tf.Variable 'mnist/linear/w:0' shape=(490, 10) dtype=float32>,
       <tf.Variable 'mnist/linear/b:0' shape=(10,) dtype=float32>]
      >>> batch_size = 16
      >>> input = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="mnist_in")
      >>> output = mnist(input)
      >>> output
      <tf.Tensor 'mnist_23/log_softmax/LogSoftmax:0' shape=(16, 10) dtype=float32>
      >>> mnist
      MNISTClassifier(
        IN:  f32[16,28,28,1, name='mnist_in_12:0'],
        OUT: f32[16,10, name='mnist_23/log_softmax/LogSoftmax:0']
        (0): Conv2d(
          1, 10, kernel_size=(1, 1), stride=(1, 1), padding=VALID
          IN:  f32[16,28,28,1, name='mnist_in_12:0'],
          OUT: f32[16,28,28,10, name='mnist_23/pre_conv/BiasAdd:0']
        )
        (1): ResidualBlock(
          IN:  f32[16,28,28,10, name='mnist_23/pre_conv/BiasAdd:0'],
          OUT: f32[16,28,28,10, name='mnist_23/residual/add:0']
          (0): ReLU(
            IN:  f32[16,28,28,10, name='mnist_23/residual/mnist_23/pre_conv/BiasAdd_clone:0'],
            OUT: f32[16,28,28,10, name='mnist_23/residual/ReLU/Relu:0']
          )
          (1): Conv2d(
            10, 10, kernel_size=(3, 3), stride=(1, 1), padding=SAME
            IN:  f32[16,28,28,10, name='mnist_23/residual/ReLU/Relu:0'],
            OUT: f32[16,28,28,10, name='mnist_23/residual/conv_2d/BiasAdd:0']
          )
          (2): ReLU(
            IN:  f32[16,28,28,10, name='mnist_23/residual/conv_2d/BiasAdd:0'],
            OUT: f32[16,28,28,10, name='mnist_23/residual/ReLU_1/Relu:0']
          )
          (3): Conv2d(
            10, 10, kernel_size=(3, 3), stride=(1, 1), padding=SAME
            IN:  f32[16,28,28,10, name='mnist_23/residual/ReLU_1/Relu:0'],
            OUT: f32[16,28,28,10, name='mnist_23/residual/conv_2d_1/BiasAdd:0']
          )
        )
        (2): MaxPool2d(
          kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
          IN:  f32[16,28,28,10, name='mnist_23/residual/add:0'],
          OUT: f32[16,14,14,10, name='mnist_23/max_pool2d/MaxPool2d:0']
        )
        (3): ResidualBlock(
          IN:  f32[16,14,14,10, name='mnist_23/max_pool2d/MaxPool2d:0'],
          OUT: f32[16,14,14,10, name='mnist_23/residual_1/add:0']
          (0): ReLU(
            IN:  f32[16,14,14,10, name='mnist_23/residual_1/mnist_23/max_pool2d/MaxPool2d_clone:0'],
            OUT: f32[16,14,14,10, name='mnist_23/residual_1/ReLU/Relu:0']
          )
          (1): Conv2d(
            10, 10, kernel_size=(3, 3), stride=(1, 1), padding=SAME
            IN:  f32[16,14,14,10, name='mnist_23/residual_1/ReLU/Relu:0'],
            OUT: f32[16,14,14,10, name='mnist_23/residual_1/conv_2d/BiasAdd:0']
          )
          (2): ReLU(
            IN:  f32[16,14,14,10, name='mnist_23/residual_1/conv_2d/BiasAdd:0'],
            OUT: f32[16,14,14,10, name='mnist_23/residual_1/ReLU_1/Relu:0']
          )
          (3): Conv2d(
            10, 10, kernel_size=(3, 3), stride=(1, 1), padding=SAME
            IN:  f32[16,14,14,10, name='mnist_23/residual_1/ReLU_1/Relu:0'],
            OUT: f32[16,14,14,10, name='mnist_23/residual_1/conv_2d_1/BiasAdd:0']
          )
        )
        (4): MaxPool2d(
          kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
          IN:  f32[16,14,14,10, name='mnist_23/residual_1/add:0'],
          OUT: f32[16,7,7,10, name='mnist_23/max_pool2d_1/MaxPool2d:0']
        )
        (5): Flatten(
          start_dim=1, end_dim=-1
          IN:  f32[16,7,7,10, name='mnist_23/max_pool2d_1/MaxPool2d:0'],
          OUT: f32[16,490, name='mnist_23/flatten/Reshape:0']
        )
        (6): Linear(
          in_features=490, out_features=10, bias=True
          IN:  f32[16,490, name='mnist_23/flatten/Reshape:0'],
          OUT: f32[16,10, name='mnist_23/linear/BiasAdd:0']
        )
        (7): LogSoftmax(
          dim=-1
          IN:  f32[16,10, name='mnist_23/linear/BiasAdd:0'],
          OUT: f32[16,10, name='mnist_23/log_softmax/LogSoftmax:0']
        )
      )
      >>> 
  """
  def __init__(self, scope='mnist', **kwargs):
    super().__init__(scope=scope, **kwargs, body=lambda: [
      nn.Conv2d(1, 10, 1, scope='pre_conv'),
      nn.ResidualBlock(body=lambda: [
          nn.ReLU(),
          nn.Conv2d(10, 10, 3, padding=1),
          nn.ReLU(),
          nn.Conv2d(10, 10, 3, padding=1, index=1),
      ]),
      nn.MaxPool2d(2),
      nn.ResidualBlock(index=1, body=lambda: [
          nn.ReLU(),
          nn.Conv2d(10, 10, 3, padding=1),
          nn.ReLU(),
          nn.Conv2d(10, 10, 3, padding=1, index=1),
      ]),
      nn.MaxPool2d(2),
      nn.Flatten(),
      nn.Linear(7*7*10, 10),
      nn.LogSoftmax(dim=-1),
  ])

