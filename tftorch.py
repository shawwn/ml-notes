import tensorflow.compat.v1 as tf

import six

from six import with_metaclass

from functools import partial

from collections import OrderedDict
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict

import math

import numpy as np

# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')



def absolute_variable_scope(scope=None, reuse=None, **kwargs):
    if scope is None:
        scope = tf.get_variable_scope().name
    return tf.variable_scope(tf.VariableScope(name=scope, reuse=reuse, **kwargs), auxiliary_name_scope=False)


def resource_scope():
  return absolute_variable_scope(use_resource=True)


class BackwardCFunction:#(_C._FunctionBase, _ContextMethodMixin, _HookMixin):
  _is_legacy = False

  def __init__(self):
    self.saved_tensors = []
    self.grad_variables = None

  def save_for_backward(self, *tensors):
    self.saved_tensors.extend(tensors)

  def _apply(self, input_, **kwargs):
    with resource_scope():
      #return self._forward_cls.backward(self, *args)
      self.props = kwargs
      forward = partial(self._forward_cls.forward, self)
      backward = partial(self._forward_cls.backward, self)
      @tf.custom_gradient
      def func(x):
        def grad(dy, variables=None):
          self.grad_variables = variables
          return backward(dy)
        return forward(x), grad
      return func(input_)

  def apply(self, input_):
    return self._apply(input_, boxed=[input_], index=0, is_list=False)


class FunctionMeta(type):
  """Function metaclass.

  This metaclass sets up the following properties:
      _is_legacy: True if forward is not defined as a static method.
      _backward_cls: The Function class corresponding to the differentiated
          version of this function (which is generated on the fly by this
          metaclass).
  """
  def __init__(cls, name, bases, attrs):
    for super_cls in cls.mro():
      forward = super_cls.__dict__.get('forward')
      if forward is not None:
        has_static_forward = isinstance(forward, staticmethod) or isinstance(forward, classmethod)
        break
    cls._is_legacy = not has_static_forward
    # old-style functions
    if not has_static_forward:
      return super(FunctionMeta, cls).__init__(name, bases, attrs)
    backward_fn = type(name + 'Backward', (BackwardCFunction,), {'_forward_cls': cls})
    cls._backward_cls = backward_fn
    return super(FunctionMeta, cls).__init__(name, bases, attrs)


class Function(with_metaclass(FunctionMeta)):#, _C._FunctionBase, _ContextMethodMixin, _HookMixin)):
  @classmethod
  def apply(cls, *args, **kwargs): # real signature unknown
    return cls._backward_cls().apply(*args, **kwargs)
  @staticmethod
  def forward(self, input_):
    return input_
  @staticmethod
  def backward(self, grad_output):
    return grad_output


# class _FooFunction(Function):
#     @staticmethod
#     def forward(ctx, input_):
#         return input_
#     @staticmethod
#     def backward(ctx, grad_output):
#         return tf.zeros_like(grad_output)


#_FooFunction.apply(tf.ones([16]))
# x = tf.constant(100.); y = _FooFunction.apply(x); dy =  tf.gradients(y, x); 


# class _Module:
#   def __init__(self):
#     pass
#   def forward(self, x):
#     return x
#   def backward(self, dy):
#     return dy
#   def __call__(self, x):
#     class _Call(Function):
#       @staticmethod
#       def forward(ctx, *args):
#         self.ctx = ctx
#         return self.forward(*args)
#       @staticmethod
#       def backward(ctx, *args):
#         self.ctx = ctx
#         return self.backward(*args)
#     return _Call.apply(x)
    


def torch_typename(module):
  return type(module)


def globalvar(name, **kws):
  shape = kws.pop('shape')
  initializer = kws.pop('initializer', None)
  if initializer is None:
    initializer = tf.initializers.zeros
  collections = kws.pop('collections', ['variables'])
  trainable = kws.pop('trainable', True)
  use_resource = kws.pop('use_resource', True)
  dtype = kws.pop('dtype', tf.float32)
  return tf.get_variable(name, dtype=dtype, initializer=initializer, shape=shape, collections=collections, use_resource=use_resource, trainable=trainable, **kws)


def localvar(name, **kws):
  collections = kws.pop('collections', ['local_variables'])
  trainable = kws.pop('trainable', False)
  use_resource = kws.pop('use_resource', True)
  return globalvar(name, **kws, collections=collections, trainable=trainable, use_resource=use_resource)


class Module(object):
    def __init__(self, scope=None, index=None):
        self.training = True
        self._modules = OrderedDict()
        self._scope = scope
        self._index = index

    def scope(self, name=None, index=None, postfix=None, reuse=tf.AUTO_REUSE, **kwargs):
      if name is None:
        if self._scope is None:
          name = type(self).__name__
        else:
          name = self._scope
      if index is None:
        index = self._index
      if index is not None:
        if index != 0:
          name = name + '_' + str(index)
      if postfix is not None:
        name = name + postfix
      return tf.variable_scope(name, reuse=reuse, **kwargs)
    
    def globalvar(self, name, **kws):
      return globalvar(name, **kws)
    
    def localvar(self, name, **kws):
      return localvar(name, **kws)

    def register_parameter(self, name, value):
      assert not hasattr(self, name)
      if value is None:
        setattr(self, name, value)
      else:
        v = self.globalvar(name, shape=value.shape, dtype=value.dtype)
        init_(v, value)
        setattr(self, name, v)
      return getattr(self, name)

    def register_buffer(self, name, value):
      assert not hasattr(self, name)
      if value is None:
        setattr(self, name, value)
      else:
        v = self.localvar(name, shape=value.shape, dtype=value.dtype, collections=['variables'])
        init_(v, value)
        setattr(self, name, v)
      return getattr(self, name)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch_typename(module)))
        elif not isinstance(name, six.string_types):
            raise TypeError("module name should be a string. Got {}".format(
                torch_typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)
        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        return result

    def children(self) -> Iterator['Module']:
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Example::

            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        r"""Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
                    print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = ''):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m


    def train(self: T, mode: bool = True) -> T:
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self


    def eval(self: T) -> T:
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        Returns:
            Module: self
        """
        return self.train(False)




class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args: Any):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


  
  
class ReLU(Module):
  def forward(self, input):
    with self.scope():
      return tf.nn.relu(input)


class SquareFn(Function):
  @staticmethod
  def forward(self, input):
    output = tf.square(input)
    self.save_for_backward(input, output)
    return output
  @staticmethod
  def backward(self, grad_output):
    input, output = self.saved_tensors
    return tf.gradients(output, [input], grad_output)



class Identity(Module):
  def forward(self, x):
    with self.scope():
      return tf.identity(x)


class Square(Module):
  def forward(self, x):
    with self.scope():
      return SquareFn.apply(x)


class FooModel(Module):
  def __init__(self):
    super(FooModel, self).__init__()
    with self.scope():
      self.square1 = Square()
      self.square2 = Square()

  def forward(self, x):
    with self.scope():
      x = self.square1(x)
      x = self.square2(x)
      return x


relu = tf.nn.relu


def linear(input, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    output = tf.matmul(input, weight)
    if bias is not None:
        #output = tf.add(output, bias)
        output = tf.nn.bias_add(output, bias)
    return output

Tensor = tf.Variable

class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, scope='linear', **kwargs) -> None:
        super(Linear, self).__init__(scope=scope, **kwargs)
        with self.scope():
          self.in_features = in_features
          self.out_features = out_features
          # self.weight = Parameter(torch.Tensor(out_features, in_features))
          # if bias:
          #     self.bias = Parameter(torch.Tensor(out_features))
          # else:
          #     self.register_parameter('bias', None)
          # self.weight = tf.Variable(tf.zeros(shape=[out_features, in_features]), use_resource=True, name="w")
          # if bias:
          #   self.bias = tf.Variable(tf.zeros(shape=[out_features]), use_resource=True, name="b")
          # else:
          #   self.bias = None
          self.weight = self.globalvar('w', shape=[in_features, out_features])
          if bias:
            self.bias = self.globalvar('b', shape=[out_features])
          else:
            self.bias = None
          self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        with self.scope():
            return linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def assertion(condition):
  #assert condition
  if not condition:
    import pdb; pdb.set_trace()
  return condition

def assert_shape(lhs, shape):
  x = lhs
  if hasattr(lhs, 'shape'):
    lhs = lhs.shape
  if hasattr(lhs, 'as_list'):
    lhs = lhs.as_list()
  assertion(lhs == shape)
  return x
        

def shapelist(x):
  if hasattr(x, 'shape'):
    x = x.shape
  return x.as_list()


def size(tensor, index=None):
  if index is None:
    return shapelist(tensor)
  else:
    return shapelist(tensor)[index]


def dim(tensor):
  return len(shapelist(tensor))


def view(tensor, *shape, name=None):
  return tf.reshape(tensor, shape, name=name)


def permute(tensor, *pattern, name=None):
  return tf.transpose(tensor, pattern, name=name)


def cat(x, axis, name=None):
  return tf.concat(x, axis=axis, name=name)


def squeeze(x, axis=None, name=None):
  return tf.squeeze(x, axis=axis, name=name)


def sum(x, axis, name=None):
  return tf.reduce_sum(x, axis=axis)


def numel(tensor):
  return np.prod(size(tensor))


def randn(*shape):
  return tf.random.uniform(shape=shape)



def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import variables



def create_initializer_op(handle, initial_value, name):
  with ops.name_scope("IsInitialized"):
    is_initialized_op = (
        gen_resource_variable_ops.var_is_initialized_op(handle))
  assert initial_value is not None
  # pylint: disable=g-backslash-continuation
  with ops.name_scope("Assign") as n, \
       ops.colocate_with(None, ignore_existing=True), \
       ops.device(handle.device):
    # pylint: disable=protected-access
    initializer_op = (
        gen_resource_variable_ops.assign_variable_op(
            handle,
            variables._try_guard_against_uninitialized_dependencies(
                name, initial_value),
            name=n))
    return initializer_op
  

def init_(tensor, value):
  # tensor is a Variable?
  assert hasattr(tensor, 'initializer')
  assert hasattr(tensor, '_initializer_op')
  # and additionaly is a ResourceVariable? TODO: handle normal variables.
  if not hasattr(tensor, 'handle'):
    raise NotImplementedError("TODO: support non-resource ops; for now just use reource ops everywhere")
  # overwrite its initializer.
  tensor._initializer_op = create_initializer_op(
      handle=tensor.handle,
      initial_value=value,
      name=tensor.name)


def uniform_(tensor, minval, maxval):
  value = tf.random.uniform(shape=tensor.shape, minval=minval, maxval=maxval)
  init_(tensor, value)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = dim(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = size(tensor, 1)
    num_output_fmaps = size(tensor, 0)
    receptive_field_size = 1
    if dim(tensor) > 2:
         # the following tensor[0][0] adds a StridedSlice op to the graph, which
         # is then ignored, but maybe it's worth overlooking this
         # minor detail in favor of simplicity. The graph optimizer
         # should prune it anyway.
        receptive_field_size = numel(tensor[0][0])
        #receptive_field_size = np.prod(shapelist(tensor)[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    # with torch.no_grad():
    #     return tensor.uniform_(-bound, bound)
    uniform_(tensor, -bound, bound)





from collections import abc as container_abcs
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if x is None:
            return None
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)



class _ConvNd(Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      bias,
      padding_mode,
      scope=None,
      **kwargs):
    super(_ConvNd, self).__init__(scope=scope, **kwargs)
    if in_channels % groups != 0:
      raise ValueError('in_channels must be divisible by groups')
    if out_channels % groups != 0:
      raise ValueError('out_channels must be divisible by groups')
    valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
    if padding_mode not in valid_padding_modes:
      raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
        valid_padding_modes, padding_mode))

    with self.scope():
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding
      self.dilation = dilation
      self.transposed = transposed
      self.output_padding = output_padding
      self.groups = groups
      self.padding_mode = padding_mode
      if transposed:
        self.weight = self.globalvar('w', shape=[*kernel_size, out_channels, in_channels // groups])
      else:
        self.weight = self.globalvar('w', shape=[*kernel_size, in_channels, out_channels // groups])
      if bias:
        self.bias = self.globalvar('b', shape=[out_channels])
      else:
        self.bias = None
    self.reset_parameters()

  def reset_parameters(self):
    kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      uniform_(self.bias, -bound, bound)

  def extra_repr(self):
      s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
           ', stride={stride}')
      if self.padding != (0,) * len(self.padding):
          s += ', padding={padding}'
      if self.dilation != (1,) * len(self.dilation):
          s += ', dilation={dilation}'
      if self.output_padding != (0,) * len(self.output_padding):
          s += ', output_padding={output_padding}'
      if self.groups != 1:
          s += ', groups={groups}'
      if self.bias is None:
          s += ', bias=False'
      if self.padding_mode != 'zeros':
          s += ', padding_mode={padding_mode}'
      return s.format(**self.__dict__)


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    Note:

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',  # TODO: refine this type
        scope='conv_2d',
        **kwargs,
    ):
      kernel_size = _pair(kernel_size)
      stride = _pair(stride)
      #padding = _pair(padding)
      if padding == 0:
        padding = "VALID"
      elif padding == 1:
        padding = "SAME"
      else:
        padding = _pair(padding)
      dilation = _pair(dilation)
      super(Conv2d, self).__init__(
        in_channels, out_channels, kernel_size, stride, padding, dilation,
        False, _pair(0), groups, bias, padding_mode, scope=scope, **kwargs)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            raise NotImplementedError()
            # return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
            #                 weight, self.bias, self.stride,
            #                 _pair(0), self.dilation, self.groups)
        # return F.conv2d(input, weight, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)
        #assert self.padding == [0, 0]
        #padding = 'VALID'
        strides = [1, *self.stride, 1]
        #strides = self.stride
        data_format = 'NHWC'
        padding = self.padding
        if isinstance(padding, (list, tuple)):
          padding = [[0, 0], padding, padding, [0, 0]]
        #padding = [[0, 0], self.padding, self.padding, [0, 0]]
        #padding = 'VALID'
        #padding = 'SAME'
        print(self.padding)
        # data_format = 'NCHW'
        # padding = [[0, 0], [0, 0], self.padding, self.padding]
        dilations = self.dilation
        output = tf.nn.conv2d(input, weight, strides=strides, padding=padding, data_format=data_format, dilations=dilations)
        if self.bias is not None:
          #output = tf.add(output, self.bias)
          output = tf.nn.bias_add(output, self.bias)
        return output

    def forward(self, input):
        with self.scope():
          return self._conv_forward(input, self.weight)





def unpool(value, name="unpool"):
  """Unpooling operation.

  N-dimensional version of the unpooling operation from
  https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
  Taken from: https://github.com/tensorflow/tensorflow/issues/2169

  Args:
    value: a Tensor of shape [b, d0, d1, ..., dn, ch]
    name: name of the op
  Returns:
    A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
  """
  with tf.name_scope(name) as scope:
    sh = value.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
      out = tf.concat([out, tf.zeros_like(out)], i)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size, name=scope)
  return out


def unpool2(value, name=None):
  x = tf.concat([value, value, value, value], axis=3)
  return tf.nn.depth_to_space(x, 2, name=name)


def interpolate(input, scale_factor=2, name="interpolate"):
  if scale_factor % 2 != 0:
    raise ValueError("scale_factor must be a multiple of 2, got {}".format(scale_factor))
  out = input
  while scale_factor // 2 >= 1:
    out = unpool2(out, name=name)
    scale_factor //= 2
  return out



def pool(input, kernel_size, stride=None, pooling_type="AVG", padding="SAME", name=None):
  kernel_size = _pair(kernel_size)
  strides = _pair(kernel_size if stride is None else stride)
  padding = padding or "SAME"
  return tf.nn.pool(input, kernel_size, pooling_type, padding, strides=strides, name=name)
    

upsample = interpolate
downsample = partial(pool, kernel_size=2)


def max_pool2d(input, kernel_size, stride, padding="SAME", data_format="NHWC", name=None):
  kernel_size = _pair(kernel_size)
  strides = _pair(stride)
  #padding = padding or "SAME"
  if padding == 0:
    padding = "VALID"
  elif padding == 1:
    padding = "SAME"
  else:
    padding = _pair(padding)
  print('max_pool2d', padding)
  return tf.nn.max_pool2d(input, kernel_size, strides, padding, data_format, name=name)
  #return tf.nn.pool(input, kernel_size, "MAX", "SAME", strides=strides, name=name)


class MaxPool2d(Module):
  def __init__(self,
      kernel_size,
      stride,
      padding=0,
      scope='max_pool2d',
      **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.kernel_size = _pair(kernel_size)
      self.stride = _pair(stride)
      self.padding = padding
  def forward(self, input):
    with self.scope():
      return max_pool2d(input, self.kernel_size, self.stride, self.padding)


def avg_pool2d(input, kernel_size, stride=None, padding=0, data_format="NHWC", name=None):
  kernel_size = _pair(kernel_size)
  strides = _pair(kernel_size if stride is None else stride)
  #padding = padding or "SAME"
  if padding == 0:
    padding = "VALID"
  elif padding == 1:
    padding = "SAME"
  else:
    padding = _pair(padding)
  print('avg_pool2d', padding)
  return tf.nn.avg_pool2d(input, kernel_size, strides, padding, data_format, name=name)
  #return tf.nn.pool(input, kernel_size, "AVG", "SAME", strides=strides, name=name)


class AvgPool2d(Module):
  def __init__(self,
      kernel_size,
      stride=None,
      padding=0,
      scope='avg_pool2d',
      **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.kernel_size = _pair(kernel_size)
      self.stride = _pair(kernel_size if stride is None else stride)
      self.padding = padding
  def forward(self, input):
    with self.scope():
      return avg_pool2d(input, self.kernel_size, self.stride, self.padding)


def softmax(input, dim, name=None):
  return tf.nn.softmax(input, axis=dim, name=name)


class Softmax(Module):
  def __init__(self,
      dim,
      scope='softmax',
      **kwargs):
    super().__init__(scope=scope, **kwargs)
    with self.scope():
      self.dim = dim
  def forward(self, input):
    with self.scope():
      return softmax(input, dim=self.dim)


def bmm(input, mat2, transpose_a=False, transpose_b=False, name=None):
  return tf.matmul(input, mat2, transpose_a=transpose_a, transpose_b=transpose_b, name=name)


def zeros(*size, out=None, **kwargs):
  if out is not None:
    raise NotImplementedError()
  return tf.zeros(shape=size, **kwargs)


def ones(*size, out=None, **kwargs):
  if out is not None:
    raise NotImplementedError()
  return tf.ones(shape=size, **kwargs)



def zeros_(tensor):
  init_(tensor, tf.zeros_like(tensor))


def ones_(tensor):
  init_(tensor, tf.ones_like(tensor))



class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        scope='batchnorm',
        **kwargs,

    ) -> None:
        super(_NormBase, self).__init__(scope=scope, **kwargs)
        with self.scope():
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.track_running_stats = track_running_stats
            if self.affine:
                #self.weight = Parameter(torch.Tensor(num_features))
                #self.bias = Parameter(torch.Tensor(num_features))
                self.weight = self.globalvar('weight', shape=[num_features])
                self.bias = self.globalvar('bias', shape=[num_features])
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
            if self.track_running_stats:
                self.register_buffer('accumulated_mean', tf.zeros(num_features))
                self.register_buffer('accumulated_var', tf.ones(num_features))
                #self.register_buffer('accumulation_counter', tf.tensor(0, dtype=torch.long))
                self.register_buffer('accumulation_counter', tf.zeros([]))
            else:
                self.register_parameter('accumulated_mean', None)
                self.register_parameter('accumulated_var', None)
                self.register_parameter('accumulation_counter', None)
            self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            zeros_(self.accumulated_mean)
            ones_(self.accumulated_var)
            zeros_(self.accumulation_counter)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     version = local_metadata.get('version', None)
    #
    #     if (version is None or version < 2) and self.track_running_stats:
    #         # at version 2: added num_batches_tracked buffer
    #         #               this should have a default value of 0
    #         num_batches_tracked_key = prefix + 'num_batches_tracked'
    #         if num_batches_tracked_key not in state_dict:
    #             state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
    #
    #     super(_NormBase, self)._load_from_state_dict(
    #         state_dict, prefix, local_metadata, strict,
    #         missing_keys, unexpected_keys, error_msgs)



class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, scope='batch_norm', **kwargs):
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, scope=scope, **kwargs)

    def forward(self, input):
        with self.scope():
            self._check_input_dim(input)

            # exponential_average_factor is set to self.momentum
            # (when it is available) only so that it gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.accumulation_counter is not None:
                    self.accumulation_counter = self.accumulation_counter + 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.accumulation_counter)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.accumulated_mean is None) and (self.accumulated_var is None)

            r"""
            Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
            passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
            used for normalization (i.e. in eval mode when buffers are not None).
            """
            return batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.accumulated_mean if not self.training or self.track_running_stats else None,
                self.accumulated_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


def batch_norm(input, mean, variance, weight, bias, training, exponential_average_factor, variance_epsilon):
  # # TODO: exponential_average_factor
  # out = tf.nn.batch_normalization(input, mean, variance, offset=bias, scale=weight, variance_epsilon=variance_epsilon)
  inv_var = tf.math.rsqrt(variance + variance_epsilon)
  weight_v = 1.0 if weight is None else weight
  bias_v = 0.0 if bias is None else bias
  alpha = inv_var * weight_v
  beta = bias_v - mean * alpha
  out = input * alpha + beta
  return out




class BatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the accumulated_mean and accumulated_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`accumulated_mean` and :attr:`accumulated_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if dim(input) != 2 and dim(input) != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(dim(input)))


class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the accumulated_mean and accumulated_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`accumulated_mean` and :attr:`accumulated_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if dim(input) != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(dim(input)))


class BatchNorm3d(_BatchNorm):
    r"""Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the accumulated_mean and accumulated_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`accumulated_mean` and :attr:`accumulated_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if dim(input) != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(dim(input)))



class Embedding(Module):
  def __init__(self, num_embeddings, embedding_dim, max_norm=None, scope=None, **kwargs):
    super().__init__(scope=scope, **kwargs)
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.max_norm = max_norm
    with self.scope():
      self.weight = self.globalvar('w', shape=[num_embeddings, embedding_dim])

  def forward(self, input):
    with self.scope():
      return embedding(input, self.weight, max_norm=self.max_norm)


def embedding(input, params, max_norm=None, name=None):
  if False:
    if input.dtype not in [tf.int32, tf.int64]:
      input = tf.cast(input, tf.int32)
    unhot = tf.argmax(input, axis=-1)
    return tf.nn.embedding_lookup(params, input, max_norm=max_norm, name=name)
  else:
    return tf.matmul(input, params)


