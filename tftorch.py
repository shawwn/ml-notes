import tensorflow.compat.v1 as tf

import six

from six import with_metaclass

from functools import partial

from collections import OrderedDict
from typing import Union, Tuple, Any, Callable, Iterable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict

from itertools import islice

import operator

#from torch._jit_internal import _copy_to_script_wrapper
def _copy_to_script_wrapper(fn):
  return fn

import math

import numpy as np

import builtins as py

import re

# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')

from typing import TypeVar, Union, Tuple
#from .. import Tensor

# Create some useful type aliases

# Template for arguments which can be supplied as a tuple, or which can be a scalar which PyTorch will internally
# broadcast to a tuple.
# Comes in several variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d operations.
T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]
_scalar_or_tuple_4_t = Union[T, Tuple[T, T, T, T]]
_scalar_or_tuple_5_t = Union[T, Tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t = Union[T, Tuple[T, T, T, T, T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]
_size_4_t = _scalar_or_tuple_4_t[int]
_size_5_t = _scalar_or_tuple_5_t[int]
_size_6_t = _scalar_or_tuple_6_t[int]

# For arguments that represent a ratio to adjust each dimension of an input with (eg, upsampling parameters)
_ratio_2_t = _scalar_or_tuple_2_t[float]
_ratio_3_t = _scalar_or_tuple_3_t[float]
_ratio_any_t = _scalar_or_tuple_any_t[float]

#_tensor_list_t = _scalar_or_tuple_any_t[Tensor]

# For the return value of max pooling operations that may or may not return indices.
# With the proposed 'Literal' feature to Python typing, it might be possible to
# eventually eliminate this.
#_maybe_indices_t = _scalar_or_tuple_2_t[Tensor]



def calling(op, nresults=1):
  if callable(op):
    op = op()
  if not isinstance(op, (tuple, list)):
    op = [op]
  if nresults is not None:
    assert len(op) == nresults
  return op


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

# class Parameter(object):
#   def __init__(self, initial_value, name, trainable=True):
#     self.initial_value = initial_value
#     self.trainable = trainable
#     self.name = name


from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables

class ParameterMeta(type):
  def __init__(cls, *args, **kwargs):
    print('ParameterMeta.__init__', cls, args, kwargs)
  def __call__(cls, *args, **kwargs):
    print('ParameterMeta.__call__', cls, args, kwargs)
    if cls == Parameter:
      #import pdb; pdb.set_trace()
      name = kwargs.pop('name')
      trainable = kwargs.pop('trainable', True)
      (value,) = args
      with absolute_variable_scope(reuse=tf.AUTO_REUSE):
        initial_value = value
        if callable(value):
          value = value()
        v = globalvar(name, shape=size(value), dtype=value.dtype, trainable=trainable)
        init_(v, initial_value)
      return v
    return cls(*args, **kwargs)
  def __instancecheck__(cls, instance):
    if isinstance(instance, variables.Variable):
      #print('ParameterMeta.__instancecheck__', cls, instance)
      return True
    return super().__instancecheck__(instance)
  def __subclasscheck__(cls, subclass):
    if issubclass(subclass, variables.Variable):
      #print('ParameterMeta.__subclasscheck__', cls, subclass)
      return True
    return super().__subclasscheck__(subclass)

class Parameter(object, metaclass=ParameterMeta):
  def __init__(self, *args, **kwargs):
    print('Parameter.__init__', args, kwargs)

  def __new__(cls, *args, **kwargs):
    print(['Parameter.__new__', cls, args, kwargs])
    try:
      instance = super().__new__(cls, *args, **kwargs)
    except TypeError:
      instance = super().__new__(cls)
    return instance



from tensorflow.python.framework import tensor_like

class TensorMeta(type):
  def __init__(cls, *args, **kwargs):
    print('TensorMeta.__init__', cls, args, kwargs)
  def __call__(cls, *args, **kwargs):
    print('TensorMeta.__call__', cls, args, kwargs)
    if cls == Tensor:
      value = tf.zeros(shape=args, **kwargs)
      return value
    return cls(*args, **kwargs)
  def __instancecheck__(cls, instance):
    if isinstance(instance, tensor_like._TensorLike):
      #print('TensorMeta.__instancecheck__', cls, instance)
      return True
    return super().__instancecheck__(instance)
  def __subclasscheck__(cls, subclass):
    if issubclass(subclass, tensor_like._TensorLike):
      #print('TensorMeta.__subclasscheck__', cls, subclass)
      return True
    return super().__subclasscheck__(subclass)


class Tensor(object, metaclass=TensorMeta):
  def __init__(self, *args, **kwargs):
    print('Tensor.__init__', args, kwargs)

  def __new__(cls, *args, **kwargs):
    print(['Tensor.__new__', cls, args, kwargs])
    try:
      instance = super().__new__(cls, *args, **kwargs)
    except TypeError:
      instance = super().__new__(cls, **kwargs)
    if False and len(args) > 0 and isinstance(args[0], tf.Operation) and len(args[0].inputs) > 0:
      handle = args[0].inputs[0]
      if hasattr(handle, '_names'):
        import pdb; pdb.set_trace()
        instance._names = copy(handle._names)
        print('Set names', instance._names)
      else:
        vs = tf.all_variables()
        xs = [x.handle for x in vs if hasattr(x, 'handle')]
        if handle in xs:
          v = vs[xs.index(handle)]
          if hasattr(v, '_names'):
            instance._names = copy(v._names)
            print('Set names', instance._names)
      # print('TKTK')
      # setattr(instance, '_names', args[0]._names)
    return instance


from copy import copy


def TORCH_CHECK(condition, *message):
  if not condition:
    fmt = ' '.join(['%s' for x in message])
    tf.logging.error(fmt, *message)
    fmt = ' '.join(['{}' for x in message])
    import pdb; pdb.set_trace()
    raise ValueError(fmt.format(*message))

TORCH_INTERNAL_ASSERT = TORCH_CHECK


class ModuleAttributeError(AttributeError):
    """ When `__getattr__` raises AttributeError inside a property,
    AttributeError is raised with the property name instead of the
    attribute that initially raised AttributeError, making the error
    message uninformative. Using `ModuleAttributeError` instead
    fixes this issue."""


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module(object):
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """

    r"""This allows better BC support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""
    _version: int = 1
  
    def __init__(self, scope=None, index=None, index_prefix='_', index_bias=0):
        self.training = True
        self._parent_scope = tf.get_variable_scope().name
        self._scope = scope
        self._variable_scope = None
        self._index = index
        self._index_prefix = index_prefix
        self._index_bias = index_bias
        self._updates = OrderedDict()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._non_persistent_buffers_set = set()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        self._input = None
        self._output = None

    def get_scope_name(self, name=None, index=None, postfix=None, prefix=None):
      if name is None:
        if self._scope is None:
          name = type(self).__name__
        else:
          name = self._scope
      if index is None:
        index = self._index
      if index is not None:
        if index != 0:
          name = name + self._index_prefix + str(index+self._index_bias)
      if postfix is not None:
        name = name + postfix
      if prefix is not None:
        name = prefix + name
      return name

    def scope(self, name=None, index=None, postfix=None, prefix=None, **kwargs):
      name = self.get_scope_name(name=name, index=index, postfix=postfix, prefix=prefix)
      return tf.variable_scope(name, reuse=tf.AUTO_REUSE, **kwargs)

    def as_default(self):
      parent_scope = self._parent_scope
      if not isinstance(parent_scope, six.string_types):
        raise TypeError("parent_scope should be a string. "
                        "Got {}".format(torch_typename(parent_scope)))
      if len(parent_scope) > 0:
        return self.scope(prefix=parent_scope+'/')
      else:
        return self.scope()
    
    def globalvar(self, name, **kws):
      return globalvar(name, **kws)
    
    def localvar(self, name, **kws):
      return localvar(name, **kws)

    def register_parameter(self, name, value):
      assert not hasattr(self, name)
      if value is None:
        setattr(self, name, value)
      else:
        initial_value = value
        if callable(value):
          value = value()
        v = self.globalvar(name, shape=value.shape, dtype=value.dtype)
        init_(v, initial_value)
        setattr(self, name, v)
      return getattr(self, name)

    def register_buffer(self, name, value):
      assert not hasattr(self, name)
      if value is None:
        setattr(self, name, value)
      else:
        initial_value = value
        if callable(value):
          value = value()
        v = self.localvar(name, shape=value.shape, dtype=value.dtype, collections=['variables'])
        init_(v, initial_value)
        setattr(self, name, v)
      return getattr(self, name)
    
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: py.bool = True) -> None:
        r"""Adds a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor): buffer to be registered.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        # if persistent is False and isinstance(self, torch.jit.ScriptModule):
        #     raise RuntimeError("ScriptModule does not support non-persistent buffers")

        # TKTK: Try to support values created via lambda
        initial_value = tensor
        if callable(tensor):
          tensor = tensor()

        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, six.string_types):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(torch_typename(name)))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(torch_typename(tensor), name))
        else:
            if not isinstance(tensor, Parameter):
                tensor = Parameter(initial_value, trainable=False, name=name)
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, six.string_types):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch_typename(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if not isinstance(param, Parameter) and isinstance(param, Tensor):
            param = self.globalvar(name, param)

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch_typename(param), name))
        elif getattr(param, 'grad_fn', None):
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param

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

    def register_update(self, name: str, ops) -> None:
        if not isinstance(name, six.string_types):
            raise TypeError("update name should be a string. Got {}".format(
                torch_typename(name)))
        elif hasattr(self, name) and name not in self._updates:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("update name can't contain \".\"")
        elif name == '':
            raise KeyError("update name can't be empty string \"\"")
        self._updates[name] = ops

    def should_update(self):
      if self.training:
        return True
      return False

    def maybe_update(self, name: str, yes, no, *, should=None):
      if should is None:
        should = self.should_update
      if should():
        yes = calling(yes, 1)[0]
        self.register_update(name=name, ops=yes)
        return yes
      else:
        self.register_update(name=name, ops=None)
        no = calling(no, 1)[0]
        return no

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
        self.__dict__['_input'] = [input, kwargs]
        result = self.forward(*input, **kwargs)
        self.__dict__['_output'] = [result]
        return result

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def updates(self, recurse: py.bool = True) -> Iterator[Iterable[tf.Operation]]:
        r"""Returns an iterator over module updates.

        Args:
            recurse (bool): if True, then yields updates of this module
                and all submodules. Otherwise, yields only updates that
                are direct members of this module.

        Yields:
            Operations: module updates

        """
        for name, ops in self.named_updates(recurse=recurse):
            yield ops

    def named_updates(self, prefix: str = '', recurse: py.bool = True) -> Iterator[Tuple[str, Iterable[tf.Operation]]]:
        r"""Returns an iterator over module updates, yielding both the
        name of the update as well as the updates themselves.

        Args:
            prefix (str): prefix to prepend to all update names.
            recurse (bool): if True, then yields updates of this module
                and all submodules. Otherwise, yields only updates that
                are direct members of this module.

        Yields:
            (string, Operation[]): Tuple containing the name and updates

        """
        gen = self._named_members(
            lambda module: module._updates.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def parameters(self, recurse: py.bool = True) -> Iterator[Parameter]:
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: py.bool = True) -> Iterator[Tuple[str, Tensor]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse: py.bool = True) -> Iterator[Tensor]:
        r"""Returns an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: py.bool = True) -> Iterator[Tuple[str, Tensor]]:
        r"""Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            (string, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

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

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Support loading old checkpoints that don't have the following attrs:
        if '_forward_pre_hooks' not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if '_state_dict_hooks' not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if '_load_state_dict_pre_hooks' not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
        if '_non_persistent_buffers_set' not in self.__dict__:
            self._non_persistent_buffers_set = set()

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise ModuleAttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch_typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch_typename(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch_typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _register_state_dict_hook(self, hook):
        r"""These hooks will be called with arguments: `self`, `state_dict`,
        `prefix`, `local_metadata`, after the `state_dict` of `self` is set.
        Note that only parameters and buffers of `self` or its children are
        guaranteed to exist in `state_dict`. The hooks may modify `state_dict`
        inplace or return a new one.
        """
        handle = hooks.RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else detach(param, name=param.name.rsplit(':', 1)[0])
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else detach(buf, name=buf.name.rsplit(':', 1)[0])

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrederedDict` is created and returned.
    T_destination = TypeVar('T_destination', bound=Mapping[str, Tensor])

    @overload
    def state_dict(self, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    # TODO: annotate with OrderedDict not Dict, but there is a problem:
    # https://docs.python.org/3/library/typing.html#typing.OrderedDict
    @overload
    def state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Tensor]:
        ...

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def _reprs(self, **kwargs):
      return ', '.join([repr(k) + '=' + self.pretty_repr(v) for k, v in kwargs.items()])

    def _pretty_sub(self, string):
      rx = re.compile(r"""<tf.Tensor '(?P<name>.+?)' shape=[(](?P<shape>.*?)[)] dtype=(?P<dtype>.*?)>""")
      def sub(m):
        found = dict(m.groupdict())
        if 'dtype' in found:
          found['dtype'] = found['dtype'].replace('float', 'f')
          found['dtype'] = found['dtype'].replace('int', 'i')
        if 'shape' in found:
          found['shape'] = found['shape'].replace(', ', ',')
        return "{dtype}[{shape}, name={name!r}]".format(**found)
      return rx.sub(sub, string)

    def pretty_repr(self, v):
      s = repr(v)
      s = self._pretty_sub(s)
      return s

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        if self._input is not None:
            args, kwargs = self._input
            mod_str = self.pretty_repr(args)[1:-1]
            if len(kwargs) > 0:
              mod_str += ' ' + self.pretty_repr(kwargs)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('IN:  ' + mod_str)
        if self._output is not None:
            mod_str = self.pretty_repr(self._output)[1:-1]
            mod_str = _addindent(mod_str, 2)
            child_lines.append('OUT: ' + mod_str)
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        # if self._input is not None:
        #     args, kwargs = self._input
        #     main_str += repr(args)[1:-1]
        #     if len(kwargs) > 0:
        #       main_str += ' **' + repr(kwargs) + ','
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        # if self._output is not None:
        #     main_str += ' -> ' + repr(self._output)[1:-1]
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)




# class Sequential(Module):
#     r"""A sequential container.
#     Modules will be added to it in the order they are passed in the constructor.
#     Alternatively, an ordered dict of modules can also be passed in.

#     To make it easier to understand, here is a small example::

#         # Example of using Sequential
#         model = nn.Sequential(
#                   nn.Conv2d(1,20,5),
#                   nn.ReLU(),
#                   nn.Conv2d(20,64,5),
#                   nn.ReLU()
#                 )

#         # Example of using Sequential with OrderedDict
#         model = nn.Sequential(OrderedDict([
#                   ('conv1', nn.Conv2d(1,20,5)),
#                   ('relu1', nn.ReLU()),
#                   ('conv2', nn.Conv2d(20,64,5)),
#                   ('relu2', nn.ReLU())
#                 ]))
#     """

#     def __init__(self, *args: Any):
#         super(Sequential, self).__init__()
#         if len(args) == 1 and isinstance(args[0], OrderedDict):
#             for key, module in args[0].items():
#                 self.add_module(key, module)
#         else:
#             for idx, module in enumerate(args):
#                 self.add_module(str(idx), module)

#     def __iter__(self) -> Iterator[Module]:
#         return iter(self._modules.values())

#     def forward(self, input, *args, **kwargs):
#         for module in self:
#             input = module(input, *args, **kwargs)
#         return input



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

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args: Any, scope=None, body=None, **kwargs):
        super(Sequential, self).__init__(scope=scope, **kwargs)
        with self.scope():
          args = list(args)
          if body is not None:
            args.extend(body())
          if len(args) == 1 and isinstance(args[0], OrderedDict):
              for key, module in args[0].items():
                  self.add_module(key, module)
          else:
              for idx, module in enumerate(args):
                  self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self: T, idx) -> T:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input, *args, **kwargs):
        with self.scope():
            for module in self:
                input = module(input, *args, **kwargs)
            return input


class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    @_copy_to_script_wrapper
    def __getitem__(self, idx: int) -> Module:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self: T, modules: Iterable[Module]) -> T:
        return self.extend(modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Module) -> None:
        r"""Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self: T, module: Module) -> T:
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self: T, modules: Iterable[Module]) -> T:
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def forward(self):
        raise NotImplementedError()


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    :class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    :class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ModuleDict.update`, the order of the merged 
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~torch.nn.ModuleDict` (the argument to 
      :meth:`~torch.nn.ModuleDict.update`).

    Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    @_copy_to_script_wrapper
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    @_copy_to_script_wrapper
    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()

    def pop(self, key: str) -> Module:
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[Tuple[str, Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        r"""Update the :class:`~torch.nn.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.nn.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.nn.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.nn.Module`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]

    def forward(self):
        raise NotImplementedError()


class ParameterList(Module):
    r"""Holds parameters in a list.

    :class:`~torch.nn.ParameterList` can be indexed like a regular Python
    list, but parameters it contains are properly registered, and will be
    visible by all :class:`~torch.nn.Module` methods.

    Arguments:
        parameters (iterable, optional): an iterable of :class:`~torch.nn.Parameter` to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, parameters: Optional[Iterable['Parameter']] = None) -> None:
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    @overload
    def __getitem__(self, idx: int) -> 'Parameter':
        ...

    @overload
    def __getitem__(self: T, idx: slice) -> T:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._parameters.values())[idx])
        else:
            idx = self._get_abs_string_index(idx)
            return self._parameters[str(idx)]

    def __setitem__(self, idx: int, param: 'Parameter') -> None:
        idx = self._get_abs_string_index(idx)
        return self.register_parameter(str(idx), param)

    def __setattr__(self, key: Any, value: Any) -> None:
        if not isinstance(value, torch.nn.Parameter):
            warnings.warn("Setting attributes on ParameterList is not supported.")
        super(ParameterList, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator['Parameter']:
        return iter(self._parameters.values())

    def __iadd__(self: T, parameters: Iterable['Parameter']) -> T:
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self: T, parameter: 'Parameter') -> T:
        """Appends a given parameter at the end of the list.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self: T, parameters: Iterable['Parameter']) -> T:
        """Appends parameters from a Python iterable to the end of the list.

        Arguments:
            parameters (iterable): iterable of parameters to append
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('ParameterList should not be called.')

    def _replicate_for_data_parallel(self):
        warnings.warn("nn.ParameterList is being used with DataParallel but this is not "
                      "supported. This list will appear empty for the models replicated "
                      "on each GPU except the original one.")

        return super(ParameterList, self)._replicate_for_data_parallel()


class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~torch.nn.ParameterDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.ParameterDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.ParameterDict` (the argument to
      :meth:`~torch.nn.ParameterDict.update`).

    Note that :meth:`~torch.nn.ParameterDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.nn.Parameter`) or an iterable of key-value pairs
            of type (string, :class:`~torch.nn.Parameter`)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters: Optional[Mapping[str, 'Parameter']] = None) -> None:
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key: str) -> 'Parameter':
        return self._parameters[key]

    def __setitem__(self, key: str, parameter: 'Parameter') -> None:
        self.register_parameter(key, parameter)

    def __delitem__(self, key: str) -> None:
        del self._parameters[key]

    def __setattr__(self, key: Any, value: Any) -> None:
        if not isinstance(value, torch.nn.Parameter):
            warnings.warn("Setting attributes on ParameterDict is not supported.")
        super(ParameterDict, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._parameters

    def clear(self) -> None:
        """Remove all items from the ParameterDict.
        """
        self._parameters.clear()

    def pop(self, key: str) -> 'Parameter':
        r"""Remove key from the ParameterDict and return its parameter.

        Arguments:
            key (string): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys.
        """
        return self._parameters.keys()

    def items(self) -> Iterable[Tuple[str, 'Parameter']]:
        r"""Return an iterable of the ParameterDict key/value pairs.
        """
        return self._parameters.items()

    def values(self) -> Iterable['Parameter']:
        r"""Return an iterable of the ParameterDict values.
        """
        return self._parameters.values()

    def update(self, parameters: Mapping[str, 'Parameter']) -> None:
        r"""Update the :class:`~torch.nn.ParameterDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`parameters` is an ``OrderedDict``, a :class:`~torch.nn.ParameterDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.nn.Parameter`, or an iterable of
                key-value pairs of type (string, :class:`~torch.nn.Parameter`)
        """
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError("ParametersDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(parameters).__name__)

        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError("ParameterDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(p).__name__)
                if not len(p) == 2:
                    raise ValueError("ParameterDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(p)) +
                                     "; 2 is required")
                self[p[0]] = p[1]

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('ParameterDict should not be called.')

    def _replicate_for_data_parallel(self):
        warnings.warn("nn.ParameterDict is being used with DataParallel but this is not "
                      "supported. This dict will appear empty for the models replicated "
                      "on each GPU except the original one.")

        return super(ParameterDict, self)._replicate_for_data_parallel()


  
  
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

#Tensor = tf.Variable

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


def flatten(tensor, start_dim=1, end_dim=-1, name=None):
  dims = dim(tensor)
  if end_dim < 0:
    end_dim = dims + end_dim
  shape = shapelist(tensor)
  out_shape = shape[0:start_dim] + [-1] + shape[end_dim+1:]
  return tf.reshape(tensor, shape=out_shape, name=name)


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


def detach(v, name=None):
  return tf.stop_gradient(v, name=name)


def clone(tensor, name=None):
  if name is None:
    name = tensor.name.rsplit(':', 1)[0] + '_clone'
  return tf.identity(tensor, name=name)



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
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.eager import context
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util import compat
from tensorflow.python.training.tracking import base as trackable


def in_graph_mode():
  return not context.executing_eagerly()


def create_initializer_op(handle, initial_value, name=None, dtype=None, shape=None):
  if initial_value is None:
    raise ValueError("initial_value must be specified.")
  init_from_fn = callable(initial_value)
  #import pdb; pdb.set_trace()
  if isinstance(name, str):
    handle_name = name
    name, suffix = name.rsplit(':', 1)
    suffix = ':' + suffix
  else:
    handle_name = None
    suffix = ''
  with ops.init_scope():
    _in_graph_mode = in_graph_mode()
    with ops.name_scope(
      name,
      "Variable", [] if init_from_fn else [initial_value],
      #skip_on_eager=False, # doesn't exist in tf1.15
      ) as name:

      # pylint: disable=protected-access
      if handle_name is None:
        handle_name = ops.name_from_scope_name(name)
      if _in_graph_mode:
        shared_name = handle_name
        unique_id = shared_name
      else:
        # When in eager mode use a uid for the shared_name, to prevent
        # accidental sharing.
        unique_id = "%s_%d" % (handle_name, ops.uid())
        shared_name = None  # Never shared
      # Use attr_scope and device(None) to simulate the behavior of
      # colocate_with when the variable we want to colocate with doesn't
      # yet exist.
      device_context_manager = (
          ops.device if _in_graph_mode else ops.NullContextmanager)
      attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              s=[compat.as_bytes("loc:@%s" % handle_name)]))
      #import pdb; pdb.set_trace()
      with ops.get_default_graph()._attr_scope({"_class": attr}):
        with ops.name_scope("Initializer"), device_context_manager(None):
          if init_from_fn:
            initial_value = initial_value()
          if isinstance(initial_value, trackable.CheckpointInitialValue):
            raise NotImplementedError() # TODO
            # self._maybe_initialize_trackable()
            # self._update_uid = initial_value.checkpoint_position.restore_uid
            # initial_value = initial_value.wrapped_value
          initial_value = ops.convert_to_tensor(initial_value,
                                                name="initial_value",
                                                dtype=dtype)
        if shape is not None:
          if not initial_value.shape.is_compatible_with(shape):
            raise ValueError(
                "The initial value's shape (%s) is not compatible with "
                "the explicitly supplied `shape` argument (%s)." %
                (initial_value.shape, shape))
        else:
          shape = initial_value.shape
        handle = resource_variable_ops.eager_safe_variable_handle(
            initial_value=initial_value,
            shape=shape,
            shared_name=shared_name,
            name=name,
            graph_mode=_in_graph_mode)
      #import pdb; pdb.set_trace()
      # pylint: disable=protected-access
      if (_in_graph_mode and initial_value is not None and
          initial_value.op._get_control_flow_context() is not None):
        raise ValueError(
            "Initializer for variable %s is from inside a control-flow "
            "construct, such as a loop or conditional. When creating a "
            "variable inside a loop or conditional, use a lambda as the "
            "initializer." % handle_name)
      # pylint: enable=protected-access
      dtype = initial_value.dtype.base_dtype
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
                # variables._try_guard_against_uninitialized_dependencies(
                #     name, initial_value),
                initial_value,
                name=n))
        return initializer_op, is_initialized_op
  

def init_(tensor, value):
  # tensor is a Variable?
  assert hasattr(tensor, 'initializer')
  assert hasattr(tensor, '_initializer_op')
  # and additionaly is a ResourceVariable? TODO: handle normal variables.
  if not hasattr(tensor, 'handle'):
    raise NotImplementedError("TODO: support non-resource ops; for now just use reource ops everywhere")
  # # overwrite its initializer.
  # #tensor._initializer_op, tensor._is_initialized_op = create_initializer_op(
  # initializer_op, is_initialized_op = create_initializer_op(
  #     handle=tensor.handle,
  #     initial_value=value,
  #     name=tensor.name,
  #     )
  with ops.init_scope():
    with tf.control_dependencies([tensor.initializer]):
      initializer_op = tensor.assign(value() if callable(value) else value, read_value=False, use_locking=True)
      tf.add_to_collection('tftorch_initializers', initializer_op)



def uniform_(tensor, minval, maxval):
  # need to create value in a lambda due to creating variables in while
  # loops on tensorflow
  value = lambda: tf.random.uniform(shape=tensor.shape, minval=minval, maxval=maxval)
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


def max_pool2d(input, kernel_size, stride, padding="SAME", dilation=1, ceil_mode=False, return_indices=False, *, data_format="NHWC", name=None):
  if dilation != 1:
    import pdb; pdb.set_trace()
    raise NotImplementedError()
  if ceil_mode != False:
    import pdb; pdb.set_trace()
    raise NotImplementedError()
  if return_indices != False:
    import pdb; pdb.set_trace()
    raise NotImplementedError()
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


# class MaxPool2d(Module):
#   def __init__(self,
#       kernel_size,
#       stride=None,
#       padding=0,
#       scope='max_pool2d',
#       **kwargs):
#     super().__init__(scope=scope, **kwargs)
#     if stride is None:
#       stride = kernel_size
#     with self.scope():
#       self.kernel_size = _pair(kernel_size)
#       self.stride = _pair(stride)
#       self.padding = padding
#   def forward(self, input):
#     with self.scope():
#       return max_pool2d(input, self.kernel_size, self.stride, self.padding)


class _MaxPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool

    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None,
                 padding: _size_any_t = 0, dilation: _size_any_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False, scope=None, **kwargs) -> None:
        if scope is None:
            scope = self.__class__.scope_name
        super(_MaxPoolNd, self).__init__(scope=scope, **kwargs)
        with self.scope():
            self.kernel_size = kernel_size
            self.stride = stride if (stride is not None) else kernel_size
            self.padding = padding
            self.dilation = dilation
            self.return_indices = return_indices
            self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)


class MaxPool1d(_MaxPoolNd):
    r"""Applies a 1D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the
    sliding window. This `link`_ has a nice visualization of the pooling parameters.

    Args:
        kernel_size: The size of the sliding window, must be > 0.
        stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Examples::

        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    scope_name='max_pool1d'
    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t

    def forward(self, input: Tensor) -> Tensor:
        with self.scope():
            return max_pool1d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)


class MaxPool2d(_MaxPoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    scope_name='max_pool2d'
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: Tensor) -> Tensor:
        with self.scope():
            return max_pool2d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)


class MaxPool3d(_MaxPoolNd):
    r"""Applies a 3D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50,44, 31)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """  # noqa: E501

    scope_name='max_pool3d'
    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    def forward(self, input: Tensor) -> Tensor:
        with self.scope():
            return max_pool3d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)



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
  init_(tensor, lambda: tf.zeros_like(tensor))


def ones_(tensor):
  init_(tensor, lambda: tf.ones_like(tensor))



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
                self.register_buffer('accumulated_mean', lambda: tf.zeros(num_features))
                self.register_buffer('accumulated_var', lambda: tf.ones(num_features))
                #self.register_buffer('accumulation_counter', lambda: tf.tensor(0, dtype=torch.long))
                self.register_buffer('accumulation_counter', lambda: tf.zeros([]))
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


def softplus(input, beta=1, threshold=20) -> Tensor:
  r"""Applies element-wise, the function :math:`\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))`.

For numerical stability the implementation reverts to the linear function
when :math:`input \times \beta > threshold`.

See :class:`~torch.nn.Softplus` for more details."""
  if beta == 1:
    # fast path
    return tf.where(input > threshold, input, tf.math.softplus(input))
  else:
    x = input * beta
    return tf.where(x > threshold, x, tf.math.softplus(x) / beta)


def mean(input, dim=None, keepdim=False, *, out=None) -> Tensor:
  r"""Returns the mean value of all elements in the :attr:`input` tensor.

Args:
    {input}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.2294, -0.5481,  1.3288]])
    >>> torch.mean(a)
    tensor(0.3367)

.. function:: mean(input, dim, keepdim=False, *, out=None) -> Tensor

Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> torch.mean(a, 1)
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
    >>> torch.mean(a, 1, True)
    tensor([[-0.0163],
            [-0.5085],
            [-0.4599],
            [ 0.1807]])
  """
  if out is not None:
    raise NotImplementedError()
  return tf.reduce_mean(input, axis=dim, keepdims=keepdim)



from typing import Tuple, Union
from torch import Tensor
from torch import Size


class Flatten(Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Examples::
        >>> input = torch.randn(32, 1, 5, 5)
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([32, 288])
    """
    __constants__ = ['start_dim', 'end_dim']
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1, scope='flatten', **kwargs) -> None:
        super(Flatten, self).__init__(scope=scope, **kwargs)
        with self.scope():
            self.start_dim = start_dim
            self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        with self.scope():
            return flatten(input, self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return 'start_dim={}, end_dim={}'.format(
            self.start_dim, self.end_dim
        )


class Unflatten(Module):
    r"""
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it can
      be either `int` or `str` when `Tensor` or `NamedTensor` is used, respectively.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or `torch.Size` for `Tensor` input or a `NamedShape` (tuple of `(name, size)` tuples)
      for `NamedTensor` input.

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`

    Args:
        dim (Union[int, str]): Dimension to be unflattened
        unflattened_size (Union[torch.Size, NamedShape]): New shape of the unflattened dimension

    Examples:
        >>> input = torch.randn(2, 50)
        >>> # With tuple of ints
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(output)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With torch.Size
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, torch.Size([2, 5, 5]))
        >>> )
        >>> output = m(output)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With namedshape (tuple of tuples)
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten('features', (('C', 2), ('H', 50), ('W',50)))
        >>> )
        >>> output = m(output)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
    """
    NamedShape = Tuple[Tuple[str, int]]

    __constants__ = ['dim', 'unflattened_size']
    dim: Union[int, str]
    unflattened_size: Union[Size, NamedShape]

    def __init__(self, dim: Union[int, str], unflattened_size: Union[Size, NamedShape], scope='unflatten', **kwargs) -> None:
        super(Unflatten, self).__init__(scope=scope, **kwargs)
        with self.scope():

            if isinstance(dim, int):
                self._require_tuple_int(unflattened_size)
            elif isinstance(dim, str):
                self._require_tuple_tuple(unflattened_size)
            else:
                raise TypeError("invalid argument type for dim parameter")

            self.dim = dim
            self.unflattened_size = unflattened_size

    def _require_tuple_tuple(self, input):
        if (isinstance(input, tuple)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, tuple):
                    raise TypeError("unflattened_size must be tuple of tuples, " + 
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
            return
        raise TypeError("unflattened_size must be a tuple of tuples, " +
                        "but found type {}".format(type(input).__name__))

    def _require_tuple_int(self, input):
        if (isinstance(input, tuple)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError("unflattened_size must be tuple of ints, " + 
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
            return
        raise TypeError("unflattened_size must be a tuple of ints, but found type {}".format(type(input).__name__))

    def forward(self, input: Tensor) -> Tensor:
        with self.scope():
            return unflatten(input, self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        return 'dim={}, unflattened_size={}'.format(self.dim, self.unflattened_size)


def log_softmax(input, dim=None, _stacklevel=3, dtype=None, name=None):
    # type: (Tensor, Optional[int], int, Optional[int]) -> Tensor
    r"""Applies a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    See :class:`~torch.nn.LogSoftmax` for more details.

    Arguments:
        input (Tensor): input
        dim (int): A dimension along which log_softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    # if not torch.jit.is_scripting():
    #     if type(input) is not Tensor and has_torch_function((input,)):
    #         return handle_torch_function(
    #             log_softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    # if dim is None:
    #     dim = _get_softmax_dim('log_softmax', dim(input), _stacklevel)
    # if dtype is None:
    #     ret = input.log_softmax(dim)
    # else:
    #     ret = input.log_softmax(dim, dtype=dtype)
    if dtype is not None:
      input = tf.cast(input, dtype)
    ret = tf.nn.log_softmax(input, axis=dim, name=name)
    return ret



class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Arguments:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax()
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None, scope='log_softmax', **kwargs) -> None:
        super(LogSoftmax, self).__init__(scope=scope, **kwargs)
        with self.scope():
            self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        with self.scope():
            return log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)




class ResidualBlock(Sequential):
    r"""A residual block with an identity shortcut connection.
    As in :class:`~torch.nn.Sequential`, modules will be added to it in the
    order they are passed in the constructor, and an :class:`OrderedDict` can be
    passed instead. The final module's output will be added to the original
    input and returned. The input and output must be :ref:`broadcastable
    <broadcasting-semantics>`.
    Here is an example MNIST classifier::
        model = nn.Sequential(
            nn.Conv2d(1, 10, 1),
            nn.ResidualBlock(
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, padding=1),
            ),
            nn.MaxPool2d(2),
            nn.ResidualBlock(
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, padding=1),
            ),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7*7*10, 10),
            nn.LogSoftmax(dim=-1),
        )
    See: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual
    Learning for Image Recognition" (https://arxiv.org/abs/1512.03385), and
    "Identity Mappings in Deep Residual Networks"
    (https://arxiv.org/abs/1603.05027).
    """
    def __init__(self, *args, scope='residual', **kwargs):
        super(ResidualBlock, self).__init__(*args, scope=scope, **kwargs)

    def forward(self, input):
        with self.scope():
            output = clone(input)
            for module in self:
                output = module(output)
            return input + output


class ResidualBlockWithShortcut(ModuleDict):
    r"""A residual block with a non-identity shortcut connection.
    As in :class:`~torch.nn.Sequential`, modules will be added to the 'main'
    branch in the order they are passed in the constructor, and an
    :class:`OrderedDict` can be passed instead. The :attr:`shortcut` keyword
    argument specifies a module that performs the mapping for the shortcut
    connection.  The output of the 'main' branch will be added to the output of
    the 'shortcut' branch and returned. They must be :ref:`broadcastable
    <broadcasting-semantics>`.
    This module is useful where the 'main' branch has an output shape that is
    different from its input shape, so :class:`~torch.nn.ResidualBlock` cannot
    be used. The shortcut mapping may be used to adjust the shape of the input
    to match. Here is an example MNIST classifier::
        model = nn.Sequential(
            nn.ResidualBlockWithShortcut(
                nn.Conv2d(1, 10, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, stride=2, padding=1),
                shortcut=nn.Conv2d(1, 10, 1, stride=2, bias=False),
            ),
            nn.ResidualBlockWithShortcut(
                nn.ReLU(),
                nn.Conv2d(10, 20, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(20, 20, 3, stride=2, padding=1),
                shortcut=nn.Conv2d(10, 20, 1, stride=2, bias=False),
            ),
            nn.Flatten(),
            nn.Linear(7*7*20, 10),
            nn.LogSoftmax(dim=-1),
        )
    See: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual
    Learning for Image Recognition" (https://arxiv.org/abs/1512.03385), and
    "Identity Mappings in Deep Residual Networks"
    (https://arxiv.org/abs/1603.05027).
    """

    def __init__(self, *args, shortcut=Identity(), scope='residual_sc', **kwargs):
        super(ResidualBlockWithShortcut, self).__init__(scope=scope, **kwargs)
        with self.scope():
            self.main = Sequential(*args)
            self.shortcut = shortcut

    def forward(self, input):
        with self.scope():
            output_main = self.main(clone(input))
            output_shortcut = self.shortcut(input)
            return output_main + output_shortcut
