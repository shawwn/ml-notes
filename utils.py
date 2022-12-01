import tensorflow as tf


# # A highly simplified convenience class for sampling from distributions
# # One could also use PyTorch's inbuilt distributions package.
# # Note that this class requires initialization to proceed as
# # x = Distribution(torch.randn(size))
# # x.init_distribution(dist_type, **dist_kwargs)
# # x = x.to(device,dtype)
# # This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
# class Distribution(torch.Tensor):
#   # Init the params of the distribution
#   def init_distribution(self, dist_type, **kwargs):    
#     self.dist_type = dist_type
#     self.dist_kwargs = kwargs
#     if self.dist_type == 'normal':
#       self.mean, self.var = kwargs['mean'], kwargs['var']
#     elif self.dist_type == 'categorical':
#       self.num_categories = kwargs['num_categories']

#   def sample_(self):
#     if self.dist_type == 'normal':
#       self.normal_(self.mean, self.var)
#     elif self.dist_type == 'categorical':
#       self.random_(0, self.num_categories)    
#     # return self.variable
    
#   # Silly hack: overwrite the to() method to wrap the new object
#   # in a distribution as well
#   def to(self, *args, **kwargs):
#     new_obj = Distribution(self)
#     new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
#     new_obj.data = super().to(*args, **kwargs)    
#     return new_obj


# # Convenience function to prepare a z and y vector
# def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', 
#                 fp16=False,z_var=1.0):
#   z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
#   z_.init_distribution('normal', mean=0, var=z_var)
#   z_ = z_.to(device,torch.float16 if fp16 else torch.float32)   
  
#   if fp16:
#     z_ = z_.half()

#   y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
#   y_.init_distribution('categorical',num_categories=nclasses)
#   y_ = y_.to(device, torch.int64)
#   return z_, y_

def distribution(dist_type, shape, *, seed=None, **kwargs):
  if dist_type == 'normal':
    if seed is None:
      return tf.random.normal(shape=shape, **kwargs)
    else:
      return tf.random.stateless_normal(shape=shape, seed=seed, **kwargs)
  elif dist_type == 'categorical':
    num_categories = kwargs.pop('num_categories')
    if seed is None:
      return tf.random.uniform(shape=shape, minval=0, maxval=num_categories, dtype=tf.int64)
    else:
      return tf.random.stateless_uniform(shape=shape, seed=seed, minval=0, maxval=num_categories, dtype=tf.int64)
  else:
    raise NotImplementedError()

def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', 
                fp16=False,z_var=1.0, seed=None):
  #z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  #z_.init_distribution('normal', mean=0, var=z_var)
  #z_ = z_.to(device,torch.float16 if fp16 else torch.float32)   
  z_ = distribution('normal', shape=[G_batch_size, dim_z], mean=0.0, stddev=z_var, seed=seed)
  
  if fp16:
    #z_ = z_.half()
    z_ = tf.cast(z_, tf.float16)

  # y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
  # y_.init_distribution('categorical',num_categories=nclasses)
  # y_ = y_.to(device, torch.int64)
  y_ = distribution('categorical', shape=[G_batch_size], num_categories=nclasses, seed=seed)
  assert y_.dtype == tf.int64
  return z_, y_

# From tinygrad

import pickle
import numpy as np
from math import prod

def fetch(url):
  if url.startswith("/"):
    with open(url, "rb") as f:
      dat = f.read()
    return dat
  import requests, os, hashlib, tempfile
  fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp) and os.stat(fp).st_size > 0 and os.getenv("NOCACHE", None) is None:
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    print("fetching %s" % url)
    r = requests.get(url)
    assert r.status_code == 200
    dat = r.content
    with open(fp+".tmp", "wb") as f:
      f.write(dat)
    os.rename(fp+".tmp", fp)
  return dat

def my_unpickle(fb0):
  key_prelookup = {}
  class HackTensor:
    def __new__(cls, *args):
      #print(args)
      ident, storage_type, obj_key, location, obj_size = args[0][0:5]
      assert ident == 'storage'

      assert prod(args[2]) == obj_size
      ret = np.zeros(args[2], dtype=storage_type)
      key_prelookup[obj_key] = (storage_type, obj_size, ret, args[2], args[3])
      return ret

  class HackParameter:
    def __new__(cls, *args):
      #print(args)
      pass

  class Dummy:
    pass

  class MyPickle(pickle.Unpickler):
    def find_class(self, module, name):
      #print(module, name)
      if name == 'FloatStorage':
        return np.float32
      if name == 'LongStorage':
        return np.int64
      if name == 'HalfStorage':
        return np.float16
      if module == "torch._utils":
        if name == "_rebuild_tensor_v2":
          return HackTensor
        elif name == "_rebuild_parameter":
          return HackParameter
      else:
        try:
          return pickle.Unpickler.find_class(self, module, name)
        except Exception:
          return Dummy

    def persistent_load(self, pid):
      return pid

  return MyPickle(fb0).load(), key_prelookup

def fake_torch_load_zipped(fb0, load_weights=True):
  import zipfile
  with zipfile.ZipFile(fb0, 'r') as myzip:
    with myzip.open('archive/data.pkl') as myfile:
      ret = my_unpickle(myfile)
    if load_weights:
      for k,v in ret[1].items():
        with myzip.open(f'archive/data/{k}') as myfile:
          if v[2].dtype == "object":
            print(f"issue assigning object on {k}")
            continue
          np.copyto(v[2], np.frombuffer(myfile.read(), v[2].dtype).reshape(v[3]))
  return ret[0]

def fake_torch_load(b0):
  import io
  import struct

  # convert it to a file
  fb0 = io.BytesIO(b0)

  if b0[0:2] == b"\x50\x4b":
    return fake_torch_load_zipped(fb0)

  # skip three junk pickles
  pickle.load(fb0)
  pickle.load(fb0)
  pickle.load(fb0)

  ret, key_prelookup = my_unpickle(fb0)

  # create key_lookup
  key_lookup = pickle.load(fb0)
  key_real = [None] * len(key_lookup)
  for k,v in key_prelookup.items():
    key_real[key_lookup.index(k)] = v

  # read in the actual data
  for storage_type, obj_size, np_array, np_shape, np_strides in key_real:
    ll = struct.unpack("Q", fb0.read(8))[0]
    assert ll == obj_size
    bytes_size = {np.float32: 4, np.int64: 8}[storage_type]
    mydat = fb0.read(ll * bytes_size)
    np.copyto(np_array, np.frombuffer(mydat, storage_type).reshape(np_shape))

    # numpy stores its strides in bytes
    real_strides = tuple([x*bytes_size for x in np_strides])
    np_array.strides = real_strides

  return ret

def get_child(parent, key):
  obj = parent
  for k in key.split('.'):
    if k.isnumeric():
      obj = obj[int(k)]
    elif isinstance(obj, dict):
      obj = obj[k]
    else:
      obj = getattr(obj, k)
  return obj

