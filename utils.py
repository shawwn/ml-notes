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

