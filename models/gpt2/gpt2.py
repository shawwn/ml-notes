import math

import numpy as np
import tensorflow as tf
import tflex

tf1 = tf.compat.v1
tf1.disable_eager_execution()


try:
  from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
except ImportError:
  sparse_csr_matrix_ops = None


# from tensorflow.python.training import HParams

import functools


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn



@op_scope
def default_hparams(trainable=True, dtype=tf.float32, scope='model'):
    return {
        'n_vocab': 50257,
        'n_ctx': 1024,
        'n_embd': 768,
        'n_head': 12,
        'n_layer': 12,
        'res_dropout': 0.0,
        'attn_dropout': 0.0,
        'embed_dropout': 0.0,
        'dtype': dtype,
        'trainable': trainable,
        'scope': scope,
        'precision': 'bfloat16' if dtype == tf.bfloat16 else 'float32',
        'scale_by_depth': False,
        'scale_by_in': False,
    }

@op_scope
def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=tf.int64)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

@op_scope
def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

@op_scope
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

@op_scope
def norm(x, scope, *, axis=-1, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf1.variable_scope(scope):
        n_state = shape_list(x)[-1]
        if params["precision"] == "bfloat16":
            g = tf1.get_variable('g', [n_state], initializer=tf.constant_initializer(1, dtype=tf.bfloat16), dtype=tf.bfloat16)
            b = tf1.get_variable('b', [n_state], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=tf.bfloat16)
        else:
            g = tf1.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
            b = tf1.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.math.rsqrt(s + epsilon)
        x = x*g + b
        return x

@op_scope
def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, u, v = shape_list(x)
    m = u * v
    return tf.reshape(x, start + [n, m//n])

@op_scope
def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

@op_scope
def conv1d(x, scope, nf, *, w_init_stdev=0.02, params=None, scale=False):
    if params["scale_by_depth"] and scale: # Scale by sqrt(num_layers), only happens at the final projection before a res block output
        w_init_stdev = w_init_stdev * (1. / math.sqrt(params["n_layer"]))
    if params["scale_by_in"]: # Scale by sqrt(num_input_features)
        w_init_stdev = w_init_stdev * (1. / math.sqrt(shape_list(x)[-1]))

    with tf1.variable_scope(scope):
        *start, nx = shape_list(x)
        if params["precision"] == "bfloat16":
            w = tf1.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=tf.bfloat16), dtype=tf.bfloat16)
            b = tf1.get_variable('b', [nf], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=tf.bfloat16)
        else:
            w = tf1.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
            b = tf1.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

# @op_scope
# def attention_mask(nd, ns, *, dtype):
#     """1's in the lower triangle, counting from the lower right corner.

#     Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
#     """
#     i = tf.range(nd)[:,None]
#     j = tf.range(ns)
#     m = i >= j - ns + nd
#     return tf.cast(m, dtype)


@op_scope
def attention_mask(nd, ns, band=-1, bandx=0, *, dtype=tf.float32, dims=[]):
  #return tf.linalg.band_part(tf.ones([*dims, nd, ns], dtype=dtype), band, ns-nd)
  return tf.linalg.band_part(tf.ones([*dims, nd, ns], dtype=dtype), band, bandx)


@op_scope
def attn(x, scope, n_state, *, past, params, batch_size, seq_length, train=False):
    assert x.shape.ndims == 2  # Should be [batch*sequence, features]
    assert n_state % params["n_head"] == 0
    *start, hidden_size = shape_list(x)
    num_attention_heads = params["n_head"]
    assert(hidden_size % num_attention_heads == 0)
    size_per_head = hidden_size // num_attention_heads

    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    @op_scope
    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        x = tf.reshape(x, [batch_size, seq_length, num_attention_heads, size_per_head])
        x = split_states(x, params["n_head"])
        return tf.transpose(x, [0, 2, 1, 3])

    @op_scope
    def merge_heads(x):
        # Reverse of split_heads
        x = tf.transpose(x, [0, 2, 1, 3])
        x = merge_states(x)
        x = tf.reshape(x, [batch_size * seq_length, num_attention_heads * size_per_head])
        return x

    # @op_scope
    # def mask_attn_weights(w):
    #     # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
    #     _, _, nd, ns = shape_list(w)
    #     b = attention_mask(nd, ns, dtype=w.dtype)
    #     b = tf.reshape(b, [1, 1, nd, ns])
    #     w = w*b - tf.cast(1e10, w.dtype)*(1-b)
    #     return w

    @op_scope
    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        *_, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        #b = tf.reshape(b, [1, 1, nd, ns])
        b = tf.reshape(b, w.shape)
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    @op_scope
    def multihead_attn(q, k, v):
        if False:
          w = tf.matmul(q, k, transpose_b=True)
          w = w * tf.math.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))

          w = mask_attn_weights(w)
          w = softmax(w)

          w = dropout(w, params["attn_dropout"], train)

          a = tf.matmul(w, v)
          return a
        if False:
          w = tf.matmul(q, k, transpose_b=True)
          w = w * tf.math.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))
          w1 = tf.exp(w)
          w = tf.sparse.from_dense(w1)
          w / tf.sparse.reduce_sum(w, axis=-1)
          tf.reduce_sum(ex, axis=axis, keepdims=True)
        if True:
          if False:
            w = tf.matmul(q, k, transpose_b=True)
            w = w * tf.math.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))

            w = mask_attn_weights(w)
            #w = softmax(w)

            # TKTK: this works
            #w = tf.nn.softmax(w)

            #w = dropout(w, params["attn_dropout"], train)

            #import pdb; pdb.set_trace()
            #a = tf.matmul(w, v)
          if False:
            assert batch_size == 1
            w = tf.squeeze(w, axis=0)
            v = tf.squeeze(v, axis=0)
            a = tf.stack([tf.matmul(*x, a_is_sparse=True) for x in zip(tf.unstack(w, axis=0), tf.unstack(v, axis=0))])
            a = tf.expand_dims(a, axis=0)
            #a = tf.expand_dims(tf.stack([tf.matmul(*x, a_is_sparse=True) for x in list(zip(tf.unstack(tf.squeeze(w)), tf.unstack(tf.squeeze(v))))]), 0)
            return a
          else:
            B, H, S, F = shape_list(q)
            if S is None:
              S = seq_length
            # TODO: This is pretty inefficient with large batch sizes.
            #w = tf.reshape(w, [B*H, S, S])
            q = tf.reshape(q, [B*H, S, F])
            k = tf.reshape(k, [B*H, S, F])
            v = tf.reshape(v, [B*H, S, F])
            #a = tf.stack([tf.matmul(w0, v0, a_is_sparse=True) for w0, v0 in zip(tf.unstack(w, axis=0), tf.unstack(v, axis=0))])
            #a = tf.stack([tf.matmul(tf.nn.softmax(w0), v0, a_is_sparse=True) for w0, v0 in zip(tf.unstack(w, axis=0), tf.unstack(v, axis=0))])
            def process(q, k, v):
              w = tf.matmul(q, k, transpose_b=True)
              w = w * tf.math.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))
              w = mask_attn_weights(w)
              w = tf.nn.softmax(w)
              return tf.matmul(w, v, a_is_sparse=True)
            a = tf.stack([process(q0, k0, v0) for q0, k0, v0 in zip(
              tf.unstack(q, axis=0),
              tf.unstack(k, axis=0),
              tf.unstack(v, axis=0))])
            a = tf.reshape(a, [B, H, S, F])
            return a
          @op_scope
          def inner(args):
            w, v = args
            a = tf.raw_ops.SparseMatMul(a=w, b=v, a_is_sparse=True)
            return a
          assert batch_size == 1
          argz = [tf.squeeze(x, axis=0) for x in [w, v]]
          a = tf.vectorized_map(inner, argz)
          a = tf.expand_dims(a, axis=0)
          return a
          #@op_scope
          # def inner(args):
          #   @op_scope
          #   def inner2(args):
          #     w, v = args
          #     a = tf.raw_ops.SparseMatMul(a=w, b=v, a_is_sparse=True)
          #     return a
          #   return tf.vectorized_map(inner2, args)
          # #assert batch_size == 1
          # #argz = [tf.squeeze(x, axis=0) for x in [w, v]]
          # #a = tf.vectorized_map(inner, argz)
          # #a = tf.expand_dims(a, axis=0)
          # a = tf.vectorized_map(inner, [w, v])
          # return a
          
          B, H, S, F = shape_list(q)
          assert batch_size == 1
          @op_scope
          def inner(args):
            Q, K, V, W = args
            #import pdb; pdb.set_trace()
            #W = tf.matmul(Q, K, transpose_b=True)
            #W = W * tf.math.rsqrt(tf.cast(shape_list(V)[-1], W.dtype))
            M = attention_mask(seq_length, seq_length)
            W = W*M - tf.cast(1e10, W.dtype)*(1-M)
            W2 = softmax(M*W)
            #A = tf.matmul(W2, V, a_is_sparse=True)
            A = tf.raw_ops.SparseMatMul(a=W2, b=V, a_is_sparse=True)
            return A
          #tf.vectorized_map(lambda a: tf.matmul(a[0], a[1], a_is_sparse=True), [w[0], v[0]])
          #import pdb; pdb.set_trace()
          argz = [tf.squeeze(x, axis=0) for x in [q,k,v, w]]
          #a = tf.vectorized_map(inner, argz, fallback_to_while_loop=False)
          a = tf.vectorized_map(inner, argz, fallback_to_while_loop=True)
          a = tf.expand_dims(a, axis=0)
          #import pdb; pdb.set_trace()
          return a
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        B, H, S, F = shape_list(q)
        if S is None:
          S = seq_length
        if False:
          #import pdb; pdb.set_trace()
          w = tf.sparse.from_dense(w)
          #band = tf.maximum(S - 32, -1)
          band = -1
          mask = attention_mask(S, S, band, dims=[B, H])
          w1 = tf.sparse.retain(w, tf.reshape(mask, [-1]))
          # (0) Invalid argument: Compilation failure: Detected unsupported operations when trying to compile graph cluster_3_10943035373880407667[] on XLA_TPU_JIT: SparseSoftmax (No registered 'SparseSoftmax' OpKernel for XLA_TPU_JIT devices compatible with node node model_23/h0/attn/SparseSoftmax/SparseSoftmax (defined at /Users/bb/ml/ml-notes-3/models/gpt2/gpt2.py:147) )node model_23/h0/attn/SparseSoftmax/SparseSoftmax (defined at /Users/bb/ml/ml-notes-3/models/gpt2/gpt2.py:147) 
          #import pdb; pdb.set_trace()
          w2 = tf.sparse.softmax(w1)
          #w2 = w1
          w3 = tf.sparse.reshape(w2, [B*H, S, S])
          # a_st = w3
          # a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(a_st.indices, a_st.values, a_st.dense_shape)
          # a_sm = sparse_csr_matrix_ops.sparse_matrix_softmax(a_sm, tf.float32)

          v1 = tf.reshape(v, [B*H, S,F])
          #import pdb; pdb.set_trace()
          a = tf.sparse.matmul(tf.sparse.reshape(w3, [B*H*S, S]), tf.reshape(v1, [B*H*S, F]))
          # a = sparse_csr_matrix_ops.sparse_matrix_mat_mul(a_sm, v1)
          a = tf.reshape(a, [B, H, S, F])
        elif True:
          w = w * tf.math.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))

          m = attention_mask(S, S, dtype=w.dtype, dims=[B, H])
          w3 = tf.nn.softmax(m*w)
          m1 = tf.reshape(m, [B*H, S*S])
          w1 = tf.reshape(w, [B*H, S*S])
          w2 = tf.nn.softmax(m*w)
          #w2 = dropout(w2, params["attn_dropout"], train)
          v1 = tf.reshape(v, [B*H*S, F])
          a = tf.matmul(w2, v1, a_is_sparse=True, transpose_b=True)
          a = tf.reshape(a, [B, H, S, F])
          #import pdb; pdb.set_trace()
        else:
          w = w * tf.math.rsqrt(tf.cast(shape_list(v)[-1], w.dtype))

          w = mask_attn_weights(w)
          w = softmax(w)

          w = dropout(w, params["attn_dropout"], train)

          a = tf.matmul(w, v)
        return a

    with tf1.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, params=params)
        q, k, v = map(split_heads, tf.split(c, 3, axis=-1))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, params=params)
        a = dropout(a, params["res_dropout"], train)
        return a, present


@op_scope
def mlp(x, scope, n_state, *, params, train=False):
    with tf1.variable_scope(scope):
        nx = shape_list(x)[-1]
        h = gelu(conv1d(x, 'c_fc', n_state, params=params))
        h2 = conv1d(h, 'c_proj', nx, params=params, scale=True)
        h2 = dropout(h2, params["res_dropout"], train)
        return h2


@op_scope
def block(x, scope, *, past, params, attn, train=False, **attn_kws):
    with tf1.variable_scope(scope):
        nx = shape_list(x)[-1]
        ln_1 = norm(x, 'ln_1', params=params)
        a, present = attn(ln_1, 'attn', nx, past=past, params=params, train=train, **attn_kws)
        x = x + a
        ln_2 = norm(x, 'ln_2', params=params)
        m = mlp(ln_2, 'mlp', nx*4, params=params, train=train)
        x = x + m
        return x, present

@op_scope
def past_shape(*, params, batch_size=None, sequence=None):
    return [batch_size, params["n_layer"], 2, params["n_head"], sequence, params["n_embd"] // params["n_head"]]

@op_scope
def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

@op_scope
def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

@op_scope
def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x

@op_scope
def _assert_float_dtype(dtype):
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)
    return dtype


@op_scope
def model(X, params, labels=None, past=None, scope='model', reuse=False, train=False):
    with tf1.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        if params["precision"] == "bfloat16":
            wpe = tf1.get_variable('wpe', [params["n_ctx"], params["n_embd"]], # Position encoding
                             initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.bfloat16), dtype=tf.bfloat16)
            wte = tf1.get_variable('wte', [params["n_vocab"], params["n_embd"]], # Text encoding
                             initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.bfloat16), dtype=tf.bfloat16)

        else:
            wpe = tf1.get_variable('wpe', [params["n_ctx"], params["n_embd"]], # Position encoding
                                initializer=tf.random_normal_initializer(stddev=0.01))
            wte = tf1.get_variable('wte', [params["n_vocab"], params["n_embd"]], # Text encoding
                                initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]

        wpe = dropout(wpe, params["embed_dropout"], train)
        wte = dropout(wte, params["embed_dropout"], train)

        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        ## We keep the representation as a 2D tensor to avoid re-shaping it back and
        ## forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        ## the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        ## help the optimizer.
        batch_size, seq_length, hidden_size = shape_list(h)
        h = tf.reshape(h, [batch_size * seq_length, hidden_size])

        # Transformer
        presents = []
        activations = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * params["n_layer"]
        assert len(pasts) == params["n_layer"]
        checkpoint=False if 'memory_saving_gradients' not in params else params['memory_saving_gradients']
        every = 1 if 'memory_saving_checkpoints' not in params else params['memory_saving_checkpoints']
        for layer, past in enumerate(pasts):
            def block0(h):
              with tflex.absolute_variable_scope(scope, reuse=tf.AUTO_REUSE, use_resource=True):
                h1, present = block(h, 'h%d' % layer, past=past, params=params, attn=attn, train=train, batch_size=batch, seq_length=sequence)
              return h1
            @tf.custom_gradient
            def block1(x):
              #x1, present = block0(x)
              x1 = block0(x)
              def grad(dy, variables=None):
                x0 = tf.stop_gradient(x)
                y = block0(x0)
                input_grads = tf.gradients(y, x0, dy)
                if variables is not None:
                  variable_grads = tf.gradients(y, variables, dy)
                  return input_grads, variable_grads
                return input_grads
                # x2 = tf.stop_gradient(x)
                # #x1, present1 = block0(x)
                # x3 = block0(x2)
                # #result = tf.gradients(x1, x, dy)
                # result = tf.gradients(x3, variables, dy)
                # import pdb; pdb.set_trace()
                # return result
              #return (x1, present), grad
              return x1, grad
            #h, present = block1(h)
            h = block1(h)
            if checkpoint and (isinstance(every, int) and layer % every == 0 or layer in every):
                tf.logging.info('checkpointing layer %d', layer)
                tf.add_to_collection('checkpoints', h)
            #presents.append(present)
            activations.append(h)
        results['present'] = tf.stack(presents, axis=1) if len(presents) > 0 else presents
        results['activations'] = activations
        h = norm(h, 'ln_f', params=params)

        h_flat = tf.reshape(h, [batch*sequence, params["n_embd"]])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, params["n_vocab"]])
        results['logits'] = logits
        return results

