import math

import numpy as np
import tensorflow as tf

from tensorflow.contrib.training import HParams

def default_hparams(trainable=True, dtype=tf.float32, scope='model'):
    # return HParams(
    #     n_vocab=50257,
    #     n_ctx=1024,
    #     n_embd=768,
    #     n_head=12,
    #     n_layer=12,
    #     res_dropout=0.0,
    #     attn_dropout=0.0,
    #     dtype=dtype,
    #     trainable=trainable,
    #     scope=scope,
    # )
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

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        if params["precision"] == "bfloat16":
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1, dtype=tf.bfloat16), dtype=tf.bfloat16)
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=tf.bfloat16)
        else:
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, u, v = shape_list(x)
    m = u * v
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02, params=None, scale=False):
    if params["scale_by_depth"] and scale: # Scale by sqrt(num_layers), only happens at the final projection before a res block output
        w_init_stdev = w_init_stdev * (1. / math.sqrt(params["n_layer"]))
    if params["scale_by_in"]: # Scale by sqrt(num_input_features)
        w_init_stdev = w_init_stdev * (1. / math.sqrt(x.shape[-1].value))

    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        if params["precision"] == "bfloat16":
            w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=tf.bfloat16), dtype=tf.bfloat16)
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=tf.bfloat16)
        else:
            w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, params, batch_size, seq_length, train=False):
    assert x.shape.ndims == 2  # Should be [batch*sequence, features]
    assert n_state % params["n_head"] == 0
    *start, hidden_size = shape_list(x)
    num_attention_heads = params["n_head"]
    assert(hidden_size % num_attention_heads == 0)
    size_per_head = hidden_size // num_attention_heads

    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        x = tf.reshape(x, [batch_size, seq_length, num_attention_heads, size_per_head])
        x = split_states(x, params["n_head"])
        return tf.transpose(x, [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        x = tf.transpose(x, [0, 2, 1, 3])
        x = merge_states(x)
        x = tf.reshape(x, [batch_size * seq_length, num_attention_heads * size_per_head])
        return x

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)

        w = dropout(w, params["attn_dropout"], train)

        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
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


def mlp(x, scope, n_state, *, params, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state, params=params))
        h2 = conv1d(h, 'c_proj', nx, params=params, scale=True)
        h2 = dropout(h2, params["res_dropout"], train)
        return h2


def combine(concat, *argv):
  if concat:
    y = tf.concat(list(argv), axis=-1)
  else:
    y = tuple(argv)
  return y


def split(concat, x):
  if concat or type(x) != tuple:
    n = x.shape[-1].value
    x1 = x[..., :n // 2]
    x2 = x[..., n // 2:]
  else:
    x1, x2 = x
  return x1, x2


def block(x, scope, *, past, params, attn, train=False, **attn_kws):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        ln_1 = norm(x, 'ln_1', params=params)
        a, present = attn(ln_1, 'attn', nx, past=past, params=params, train=train, **attn_kws)
        x = x + a
        ln_2 = norm(x, 'ln_2', params=params)
        m = mlp(ln_2, 'mlp', nx*4, params=params, train=train)
        x = x + m
        return x, present


def residual(x, scope, *, past, params, attn, train=False, concat=False, **attn_kws):
    x1, x2 = split(concat, x)
    with tf.variable_scope("f"):
      f_x2, f_x2_present = block(x2, scope, past=past, params=params, attn=attn, train=train, **attn_kws)
    y1 = f_x2 + x1
    with tf.variable_scope("g"):
      f_y1, f_y1_present = block(y1, scope, past=past, params=params, attn=attn, train=train, **attn_kws)
    y2 = f_y1 + x2
    # TODO: How to deal with presents?
    return combine(concat, y1, y2), None


def residual_backward(y, scope, *, past, params, attn, train=False, concat=False, **attn_kws):
    y1, y2 = split(concat, y)
    with tf.variable_scope("g"):
      f_y1, f_y1_present = block(y1, scope, past=past, params=params, attn=attn, train=train, **attn_kws)
    x2 = y2 - f_y1
    with tf.variable_scope("f"):
      f_x2, f_x2_present = block(x2, scope, past=past, params=params, attn=attn, train=train, **attn_kws)
    x1 = y1 - f_x2
    # TODO: How to deal with presents?
    return combine(concat, x1, x2), None


def residual_grad(x, dy, scope, *, past, params, attn, train=False, concat=False, **attn_kws):
    x1, x2 = split(concat, x)
    x1, x2 = tf.stop_gradient(x1), tf.stop_gradient(x2)
    dy1, dy2 = split(concat, dy)
    y, present = residual((x1, x2), scope=scope, past=past, params=params, attn=attn, train=train, concat=False, **attn_kws)
    y1, y2 = y

    # F function weights.
    fw_list = tf.trainable_variables(tf.get_variable_scope().name + '/f/' + scope + '/')
    # G function weights.
    gw_list = tf.trainable_variables(tf.get_variable_scope().name + '/g/' + scope + '/')

    dd1 = tf.gradients(y2, [y1] + gw_list, dy2, gate_gradients=True)
    dy2_y1 = dd1[0]
    dy1_plus = dy2_y1 + dy1
    dgw = dd1[1:]
    dd2 = tf.gradients(y1, [x1, x2] + fw_list, dy1_plus, gate_gradients=True)
    dx1 = dd2[0]
    dx2 = dd2[1]
    dfw = dd2[2:]
    dx2 += tf.gradients(x2, x2, dy2, gate_gradients=True)[0]

    dw_list = list(dfw) + list(dgw)
    w_list = list(fw_list) + list(gw_list)

    # Inject dw dependency.
    with tf.control_dependencies(dw_list):
      dx = combine(concat, tf.identity(dx1), tf.identity(dx2))

    return dx, w_list, dw_list


def past_shape(*, params, batch_size=None, sequence=None):
    return [batch_size, params["n_layer"], 2, params["n_head"], sequence, params["n_embd"] // params["n_head"]]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x

def _assert_float_dtype(dtype):
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)
    return dtype


def model(X, params, labels=None, past=None, scope='model', reuse=False, train=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        if params["precision"] == "bfloat16":
            wpe = tf.get_variable('wpe', [params["n_ctx"], params["n_embd"]], # Position encoding
                             initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.bfloat16), dtype=tf.bfloat16)
            wte = tf.get_variable('wte', [params["n_vocab"], params["n_embd"]], # Text encoding
                             initializer=tf.random_normal_initializer(stddev=0.02, dtype=tf.bfloat16), dtype=tf.bfloat16)

        else:
            wpe = tf.get_variable('wpe', [params["n_ctx"], params["n_embd"]], # Position encoding
                                initializer=tf.random_normal_initializer(stddev=0.01))
            wte = tf.get_variable('wte', [params["n_vocab"], params["n_embd"]], # Text encoding
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
            h, present = residual(h, 'h%d' % layer, past=past, params=params, attn=attn, train=train, batch_size=batch, seq_length=sequence)
            if checkpoint and (isinstance(every, int) and layer % every == 0 or layer in every):
                tf.logging.info('checkpointing layer %d', layer)
                tf.add_to_collection('checkpoints', h)
            if present is not None:
              presents.append(present)
            activations.append(h)
        results['present'] = tf.stack(presents, axis=1) if len(presents) > 0 else None
        results['activations'] = activations
        h = combine(True, *h)
        h = norm(h, 'ln_f', params=params)

        h_flat = tf.reshape(h, [batch*sequence, params["n_embd"]])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, params["n_vocab"]])
        results['logits'] = logits
        #results['loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=X[:, 1:], logits=logits[:, :-1]))
        return results


def model_grad(X, params, labels=None, past=None, scope='model', reuse=tf.AUTO_REUSE, train=False, recompute=False):
    results = model(X=X, params=params, reuse=reuse, train=train)
    with tf.variable_scope(scope, reuse=reuse):
        grads_list = []
        vars_list = []

        wpe = tf.get_variable('wpe')
        wte = tf.get_variable('wte')
        gamma_final = tf.get_variable('ln_f/g')
        beta_final = tf.get_variable('ln_f/b')
        var_final = [wte, gamma_final, beta_final]
        batch, sequence = shape_list(X)
        past_length = 0 if past is None else tf.shape(past)[-2]

        h1, h2 = results['activations'][-1]
        h1, h2 = tf.stop_gradient(h1), tf.stop_gradient(h2)
        h = combine(True, h1, h2)
        h = norm(h, 'ln_f', params=params)
        h_flat = tf.reshape(h, [batch*sequence, params["n_embd"]])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, params["n_vocab"]])
        results['loss'] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=X[:, 1:], logits=logits[:, :-1]))

        _grads = tf.gradients(results['loss'], [h1, h2] + var_final, gate_gradients=True)
        dh1, dh2 = _grads[0], _grads[1]
        _grads = _grads[2:]

        # Injected dependency.
        with tf.control_dependencies(_grads):
          h_grad = (tf.identity(dh1), tf.identity(dh2))
        grads_list.extend(_grads)
        # grads_list.extend(_grads[2:])
        vars_list.extend(var_final)


        h1, h2 = results['activations'][-1]
        h1, h2 = tf.stop_gradient(h1), tf.stop_gradient(h2)
        h = (h1, h2)

        #pasts = tf.unstack(past, axis=1) if past is not None else [None] * params["n_layer"]
        nlayers = params["n_layer"]
        for layer in range(nlayers - 1, -1, -1):
          # reconstruct input.
          if layer > 0 and recompute:
            h, present = residual_backward(h, 'h%d' % layer, past=past, params=params, attn=attn, train=train, batch_size=batch, seq_length=sequence)
          else:
            h = results['activations'][layer]

          # rerun the layer, and get gradients.
          h_grad, w_list, w_grad = residual_grad(h, h_grad, 'h%d' % layer, past=past, params=params, attn=attn, train=train, batch_size=batch, seq_length=sequence)

          grads_list.extend(w_grad)
          vars_list.extend(w_list)

        def init_conv_grad(y, dy):
          wpe = tf.get_variable("wpe")
          wte = tf.get_variable("wte")
          w_list = [wpe, wte]
          dw_list = tf.gradients(y, w_list, dy)
          return w_list, dw_list

        # h1, h2 = results['activations'][0]
        # # h1, h2 = tf.stop_gradient(h1), tf.stop_gradient(h2)
        # h = (h1, h2)
        h = combine(True, h)
        h_grad = combine(True, h_grad)
        var_init, _grads = init_conv_grad(h, h_grad)
        grads_list.extend(_grads)
        vars_list.extend(var_init)

        results['grads_and_vars'] = list(zip(grads_list, vars_list))

        return results

