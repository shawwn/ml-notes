import tensorflow as tf
from models.gpt2 import gpt2, gpt2_rev
from optimizers import create_train_op
from metric_fns import perplexity_metric
from models.gpt2 import sample


def gpt2_rev_model(features, labels, mode, params):
    tf.logging.info('model_fns.py: gpt2_rev_model(features=%s, labels=%s, mode=%s, params=%s)', features, labels, mode, params)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        if params["precision"] == 'bfloat16':
            with tf.contrib.tpu.bfloat16_scope():
                output = gpt2_rev.model_grad(X=features,
                                    params=params,
                                    labels=labels,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN)

            output["logits"] = tf.cast(output["logits"], tf.float32)

        else:
            output = gpt2_rev.model_grad(X=features, params=params,
                                    labels=labels,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #from optimizers import create_train_op
        grads = output["grads_and_vars"]
        train_op = create_train_op(params, grads=grads)
        loss = output["loss"]

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    else:
      raise NotImplementedError()


def gpt2_model(features, labels, mode, params):
    tf.logging.info('model_fns.py: gpt2_model(features=%s, labels=%s, mode=%s, params=%s)', features, labels, mode, params)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        if params["precision"] == 'bfloat16':
            with tf.contrib.tpu.bfloat16_scope():
                output = gpt2.model(X=features, params=params,
                                    labels=labels,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN)

            output["logits"] = tf.cast(output["logits"], tf.float32)

        else:
            output = gpt2.model(X=features, params=params,
                                    labels=labels,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN)

        loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output["logits"], labels=labels)
        loss = tf.reduce_mean(loss_batch)

    if mode == tf.estimator.ModeKeys.TRAIN:
        #from optimizers import create_train_op
        train_op = create_train_op(params, loss=loss)

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    if mode == tf.estimator.ModeKeys.EVAL:
        #from metric_fns import perplexity_metric

        if params["use_tpu"]:
            # Metric inputs are transferred to CPU and must preserve batch dimension
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                loss=loss, eval_metrics=(perplexity_metric, {"loss": loss_batch}))
        else:
            return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss, eval_metric_ops=perplexity_metric(loss_batch))


    if mode == tf.estimator.ModeKeys.PREDICT:
        #from models.gpt2 import sample

        if not "top_k" in params.keys():
            params["top_k"] = 0

        output = sample.sample_sequence(
            params=params, length=min(params['length'] - params['text_len'], params["n_ctx"]),
            context=features,
            batch_size=params["batch_size"],
            temperature=1.0, top_k=params["top_k"]
        )

        predictions = {
            "tokens": output
        }

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
        else:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

