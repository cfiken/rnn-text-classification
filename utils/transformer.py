import tensorflow as tf
import numpy as np
from typing import List


def embedding(
        inputs,
        vocab_size: int,
        num_units: int,
        is_zero_pad: bool=True,
        is_scale: bool=True,
        scope: str='embedding',
        reuse: bool=None
        ):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable(
                'lookup_table',
                dtype=tf.float32,
                shape=[vocab_size, num_units],
                initializer=tf.contrib.layers.xavier_initializer())

        if is_zero_pad:
            zeros = tf.zeros(shape=[1, num_units])
            lookup_table = tf.concat((zeros, lookup_table[1:, :]), axis=0)

        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if is_scale:
            outputs = outputs * num_units ** 0.5

    return outputs


def positional_encoding(
        inputs,
        num_units: int,
        is_zero_pad: bool=True
        ):
    _, T = inputs.get_shape().as_list()
    N, _ = tf.unstack(tf.shape(inputs))

    position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # [batch_size, max_length]

    position_enc = np.array([
        [
            pos / np.power(10000, 2. * i / num_units)
            for i in range(num_units)
        ]
        for pos in range(T)
    ])

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # [max_length, num_units]

    lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
    outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

    if is_zero_pad:
        mask_parts = tf.sign(inputs)
        masks = tf.tile(tf.expand_dims(mask_parts, -1), [1, 1, num_units])  # [batch_size, num_units, num_units]
        paddings = tf.zeros_like(outputs)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    return outputs

def positional_encoding2(
        inputs,
        is_zero_pad: bool=True
        ):
    _, length, num_units = inputs.get_shape().as_list()
    batch_size, _, _ = tf.unstack(tf.shape(inputs))

    position_ind = tf.tile(tf.expand_dims(tf.range(length), 0), [batch_size, 1])  # [batch_size, max_length]

    position_enc = np.array([
        [
            pos / np.power(10000, 2. * i / num_units)
            for i in range(num_units)
        ]
        for pos in range(length)
    ])

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # [max_length, num_units]

    lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
    outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

    if is_zero_pad:
        mask_parts = tf.sign(tf.reduce_sum(inputs, axis=-1))
        masks = tf.tile(tf.expand_dims(mask_parts, -1), [1, 1, num_units])  # [batch_size, num_units, num_units]
        paddings = tf.zeros_like(outputs)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    return outputs


def step_encoding(
        inputs,
        step: int,
        is_zero_pad: bool=True
        ):
    _, max_length, num_units = inputs.get_shape().as_list()
    batch_size, _, _= tf.unstack(tf.shape(inputs))

    step_enc = np.array(
        [
            step / np.power(10000, 2. * i / num_units)
            for i in range(num_units)
        ]
    )
    step_enc[0::2] = np.sin(step_enc[0::2])
    step_enc[1::2] = np.cos(step_enc[1::2])  # [num_units]

    outputs = tf.tile(tf.expand_dims(tf.expand_dims(step_enc, 0), 0), [batch_size, max_length, 1])

    if is_zero_pad:
        mask_parts = tf.sign(tf.reduce_sum(inputs, axis=-1))
        masks = tf.tile(tf.expand_dims(mask_parts, -1), [1, 1, num_units])  # [batch_size, max_length, num_units]
        paddings = tf.zeros_like(outputs)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    return outputs




def multihead_attention(
        queries,  # [batch_size, max_length, num_units]
        keys,  # [batch_size, max_length, num_units]
        is_training,
        dropout_rate: float,
        num_units: int,
        num_heads: int=8,
        is_causality: bool=False,
        scope: str='multihead_attention',
        reuse: bool=None
        ):
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        # split: [num_heads, batch_size, max_length, num_units/num_heads]
        # concat: [batch_size*num_heads, max_length, num_units/num_heads]
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # transpose(K): [batch_size*num_heads, num_units/num_heads, max_length]
        # matmul: [batch_size*num_heads, max_length, max_length]
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (num_units / num_heads) ** 0.5

        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # [batch_size, max_length]
        key_masks = tf.tile(key_masks, [num_heads, 1])  # [batch_size*num_heads, max_length]
        key_masks = tf.expand_dims(key_masks, 1)  # [batch_size * num_heads, 1, max_length]
        # [num_heads*batch_size, queries.shape[1], max_length]
        key_masks = tf.tile(key_masks, [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        if is_causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            tril_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(tril_masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(tril_masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)  # [batch_size, queries.shape[1], keys.shape[2]]

        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # [batch_size, max_length]
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.expand_dims(query_masks, -1)
        query_masks = tf.tile(query_masks, [1, 1, tf.shape(keys)[1]])

        paddings = tf.zeros_like(outputs)
        outputs = tf.where(tf.equal(query_masks, 0), paddings, outputs)

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        outputs += queries
        outputs = normalize(outputs)

        output_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        output_masks = tf.expand_dims(output_masks, -1)
        output_masks = tf.tile(output_masks, [1, 1, tf.shape(outputs)[-1]])

        paddings = tf.zeros_like(outputs)
        outputs = tf.where(tf.equal(output_masks, 0), paddings, outputs)

    return outputs


def feedforward(
        inputs,
        num_units: List[int]=[2048, 512],
        scope: str='feedforward',
        reuse: bool=None
        ):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        outputs = tf.layers.dense(outputs, num_units[1], activation=tf.nn.relu)

        outputs += inputs
        outputs = normalize(outputs)

        output_masks = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1))
        output_masks = tf.expand_dims(output_masks, -1)
        output_masks = tf.tile(output_masks, [1, 1, tf.shape(outputs)[-1]])

        paddings = tf.zeros_like(outputs)
        outputs = tf.where(tf.equal(output_masks, 0), paddings, outputs)
    return outputs


def normalize(
        inputs,
        epsilon: float=1e-8,
        scope: str='normalize',
        reuse: bool=None
        ):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]  # ok -> [512]  ng -> 512

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable('beta', initializer=tf.zeros(params_shape))
        gamma = tf.get_variable('gamma', initializer=tf.ones(params_shape))
        normalized = (inputs - mean) / (variance + epsilon) ** (0.5)
        outputs = gamma * normalized + beta

    return outputs
