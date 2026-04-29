# coding=utf-8
"""Data pipeline for MultiNIST runs.

This mirrors the CIFAR input API but keeps preprocessing minimal:
"""

from absl import logging

import tensorflow.compat.v2 as tf


def _sample_uniform(shape, low, high):
    return tf.random.uniform(shape, minval=low, maxval=high, dtype=tf.float32)


def _apply_affine_to_channel(channel, angle_deg, scale, tx_frac, ty_frac):
    """
    Reproduit torch affine_grid + grid_sample (align_corners=False, bilinear, zeros).
    La matrice PyTorch mappe destination à source (convention grid_sample).
    """
    channel_4d = tf.expand_dims(tf.expand_dims(channel, 0), -1)  # (1, H, W, 1)
    h = tf.cast(tf.shape(channel)[0], tf.float32)
    w = tf.cast(tf.shape(channel)[1], tf.float32)

    theta = angle_deg * (3.141592653589793 / 180.0)
    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)

    a00 = scale * cos_t
    a01 = -scale * sin_t
    a10 = scale * sin_t
    a11 = scale * cos_t

    # Translation en pixels
    tx_pix = tx_frac * w / 2.0
    ty_pix = ty_frac * h / 2.0

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    m00 = a00
    m01 = a01 * (w / h)
    m02 = (1.0 - a00) * cx - a01 * (w / h) * cy + tx_pix
    m10 = a10 * (h / w)
    m11 = a11
    m12 = (1.0 - a11) * cy - a10 * (h / w) * cx + ty_pix

    transform = tf.stack([m00, m01, m02, m10, m11, m12, 0.0, 0.0])
    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=channel_4d,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(channel),
        interpolation='BILINEAR',
        fill_mode='CONSTANT',
        fill_value=0.0,
    )
    return tf.squeeze(transformed, axis=[0, 3])  # (H, W)


def _per_channel_random_affine(image):
    """Mirror experimental_grow PerChannelRandomAffine defaults."""
    # image: (H, W, C)
    channels_first = tf.transpose(image, [2, 0, 1])  # (C, H, W)
    c = tf.shape(channels_first)[0]
    angles = _sample_uniform([c], -5.0, 5.0)
    scales = _sample_uniform([c], 0.95, 1.05)
    tx = _sample_uniform([c], -0.05, 0.05)
    ty = _sample_uniform([c], -0.05, 0.05)

    transformed = tf.map_fn(
        lambda elems: _apply_affine_to_channel(
            elems[0], elems[1], elems[2], elems[3], elems[4]),
        (channels_first, angles, scales, tx, ty),
        fn_output_signature=tf.float32)
    return tf.transpose(transformed, [1, 2, 0])  # (H, W, C)


def build_input_fn(
    builder,
    global_batch_size,
    topology,
    is_training,
    cache_dataset=True,
):
    """Build input function compatible with the CIFAR trainer."""

    def _input_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(
            global_batch_size)
        logging.info('Global batch size: %d', global_batch_size)
        logging.info('Per-replica batch size: %d', batch_size)

        def map_fn(image, label):
            # Keep native MultiNIST geometry (28x28x3).
            image = tf.image.convert_image_dtype(image, tf.float32)
            if is_training:
                image = _per_channel_random_affine(image)
            return image, label

        dataset = builder.as_dataset(
            split='train' if is_training else 'test',
            shuffle_files=is_training,
            as_supervised=True)
        logging.info(
            'num_input_pipelines: %d', input_context.num_input_pipelines)

        if input_context.num_input_pipelines > 1:
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)
        if cache_dataset:
            dataset = dataset.cache()
        if is_training:
            dataset = dataset.shuffle(50000)
            dataset = dataset.repeat(-1)
        dataset = dataset.map(
            map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=is_training)
        prefetch_buffer_size = (
            2 * topology.num_tpus_per_task if topology else 2)
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    return _input_fn
