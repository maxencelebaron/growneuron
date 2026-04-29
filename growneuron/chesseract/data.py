# coding=utf-8
"""Data pipeline for Chesseract runs.

Matches experimental_grow defaults for Chesseract: no data augmentation.
"""

from absl import logging

import tensorflow.compat.v2 as tf


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
            image = tf.image.convert_image_dtype(image, tf.float32)
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
