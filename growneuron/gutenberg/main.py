"""GradMax on Gutenberg without TFDS registration.

This wrapper reuses the CIFAR training pipeline
`tfds.builder(...)` so that `config.dataset=gutenberg` loads local NPY files
(`train_x/y`, `valid_x/y`, `test_x/y`) directly.
"""

from __future__ import annotations

import os
import types
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from growneuron.cifar import main as cifar_main
from growneuron.gutenberg import data as gutenberg_data
from growneuron.gutenberg import wide_resnet as gutenberg_wide_resnet


class _GutenbergBuilder:
    """Small builder object mimicking the subset of TFDS API we need."""

    def __init__(self, root_dir: str):
        self._root_dir = root_dir
        train_x = np.load(os.path.join(root_dir, 'train_x.npy'))
        valid_x = np.load(os.path.join(root_dir, 'valid_x.npy'))
        test_x = np.load(os.path.join(root_dir, 'test_x.npy'))
        train_y = np.load(os.path.join(root_dir, 'train_y.npy'))
        valid_y = np.load(os.path.join(root_dir, 'valid_y.npy'))
        test_y = np.load(os.path.join(root_dir, 'test_y.npy'))

        # Match existing MultiNIST convention used elsewhere: train split
        # includes both train and valid subsets.
        self._train_x = np.concatenate([train_x, valid_x], axis=0)
        self._train_y = np.concatenate([train_y, valid_y], axis=0)
        self._test_x = test_x
        self._test_y = test_y

        # Data files are stored as (N, C, H, W).
        # Convert to (N, H, W, C) for TF.
        self._train_x = np.transpose(
            self._train_x, (0, 2, 3, 1)).astype(np.float32)
        self._test_x = np.transpose(
            self._test_x, (0, 2, 3, 1)).astype(np.float32)
        self._train_y = self._train_y.astype(np.int64)
        self._test_y = self._test_y.astype(np.int64)

        # Input shape after NHWC conversion: (27, 18, 1)
        self.input_shape = self._train_x.shape[1:]
        self.info = types.SimpleNamespace(
            splits={
                'train': types.SimpleNamespace(
                    num_examples=int(self._train_x.shape[0])),
                'test': types.SimpleNamespace(
                    num_examples=int(self._test_x.shape[0])),
            },
            features={'label': types.SimpleNamespace(num_classes=6)},
        )

    def download_and_prepare(self):
        # Data is expected to be present locally.
        return

    def as_dataset(
            self,
            split: str,
            shuffle_files: bool = False,
            as_supervised: bool = True):
        del shuffle_files
        if split == 'train':
            x, y = self._train_x, self._train_y
        elif split == 'test':
            x, y = self._test_x, self._test_y
        else:
            raise ValueError(
                f'Unsupported split for Gutenberg builder: {split}')
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if not as_supervised:
            ds = ds.map(lambda image, label: {'image': image, 'label': label})
        return ds


def _resolve_gutenberg_root(data_dir: Optional[str]) -> str:
    """Resolve dataset root that contains Gutenberg_extracted/*.npy files."""
    candidates = []
    if data_dir:
        candidates.extend([
            os.path.join(data_dir, 'Gutenberg_extracted'),
            os.path.join(data_dir, 'datasets', 'Gutenberg_extracted'),
        ])
    candidates.extend([
        os.path.join(
            os.path.expanduser('~'),
            'data',
            'datasets',
            'Gutenberg_extracted'),
        '/scratch/sdouka/data/Gutenberg_extracted',
    ])

    required = ('train_x.npy', 'train_y.npy', 'valid_x.npy', 'valid_y.npy',
                'test_x.npy', 'test_y.npy')
    for root in candidates:
        if all(os.path.exists(os.path.join(root, name)) for name in required):
            return root
    raise FileNotFoundError(
        'Could not find Gutenberg files. Checked: ' + ', '.join(candidates))


_ORIGINAL_TFDS_BUILDER = tfds.builder


def _patched_builder(name: str, *args, **kwargs):
    if name == 'gutenberg':
        root = _resolve_gutenberg_root(cifar_main.FLAGS.data_dir)
        return _GutenbergBuilder(root)
    return _ORIGINAL_TFDS_BUILDER(name, *args, **kwargs)


tfds.builder = _patched_builder
# Use dataset-specific WRN head geometry.
cifar_main.data = gutenberg_data
cifar_main.wide_resnet = gutenberg_wide_resnet


if __name__ == '__main__':
    cifar_main.app.run(cifar_main.main)
