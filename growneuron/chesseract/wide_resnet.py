"""Chesseract-specific WRN wrapper.

Keeps CIFAR WRN implementation unchanged except final pooling geometry.
"""

from growneuron.cifar import wide_resnet as cifar_wide_resnet


def create_model(**kwargs):
    """Create WRN configured for Chesseract spatial resolution (8x8)."""
    kwargs.setdefault('final_pool_size', 2)
    return cifar_wide_resnet.create_model(**kwargs)

