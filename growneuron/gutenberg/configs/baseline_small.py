"""Baseline config for Gutenberg runs."""
import ml_collections


def get_config():
  """Builds and returns config."""
  config = ml_collections.ConfigDict()
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.base_learning_rate = 0.025
  config.optimizer.decay_type = 'cosine'
  config.optimizer.nesterov = False
  config.optimizer.lr_decay_ratio = 0.2
  config.optimizer.lr_decay_epochs = [0.3, 0.6, 0.8]
  config.optimizer.lr_warmup_epochs = 1
  config.optimizer.momentum = 0.9

  config.updater = ml_collections.ConfigDict()
  config.updater.carry_optimizer = False
  config.is_outgoing_zero = False
  config.scale_epochs = False

  config.model = ml_collections.ConfigDict()
  config.model.l2_coef = 3e-4
  config.model.depth = 28
  config.model.width_multiplier = 1
  config.model.normalization_type = 'batchnorm'
  config.model.block_width_multiplier = 0.25

  config.checkpoint_interval = 50
  config.dataset = 'gutenberg'
  config.cache_dataset = True
  config.per_core_batch_size = 128
  config.num_cores = 1
  config.seed = 3
  config.train_epochs = 200
  config.log_freq = 50
  return config
