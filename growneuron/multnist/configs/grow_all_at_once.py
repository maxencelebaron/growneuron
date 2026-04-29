"""All-at-once growing config for MultiNIST."""
from growneuron.multnist.configs import baseline_small


def get_config():
  """Builds and returns config."""
  config = baseline_small.get_config()
  config.updater_type = 'all_at_once'
  config.grow_type = 'add_gradmax'
  config.grow_batch_size = 128
  config.grow_epsilon = 0.
  config.grow_scale_method = 'mean_norm'
  config.updater.carry_optimizer = True
  config.updater.update_frequency = 2500
  config.updater.start_iteration = 10000
  config.updater.n_growth_steps = 12
  config.updater.n_grow_fraction = 0.25
  config.updater.scale = 0.5
  return config
