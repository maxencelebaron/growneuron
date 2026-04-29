"""Big baseline for Gutenberg."""
from growneuron.gutenberg.configs import baseline_small


def get_config():
  """Builds and returns config."""
  config = baseline_small.get_config()
  config.model.block_width_multiplier = 1.
  return config
