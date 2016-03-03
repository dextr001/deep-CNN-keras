# A parser for a config file parameters. An example config file is available in
# params.config.


class ModelParams(object):
  """An object that defines a model's training parameters."""

  def __init__(self):
    """Sets up default values for each of the configuration values."""
    # Initialize all required config options mapping to 'None':
    config_params = [
      'classnames_file',
      'train_img_paths_file',
      'test_img_paths_file',
      'number_of_classes',
      'train_imgs_per_class',
      'test_imgs_per_class',
      'img_dimensions'
    ]
    self._params = {param: None for param in config_params}
    # Initialize default hyperparameters:
    self._params['batch_size'] = 32
    self._params['num_epochs'] = 10
    self._params['learning_rate'] = 0.01
    self._params['decay'] = 1e-6
    self._params['momentum'] = 0.9

  def __getitem__(self, key):
    """Overload brackets operator to get config values.
  
    Usage: e.g. params['classnames_file']

    Args:
      key: the key (name) of the config value.

    Returns:
      The value (if available) of that configuration parameter. If the key is
      invalid or no config value was specified, returns None.
    """
    if key in self._params:
      return self._params[key]
    return None

  def read_config_file(self, fname):
    """Reads the config parameters from the given config file.

    Args:
      fname: the filename of a correctly-formatted configuration file.

    Returns:
      False if any of the required parameters was not set.
    """
    # TODO: maybe use more clearly-defined image path files.
    config = open(fname, 'r')
    line_num = 0
    for line in config:
      line_num += 1
      line = line.strip()
      if len(line) == 0 or line.startswith('#'):
        continue
      parts = line.split()
      if len(parts) < 2:
        print 'Error: invalid config value "{}" on line {} of {}'.format(
            line, line_num, fname)
        continue
      key = parts[0]
      key = key.replace(':', '')
      value = ' '.join(parts[1:])
      if key in self._params:
        try:
          self._params[key] = eval(value)
        except:
          print 'Error: invalid config value "{}" on line {} of {}'.format(
              value, line_num, fname)
      else:
        print 'Error: unknown config key "{}" on line {} of {}'.format(
            key, line_num, fname)
    # Check that all parameters were defined.
    for key in self._params:
      if not self._params[key]:
        print 'Error: config parameter "{}" was not specified.'.format(key)
        return False
    return True
