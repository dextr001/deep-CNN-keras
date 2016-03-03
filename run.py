# Trains a deep CNN on a subset of ImageNet data.
#
# Run with GPU:
#   THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run.py [options]
#
# This example is from here:
#   https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
#
# CNN layer classes documentation:
#   http://keras.io/layers/convolutional/

import argparse
from img_loader import ImageInfo, ImageLoader
from keras.optimizers import SGD
from model import build_model
import time


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

  def read_config_file(self, fname):
    """Reads the config parameters from the given config file.

    Args:
      fname: the filename of a correctly-formatted configuration file.

    Returns:
      False if any of the required parameters was not set.
    """
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


def get_elapsed_time(start_time):
  """Returns the elapsed time, formatted as a string.
  
  Args:
    start_time: the start time (called before timing using time.time()).

  Returns:
    The elapsed time as a string (e.g. "x seconds" or "x minutes").
  """
  elapsed = time.time() - start_time
  time_units = ['seconds', 'minutes', 'hours', 'days']
  unit_index = 0
  intervals = [60, 60, 24]
  for interval in intervals:
    if elapsed < interval:
      break
    elapsed /= interval
    unit_index += 1
  elapsed = round(elapsed, 2)
  return '{} {}'.format(elapsed, time_units[unit_index])


def test_model(args):
  """Tests a model on the test data set.
  
  Args:
    args: the arguments from argparse that contains all user-specified options.
        The model weights that are to be tested must be provided.
  """
  if not args.load_weights:
    print 'Cannot test model: no weights provided.'
    return
  #img_info = ImageInfo(25, 100, 20)
  #img_info.load_test_image_paths('test/test1ImNames.txt')
  #model = build_model(img_info.num_channels, img_info.img_width,
  #                    img_info.img_height, img_info.num_classes)

def train_model(args):
  """Trains a model on the training data.

  The test data is used to report validation accuracy after each training epoch.
  The model can be trained from scratch, or existing weights can be updated.
  
  Args:
    args: the arguments from argparse that contains all user-specified options.
  """
  # Set the data parameters and image source paths.
  # TODO: use more clearly-defined image path files.
  # TODO: all of these numbers should be parameters.
  params = ModelParams()
  img_info = ImageInfo(25, 100, 20)
#  img_info = ImageInfo(25, 1160, 20)
  img_info.set_image_dimensions(128, 128, 1)  # img width, height, channels
  img_info.load_image_classnames('test/classnames.txt')
  img_info.load_train_image_paths('test/trainImNames.txt')
#  img_info.load_train_image_paths('test/extraTrainImNames.txt')
  img_info.load_test_image_paths('test/test1ImNames.txt')
  
  # Load the images into memory and preprocess appropriately.
  start_time = time.time()
  img_loader = ImageLoader(img_info)
  img_loader.load_all_images()
  print 'Data successfully loaded in {}.'.format(get_elapsed_time(start_time))
  # Get the deep CNN model for the given data.
  model = build_model(img_info.num_channels, img_info.img_width,
                      img_info.img_height, img_info.num_classes)
  if args.load_weights:
    model.load_weights(args.load_weights)
    print 'Loaded existing model weights from {}.'.format(args.load_weights)
  # Compile the model with SGD + momentum.
  print ('Compiling module...')
  start_time = time.time()
  sgd = SGD(lr=params.learning_rate, decay=params.decay,
            momentum=params.momentum, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)
  print 'Done in {}.'.format(get_elapsed_time(start_time))
  # Train the model.
  start_time = time.time()
  if not params.data_augmentation:
    print ('Training without data augmentation.')
    model.fit(img_loader.train_data, img_loader.train_labels,
              validation_data=(img_loader.test_data, img_loader.test_labels),
              batch_size=params.batch_size, nb_epoch=params.num_epochs,
              shuffle=True, show_accuracy=True, verbose=1)
  else:
    print ('Training with additional data augmentation.')
    # TODO: implement this!
  print 'Finished training in {}.'.format(get_elapsed_time(start_time))
  # Save the model if that option was specified.
  if args.save_weights:
    model.save_weights(args.save_weights)
    print 'Saved trained model weights to {}.'.format(args.save_weights)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run a deep neural network model using Keras.')
  parser.add_argument('params_file',
                      help='The file containing data paths and model params.')
  parser.add_argument('--test', dest='test_mode', action='store_true',
                      help='Test the model with weights (-load-weights).')
  parser.add_argument('-save-weights', dest='save_weights', required=False,
                      help='Save the trained weights to this file.')
  parser.add_argument('-load-weights', dest='load_weights', required=False,
                      help='Load existing weights from this file.')
  args = parser.parse_args()
  params = ModelParams()
  if not params.read_config_file(args.params_file):
    print 'Missing configuration values. Cannot continue.'
    exit(0)
#  if args.test_mode:
#    test_model(args)
#  else:
#    train_model(args)
