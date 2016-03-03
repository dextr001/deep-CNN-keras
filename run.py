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
from model_params import ModelParams
import time


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


def test_model(args, params):
  """Tests a model on the test data set.
  
  Args:
    args: the arguments from argparse that contains all user-specified options.
        The model weights that are to be tested must be provided.
    params: a ModelParams object containing the appropriate data file paths and
        data parameters.
  """
  if not args.load_weights:
    print 'Cannot test model: no weights provided.'
    return
  img_info = ImageInfo(params['number_of_classes'],
                       params['train_imgs_per_class'],
                       params['test_imgs_per_class'])
  img_info.load_image_classnames(params['classnames_file'])
  img_info.load_test_image_paths(params['test_img_paths_file'])
  # Load the images into memory and preprocess appropriately.
  start_time = time.time()
  img_loader = ImageLoader(img_info)
  img_loader.load_test_images()
  print 'Test data successfully loaded in {}.'.format(
      get_elapsed_time(start_time))
  # Build and compile the model and load its weights from the file.
  model = build_model(img_info.num_channels, img_info.img_width,
                      img_info.img_height, img_info.num_classes)
  model.load_weights(args.load_weights)
  print ('Compiling module...')
  start_time = time.time()
  model.compile()
  print 'Done in {}.'.format(get_elapsed_time(start_time))
  # Run the prediction on the test data.
  start_time = time.time()
  model.predict(X, batch_size=params['batch_size'], verbose=0)
  print 'Finished testing in {}.'.format(get_elapsed_time(start_time))


def train_model(args, params):
  """Trains a model on the training data.

  The test data is used to report validation accuracy after each training epoch.
  The model can be trained from scratch, or existing weights can be updated.
  
  Args:
    args: the arguments from argparse that contains all user-specified options.
    params: a ModelParams object containing the appropriate data file paths,
        data parameters, and training hyperparameters.
  """
  # Set the data parameters and image source paths.
  img_info = ImageInfo(params['number_of_classes'],
                       params['train_imgs_per_class'],
                       params['test_imgs_per_class'])
  img_info.set_image_dimensions(params['img_dimensions'])
  img_info.load_image_classnames(params['classnames_file'])
  img_info.load_train_image_paths(params['train_img_paths_file'])
  img_info.load_test_image_paths(params['test_img_paths_file'])
  # Load the images into memory and preprocess appropriately.
  start_time = time.time()
  img_loader = ImageLoader(img_info)
  img_loader.load_all_images()
  print 'Data successfully loaded in {}.'.format(get_elapsed_time(start_time))
  # Get the deep CNN model for the given data.
  # TODO: add option to load model from a saved file.
  model = build_model(img_info.num_channels, img_info.img_width,
                      img_info.img_height, img_info.num_classes)
  if args.load_weights:
    model.load_weights(args.load_weights)
    print 'Loaded existing model weights from {}.'.format(args.load_weights)
  # Compile the model with SGD + momentum.
  # TODO: allow options to change the optimizer.
  print ('Compiling module...')
  start_time = time.time()
  sgd = SGD(lr=params['learning_rate'], decay=params['decay'],
            momentum=params['momentum'], nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)
  print 'Done in {}.'.format(get_elapsed_time(start_time))
  # Train the model.
  start_time = time.time()
  # TODO: implement data augmentation option.
  model.fit(img_loader.train_data, img_loader.train_labels,
            validation_data=(img_loader.test_data, img_loader.test_labels),
            batch_size=params['batch_size'], nb_epoch=params['num_epochs'],
            shuffle=True, show_accuracy=True, verbose=1)
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
  if args.test_mode:
    test_model(args, params)
  else:
    train_model(args, params)
