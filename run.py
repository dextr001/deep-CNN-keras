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
from keras.models import model_from_json
from model import build_model, compile_model
from model_params import ModelParams
import numpy as np
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


def get_model(args, img_info):
  """Build the deep CNN model for the given data or load it from file.

  The model's weights are also loaded from a file if that option is specified.
  
  Args:
    args: the argparse arguments that specify whether the model should be newly
        built or loaded from a file.
    img_info: an ImageInfo object that contains information about the data.
        If the model is built from scratch, these values will be used to set the
        model's input layer dimensions.
  """
  model = None
  # Load the model from a file or build it.
  if args.load_model:
    f = open(args.load_model, 'r')
    model = model_from_json(f.read())
    f.close()
    print 'Loaded existing model from {}.'.format(args.load_model)
  else:
    model = build_model(img_info.num_channels, img_info.img_width,
                        img_info.img_height, img_info.num_classes)
  # If weights are provided, load them from a file here.
  if args.load_weights:
    model.load_weights(args.load_weights)
    print 'Loaded existing model weights from {}.'.format(args.load_weights)
  return model


def test_model(args, params):
  """Tests a model on the test data set.

  Prints out the final accuracy of the predictions on the test data. Also prints
  a normalized confusion matrix if that argument is specified by the user.
  
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
  img_info.set_image_dimensions(params['img_dimensions'])
  img_info.load_image_classnames(params['classnames_file'])
  img_info.load_test_image_paths(params['test_img_paths_file'])
  # Load the images into memory and preprocess appropriately.
  start_time = time.time()
  img_loader = ImageLoader(img_info)
  img_loader.load_test_images()
  print 'Test data successfully loaded in {}.'.format(
      get_elapsed_time(start_time))
  # Load the model and its weights and compile it.
  model = get_model(args, img_info)
  print ('Compiling module...')
  start_time = time.time()
  compile_model(model, params)
  print 'Done in {}.'.format(get_elapsed_time(start_time))
  # Run the evaluation on the test data.
  start_time = time.time()
  predictions = model.predict_classes(img_loader.test_data,
                                      batch_size=params['batch_size'])
  scores = model.predict(img_loader.test_data, batch_size=params['batch_size'])
  print 'Finished testing in {}.'.format(get_elapsed_time(start_time))
  # Compute the percentage of correct classifications.
  num_predicted = len(predictions)
  num_correct = 0
  confusion_matrix = np.zeros((25, 25))
  for i in range(num_predicted):
    predicted_class = predictions[i]
    correct = np.nonzero(img_loader.test_labels[i])
    correct = correct[0][0]
    confusion_matrix[correct][predicted_class] += 1
    if predicted_class == correct:
      num_correct += 1
  accuracy = round(float(num_correct) / float(num_predicted), 4)
  print 'Predicted classes for {} images with accuracy = {}'.format(
      num_predicted, accuracy)
  if args.confusion_matrix:
    # Normalize and print the matrix.
    per_row_max = confusion_matrix.sum(axis = 1)
    confusion_matrix = confusion_matrix.transpose() / per_row_max
    confusion_matrix = confusion_matrix.transpose()
    output = ''
    for row in confusion_matrix:
      row_list = list(row)
      output += ' '.join(map(str, row_list)) + '\n'
    fname = raw_input(
        'Enter file name for confusion matrix (leave blank to cancel): ')
    if fname:
      f = open(fname, 'w')
      f.write(output)
      f.close()
    print confusion_matrix


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
  # Load the model and (possibly) its weights.
  model = get_model(args, img_info)
  # Save the model if that option was specified.
  if args.save_model:
    f = open(args.save_model, 'w')
    f.write(model.to_json())
    f.close()
    print 'Saved model architecture to {}.'.format(args.save_model)
  # Compile the model.
  # TODO: allow options to change the optimizer.
  print ('Compiling module...')
  start_time = time.time()
  compile_model(model, params)
  print 'Done in {}.'.format(get_elapsed_time(start_time))
  # Train the model.
  start_time = time.time()
  # TODO: implement data augmentation option.
  model.fit(img_loader.train_data, img_loader.train_labels,
            validation_data=(img_loader.test_data, img_loader.test_labels),
            batch_size=params['batch_size'], nb_epoch=params['num_epochs'],
            shuffle=True, show_accuracy=True, verbose=1)
  print 'Finished training in {}.'.format(get_elapsed_time(start_time))
  # Save the weights if that option was specified.
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
  parser.add_argument('--confusion-matrix', dest='confusion_matrix',
                      action='store_true',
                      help='Prints a confusion matrix (test mode only).')
  parser.add_argument('-load-model', dest='load_model', required=False,
                      help='Load an existing model from this file.')
  parser.add_argument('-save-model', dest='save_model', required=False,
                      help='Save the model architecture to this file.')
  parser.add_argument('-load-weights', dest='load_weights', required=False,
                      help='Load existing weights from this file.')
  parser.add_argument('-save-weights', dest='save_weights', required=False,
                      help='Save the trained weights to this file.')
  args = parser.parse_args()
  params = ModelParams()
  if not params.read_config_file(args.params_file):
    print 'Missing configuration values. Cannot continue.'
    exit(0)
  if args.test_mode:
    test_model(args, params)
  else:
    train_model(args, params)
