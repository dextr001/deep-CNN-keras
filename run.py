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

  batch_size = 32
  num_epochs = 1 #50 #200
  data_augmentation = False
  learning_rate = 0.01
  decay = 1e-6
  momentum = 0.9


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


def train_model(params, save_weights_file=None, load_weights_file=None):
  """Runs the code to set up data and train a model.
  
  Args:
    params: the ModelParams object contianing the model training parameters.
    save_weights_file: provide a file name where the trained model will be saved
        after training is finished. This model can be loaded for further
        training and testing later.
    load_weights_file: provide a file name only if there is an existing Keras
        model to load and train. Otherwise, a new model will be created to fit
        the training data. If a model is provided, it must already fit the
        data parameters.
  """
  # Set the data parameters and image source paths.
  # TODO: use more clearly-defined image path files.
  # TODO: all of these numbers should be parameters.
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
  if load_weights_file:
    model.load_weights(load_weights_file)
    print 'Loaded existing model weights from {}.'.format(load_weights_file)
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
  if save_weights_file:
    model.save_weights(save_weights_file)
    print 'Saved trained model weights to {}.'.format(save_weights_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run a deep neural network model using Keras.')
  parser.add_argument('-save-weights', dest='save_weights', required=False,
                      help='Save the trained weights to this file.')
  parser.add_argument('-load-weights', dest='load_weights', required=False,
                      help='Load existing weights from this file.')
  args = parser.parse_args()
  params = ModelParams()
  train_model(params, save_weights_file=args.save_weights,
              load_weights_file=args.load_weights)
