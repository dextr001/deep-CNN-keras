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
from elapsed_timer import ElapsedTimer
from img_info import ImageInfo
from img_loader import ImageLoader
from keras.models import model_from_json
from model import build_model, compile_model
from model_params import ModelParams
import numpy as np


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
  img_info = ImageInfo(params['number_of_classes'], args.explicit_labels)
  img_info.set_image_dimensions(params['img_dimensions'])
  img_info.load_image_classnames(params['classnames_file'])
  img_info.load_test_image_paths(params['test_img_paths_file'])
  # Load the model and its weights and compile it.
  model = get_model(args, img_info)
  print ('Compiling module...')
  timer = ElapsedTimer()
  compile_model(model, params)
  print 'Done in {}.'.format(timer.get_elapsed_time())
  # Load the test images into memory and preprocess appropriately.
  timer.reset()
  img_loader = ImageLoader(img_info)
  img_loader.load_test_images()
  print 'Test data successfully loaded in {}.'.format(timer.get_elapsed_time())
  # Run the evaluation on the test data.
  timer.reset()
  predictions = model.predict_classes(img_loader.test_data,
                                      batch_size=params['batch_size'])
  print 'Finished testing in {}.'.format(timer.get_elapsed_time())
  # Compute the percentage of correct classifications.
  num_predicted = len(predictions)
  num_correct = 0
  num_classes = params['number_of_classes']
  confusion_matrix = np.zeros((num_classes, num_classes))
  misclassified = []
  # Convert the test image dictionary to a flat list.
  test_img_files = []
  for i in range(num_classes):
    for img_file in img_info.test_img_files[i]:
      test_img_files.append(img_file)
  # Compute confusion matrix and find incorrect classifications.
  for i in range(num_predicted):
    predicted_class = predictions[i]
    correct = np.nonzero(img_loader.test_labels[i])
    correct = correct[0][0]
    confusion_matrix[correct][predicted_class] += 1
    if predicted_class == correct:
      num_correct += 1
    else:
      # Save the image file name, its correct class, and its predicted class.
      misclassified.append((test_img_files[i], correct, predicted_class))
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
    f = open(args.confusion_matrix, 'w')
    f.write(output)
    f.close()
    print 'Saved confusion matrix to {}.'.format(args.confusion_matrix)
  if args.report_misclassified:
    f = open(args.report_misclassified, 'w')
    for example in misclassified:
      img_path, img_class, predicted_class = example
      f.write('{} {} {}\n'.format(img_path, img_class, predicted_class))
    f.close()
    print 'Saved misclassified images report to {}.'.format(
        args.report_misclassified)
  if args.report_scores:
    print 'Computing instance scores...'
    scores = model.predict_proba(img_loader.test_data,
                                 batch_size=params['batch_size'],
                                 verbose=0)
    f = open(args.report_scores, 'w')
    score_template = ' '.join(['{}'] * num_classes)
    for i in range(len(scores)):
      score_list = [round(score, 5) for score in scores[i]]
      score_string = score_template.format(*score_list)
      f.write('{} {}\n'.format(test_img_files[i], score_string))
    f.close()
    print 'Saved scores report to {}.'.format(args.report_scores)


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
  img_info = ImageInfo(params['number_of_classes'], args.explicit_labels)
  img_info.set_image_dimensions(params['img_dimensions'])
  img_info.load_image_classnames(params['classnames_file'])
  img_info.load_train_image_paths(params['train_img_paths_file'])
  img_info.load_test_image_paths(params['test_img_paths_file'])
  # Load the model and (possibly) its weights.
  model = get_model(args, img_info)
  # Save the model if that option was specified.
  if args.save_model:
    f = open(args.save_model, 'w')
    f.write(model.to_json())
    f.close()
    print 'Saved model architecture to {}.'.format(args.save_model)
  # Compile the model.
  print ('Compiling module...')
  timer = ElapsedTimer()
  compile_model(model, params)
  print 'Done in {}.'.format(timer.get_elapsed_time())
  # Load the images into memory and preprocess appropriately.
  timer.reset()
  img_loader = ImageLoader(img_info)
  img_loader.load_all_images()
  print 'Data successfully loaded in {}.'.format(timer.get_elapsed_time())
  # Train the model.
  timer.reset()
  # TODO: implement data augmentation option.
  model.fit(img_loader.train_data, img_loader.train_labels,
            validation_data=(img_loader.test_data, img_loader.test_labels),
            batch_size=params['batch_size'], nb_epoch=params['num_epochs'],
            shuffle=True, show_accuracy=True, verbose=1)
  print 'Finished training in {}.'.format(timer.get_elapsed_time())
  # Save the weights if that option was specified.
  if args.save_weights:
    model.save_weights(args.save_weights)
    print 'Saved trained model weights to {}.'.format(args.save_weights)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run a deep neural network model using Keras.')
  parser.add_argument('params_file',
                      help='The file containing data paths and model params.')
  parser.add_argument('--explicit-labels', dest='explicit_labels',
                      action='store_true',
                      help=('Use explicit label values for the data. The ' +
                            'data must be formatted appropriately.'))
  parser.add_argument('--test', dest='test_mode', action='store_true',
                      help='Test the model with weights (-load-weights).')
  parser.add_argument('-confusion-matrix', dest='confusion_matrix',
                      required=False,
                      help=('Saves a confusion matrix to the given file ' +
                            '(test mode only).'))
  parser.add_argument('-report-misclassified', dest='report_misclassified',
                      required=False,
                      help=('Saves a list of misclassified images to the ' +
                            'given file (test mode only).'))
  parser.add_argument('-report-scores', dest='report_scores', required=False,
                      help=('Saves a list of all classification scores to ' +
                            'the given file (test mode only).'))
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
