# The build_model() function returns the model as defined by the code in this
# file. Modify the architecture as needed.

import argparse
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from model_params import ModelParams


#################################### NOTES #####################################
# Convolution2D:
#     nb_filter,    # number of filters
#     nb_row,       # number of rows in each kernel
#     nb_col,       # number of columns in each kernel
#     border_mode,  # ['valid'] - what to do at the bordering pixels
#                   #     'valid':
#                   #     'same':
#     input_shape   # tuple indicating the image dimensions (input layer only)
#                   #     e.g.: (3, 128, 128) for 128 x 128 RGB images
################################################################################
# Activation:
#     ReLU = rectified linear unit... value itself if positive, 0 if negative.
# MaxPooling:
#     Downsamples the output of each filter. E.g. if pool_size=(2, 2), then the
#     output images are downsampled such that only the maximum value in every
#     2x2 window of the output is chosen.
# Dropout:
#     Zeros out activations in the output. This prevents co-adaptation of units
#     and thus prevents overfitting. The value is the fraction of units that get
#     dropped out.
# Flatten:
#     Flattens the input
################################################################################


def compile_model(model, params):
  """Compiles the model with the defined optimizer.

  Update the code as needed to change the optimizer.

  Args:
    model: the Keras model to be compiled.
    params: a ModelParams object that specifies the hyperparameters for the
        optimizer.
  """
  # TODO: allow options to change the optimizer.
  sgd = SGD(lr=params['learning_rate'], decay=params['decay'],
            momentum=params['momentum'], nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)


def build_model(img_channels, img_w, img_h, num_classes):
  """Builds and returns a learning model.

  Update the code as needed. TODO: possibly parse this from a text file.

  Args:
    img_channels: the number of channels in the input images (1 for grayscale,
        or 3 for RGB).
    img_w: the width (in pixels) of the input images.
    img_h: the height of the input images.
    num_classes: the number of classes that the data will have - this dictates
        the number of values produced in the output layer.

  Returns:
    A deep neural network model.
  """
  # Build the CNN model.
  model = Sequential()
  
  # Add a convolution layer:
  model.add(Convolution2D(32, 3, 3, border_mode='same',
                          input_shape=(img_channels, img_w, img_h)))
  model.add(Activation('relu'))
  
  # And another one:
  model.add(Convolution2D(32, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))
  
  # Add another convolution layer:
  model.add(Convolution2D(64, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  
  # And yet another:
  model.add(Convolution2D(64, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))
  
  # Add a fully-connected layer:
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  
  # Add a final softmax output layer:
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  return model


# If this file is run, build the model and save it to the specified file
if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Saves the model architecture to the specified file.')
  # The params_file is required because the data size must be accounted for when
  # building the model.
  parser.add_argument('params_file',
                      help='The file containing data paths and model params.')
  parser.add_argument('file_name',
                      help='The file to which the model will be saved.')
  args = parser.parse_args()
  # Read the config file.
  params = ModelParams()
  if not params.read_config_file(args.params_file):
    print 'Missing configuration values. Cannot continue.'
    exit(0)
  # Build and save the model.
#def build_model(img_channels, img_w, img_h, num_classes):
  img_dimensions = params['img_dimensions']
  model = build_model(img_dimensions[2], img_dimensions[0], img_dimensions[1],
                      params['number_of_classes'])
  f = open(args.file_name, 'w')
  f.write(model.to_json())
  f.close()
  print 'Model successfully saved to {}.'.format(args.file_name)
