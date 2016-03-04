# The build_model() function returns the model as defined by the code in this
# file. Modify the architecture as needed.

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD


def compile_model(model, params):
  """Compiles the model with the defined optimizer.

  Update the code as needed to change the optimizer.

  Args:
    model: the Keras model to be compiled.
    params: a ModelParams object that specifies the hyperparameters for the
        optimizer.
  """
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
