# Trains a deep CNN on a subset of ImageNet data.
#
# Run with GPU:
#   THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python model.py
#
# This example is from here:
#   https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
#
# CNN layer classes documentation:
#   http://keras.io/layers/convolutional/
#
#
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


from __future__ import print_function
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
from PIL import Image
import time


def read_images(im_names_file, img_dimensions, data, labels, classes,
                disp='all'):
  """Reads all images."""
  i = 0
  label = -1
  for line in im_names_file:
    impath = line.strip()
    if len(impath) == 0 or impath.startswith('#'):
      label += 1
      print ('Loading {} images for class "{}"...'.format(disp, classes[label]))
      continue
    img = Image.open(impath)
    img = img.resize(img_dimensions)
    img = img.convert('L')
    img_arr = np.asarray(img, dtype='float32')
    data[i, 0, :, :] = img_arr
    labels[i] = label
    i += 1

def load_data(img_dimensions):
  """Loads the image data, and returns it in the appropriate format."""
  # Read in the class labels:
  num_categories = 25
  classnames = open('classnames.txt', 'r')
  classes = map(str.strip, classnames.readlines())
  classnames.close()
  # Read the training data into memory:
  num_train = 2500
  train_im_names = open('trainImNames.txt', 'r')
  train_data = np.empty((num_train, 1, img_w, img_h), dtype='float32')
  train_labels = np.empty((num_train,), dtype='uint8')
  read_images(train_im_names, img_dimensions, train_data, train_labels, classes,
              disp='train')
  train_im_names.close()
  # Read the test data into memory:
  num_test = 500
  test_im_names = open('test1ImNames.txt', 'r')
  test_data = np.empty((num_test, 1, img_w, img_h), dtype='float32')
  test_labels = np.empty((num_test,), dtype='uint8')
  read_images(test_im_names, img_dimensions, test_data, test_labels, classes,
              disp='test')
  test_im_names.close()
  # Normalize the data to values between 0 and 1 and format the labels for
  # Keras, and return everything:
  train_data = train_data.astype('float32') / 255
  test_data = test_data.astype('float32') / 255
  train_labels = np_utils.to_categorical(train_labels, num_categories)
  test_labels = np_utils.to_categorical(test_labels, num_categories)
  return (train_data, train_labels), (test_data, test_labels)


# Hyperparameters:
batch_size = 32
num_epochs = 50 #200
data_augmentation = False
learning_rate = 0.01
decay = 1e-6
momentum = 0.9

# Set the data parameters and load the images (preprocessed appropriately):
img_w, img_h = 128, 128
img_channels = 1
num_classes = 25
start_time = time.time()
(train_data, train_labels), (test_data, test_labels) = load_data((img_w, img_h))
elapsed = round(time.time() - start_time, 2)
print ('Data successfully loaded in {} seconds.'.format(elapsed))


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
model.add(Dropout(0.25))

# Add another convolution layer:
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))

# And yet another:
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Add a fully-connected layer:
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Add a final softmax output layer:
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# Compile the model with SGD + momentum.
print ('Compiling module...')
start_time = time.time()
sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
elapsed = round(time.time() - start_time, 2)
print ('Done in {} seconds.'.format(elapsed))

# Train the model.
start_time = time.time()
if not data_augmentation:
  print ('Training without data augmentation.')
  model.fit(train_data, train_labels,
            validation_data=(test_data, test_labels), batch_size=batch_size,
            nb_epoch=num_epochs, shuffle=True,
            show_accuracy=True, verbose=1)
else:
  print ('Training with additional data augmentation.')
  # TODO: implement this!
elapsed = round(time.time() - start_time, 2)
print ('Finished training in {} seconds.'.format(elapsed))
