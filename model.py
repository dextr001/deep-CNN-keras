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
from img_loader import ImageInfo, ImageLoader
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import time


# Hyperparameters:
batch_size = 32
num_epochs = 50 #200
data_augmentation = False
learning_rate = 0.01
decay = 1e-6
momentum = 0.9

# Set the data parameters and load the images (preprocessed appropriately):
img_info = ImageInfo(25, 100, 20)
img_info.set_image_dimensions(128, 128, 1)
img_info.load_image_classnames('classnames.txt')
img_info.load_image_classnames('classnames.txt')
img_info.load_image_classnames('classnames.txt')

start_time = time.time()
(train_data, train_labels), (test_data, test_labels) = load_data((img_w, img_h))
elapsed = round(time.time() - start_time, 2)
print ('Data successfully loaded in {} seconds.'.format(elapsed))

#
## Build the CNN model.
#model = Sequential()
#
## Add a convolution layer:
#model.add(Convolution2D(32, 3, 3, border_mode='same',
#                        input_shape=(img_channels, img_w, img_h)))
#model.add(Activation('relu'))
#
## And another one:
#model.add(Convolution2D(32, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#
## Add another convolution layer:
#model.add(Convolution2D(64, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#
## And yet another:
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#
## Add a fully-connected layer:
#model.add(Flatten())
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#
## Add a final softmax output layer:
#model.add(Dense(num_classes))
#model.add(Activation('softmax'))
#
#
## Compile the model with SGD + momentum.
#print ('Compiling module...')
#start_time = time.time()
#sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
#elapsed = round(time.time() - start_time, 2)
#print ('Done in {} seconds.'.format(elapsed))
#
## Train the model.
#start_time = time.time()
#if not data_augmentation:
#  print ('Training without data augmentation.')
#  model.fit(train_data, train_labels,
#            validation_data=(test_data, test_labels), batch_size=batch_size,
#            nb_epoch=num_epochs, shuffle=True,
#            show_accuracy=True, verbose=1)
#else:
#  print ('Training with additional data augmentation.')
#  # TODO: implement this!
#elapsed = round(time.time() - start_time, 2)
#print ('Finished training in {} seconds.'.format(elapsed))
