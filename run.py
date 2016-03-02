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

from img_loader import ImageInfo, ImageLoader
from keras.optimizers import SGD
from model import build_model
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


# Hyperparameters:
batch_size = 32
num_epochs = 50 #200
data_augmentation = False
learning_rate = 0.01
decay = 1e-6
momentum = 0.9

# Set the data parameters and image source paths.
# TODO: use more clearly-defined image path files.
img_info = ImageInfo(25, 1160, 20)
img_info.set_image_dimensions(128, 128, 1)  # img width, height, channels
img_info.load_image_classnames('test/classnames.txt')
#img_info.load_train_image_paths('test/trainImNames.txt')
img_info.load_train_image_paths('test/extraTrainImNames.txt')
img_info.load_test_image_paths('test/test1ImNames.txt')

# Load the images into memory and preprocess appropriately.
start_time = time.time()
img_loader = ImageLoader(img_info)
img_loader.load_all_images()
print 'Data successfully loaded in {}.'.format(get_elapsed_time(start_time))

# Get the deep CNN model for the given data.
model = build_model(img_info.num_channels, img_info.img_width,
                    img_info.img_height, img_info.num_classes)

# Compile the model with SGD + momentum.
print ('Compiling module...')
start_time = time.time()
sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
print 'Done in {}.'.format(get_elapsed_time(start_time))

# Train the model.
start_time = time.time()
if not data_augmentation:
  print ('Training without data augmentation.')
  model.fit(img_loader.train_data, img_loader.train_labels,
            validation_data=(img_loader.test_data, img_loader.test_labels),
            batch_size=batch_size, nb_epoch=num_epochs, shuffle=True,
            show_accuracy=True, verbose=1)
else:
  print ('Training with additional data augmentation.')
  # TODO: implement this!
print 'Finished training in {}.'.format(get_elapsed_time(start_time))
