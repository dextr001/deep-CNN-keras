# Loads the image stuff.


class ImageData(object):
  """Loads image file paths from input files."""

  self._DEFAULT_IMG_DIMENSIONS = (256, 256, 1)

  def __init__(self)
    """Sets up the initial data variables."""
    # Initialize the data lists.
    self.img_dimensions = self._DEFAULT_IMG_DIMENSIONS
    self.classnames = []
    self.train_img_files = []
    self.test_img_files = []

  def set_image_dimensions(self, width, height, num_channels):
    """Set the training and testing image dimensions.
    
    All images fed into the neural network, for training and testing, will be
    formatted to match these dimensions.

    Args:
      width: a positive integer indicating the width of the images.
      height: a positive integer indicating the height of the images.
      num_channels: the number of channels to use. This number should be 1
          for grayscale training images or 3 to train on full RGB data.
    """
    # Make sure all the data is valid.
    if width <= 0:
      width = self._DEFAULT_IMG_DIMENSIONS[0]
    if height <= 0:
      height = self._DEFAULT_IMG_DIMENSIONS[1]
    if num_channels not in [1, 3]:
      num_channels = self._DEFAULT_IMG_DIMENSIONS[2]
    # Set the dimensions.
    self.img_dimensions = (width, height, num_channels)

  def load_image_classnames(self, fname):
    """Reads the classnames for the image data from the given file.
    
    Each class name should be on a separate line.

    Args:
      fname: the name of the file containing the list of class names.
    """
    self._read_file_data(fname, self.classnames)

  def load_train_image_paths(self, fname):
    """Reads the image paths for the training data from the given file.
    
    Each file name (full directory path) should be on a separate line. The file
    paths must be ordered by their class label.

    Args:
      fname: the name of the file containing the training image paths.
    """
    self._read_file_data(fname, self.train_img_files)

  def load_test_image_paths(self, fname):
    """Reads the image paths for the test data from the given file.
    
    Each file name (full directory path) should be on a separate line. The file
    paths must be ordered by their class label.

    Args:
      fname: the name of the file containing the test image paths.
    """
    self._read_file_data(fname, self.test_img_files)

  def _read_file_data(self, fname, destination):
    """Reads the data of the given file into the destination list.
    
    The file will be read line by line, and each line that doesn't start with a
    "#" or is not empty will be stored in the given list.

    Args:
      fname: the name of the file to read.
      destination: a list into which the file's data will be put line by line.
    """
    paths_file = open(fname, 'r')
    for line in paths_file:
      impath = line.strip()
      if len(impath) == 0 or impath.startswith('#'):
        continue
      destination.append(impath)
    paths_file.close()


#class ImageLoader(object):
#  """Reads images"""
#
#    
#  def __init__(self, num_classes, num_train_imgs_per_class,
#               num_test_imgs_per_class):
#    """
#    Args:
#      num_classes: the number of data classes in the data set.
#      num_train_imgs_per_class: the number of training images provided for each
#          class. The number of images must be the same for each class. For
#          example, if this number is 100, then for each of the 'num_classes'
#          classes, exactly 100 training images must be provided.
#      num_test_imgs_per_class: similarly, the number of test images per class.
#    """
#    self._num_classes = num_classes
#    self._num_train_imgs_per_class = num_train_img_per_class
#    self._num_test_imgs_per_class = num_test_img_per_class
#  def read_images(self, im_names_file, img_dimensions, data, labels, classes,
#                  disp='all'):
#    """Reads all images."""
#    i = 0
#    label = -1
#    for line in im_names_file:
#      impath = line.strip()
#      if len(impath) == 0 or impath.startswith('#'):
#        label += 1
#        print ('Loading {} images for class "{}"...'.format(disp, classes[label]))
#        continue
#      img = Image.open(impath)
#      img = img.resize(img_dimensions)
#      img = img.convert('L')
#      img_arr = np.asarray(img, dtype='float32')
#      data[i, 0, :, :] = img_arr
#      labels[i] = label
#      i += 1
#
#  def load_data(self, img_dimensions):
#    """Loads the image data, and returns it in the appropriate format."""
#    # Read in the class labels:
#    num_categories = 25
#    classnames = open('classnames.txt', 'r')
#    classes = map(str.strip, classnames.readlines())
#    classnames.close()
#    # Read the training data into memory:
#    num_train = 2500
#    train_im_names = open('trainImNames.txt', 'r')
#    train_data = np.empty((num_train, 1, img_w, img_h), dtype='float32')
#    train_labels = np.empty((num_train,), dtype='uint8')
#    read_images(train_im_names, img_dimensions, train_data, train_labels, classes,
#                disp='train')
#    train_im_names.close()
#    # Read the test data into memory:
#    num_test = 500
#    test_im_names = open('test1ImNames.txt', 'r')
#    test_data = np.empty((num_test, 1, img_w, img_h), dtype='float32')
#    test_labels = np.empty((num_test,), dtype='uint8')
#    read_images(test_im_names, img_dimensions, test_data, test_labels, classes,
#                disp='test')
#    test_im_names.close()
#    # Normalize the data to values between 0 and 1 and format the labels for
#    # Keras, and return everything:
#    train_data = train_data.astype('float32') / 255
#    test_data = test_data.astype('float32') / 255
#    train_labels = np_utils.to_categorical(train_labels, num_categories)
#    test_labels = np_utils.to_categorical(test_labels, num_categories)
#    return (train_data, train_labels), (test_data, test_labels)
