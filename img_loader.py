# Loads the image stuff.


class ImageData(object):
  """Defines an image data object."""

  def __init__(self):
    """?"""
    self.classnames = []
    self.train_img_files = []
    self.test_img_files = []
    self.img_dimensions = (256, 256, 3)

  def set_image_dimensions(self, width, height, num_channels):
    """?"""
    self.img_dimensions = (width, height, num_channels)

  def load_image_classnames(self, fname):
    """?"""
    self._read_file_data(fname, self.classnames)

  def load_train_image_paths(self, fname):
    """?"""
    self._read_file_data(fname, self.train_img_files)

  def load_test_image_paths(self, fname):
    """?"""
    self._read_file_data(fname, self.test_img_files)

  def _read_file_data(self, fname, destination):
    """?"""
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
