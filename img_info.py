# Handles processing image path files (which define where the data is stored)
# and their associated labels.


class ImageInfo(object):
  """Loads image file paths from input files."""

  _DEFAULT_IMG_DIMENSIONS = (256, 256, 1)

  def __init__(self, params, test_only=False):
    """Sets data variables and loads the file paths.

    Args:
      params: a ModelParams object containing the appropriate data file paths
          and data parameters.
      test_only: True if we should ignore the training data and not load it.
    """
    # Set the data values.
    self.num_classes = params['number_of_classes']
    # Initialize the data lists.
    self.img_dimensions = self._process_image_dimensions(
        params['img_dimensions'])
    self.classnames = self._load_image_classnames(params['classnames_file'])
    # Set up data structures for training and test images.
    self.train_img_files = {}
    self.test_img_files = {}
    for i in range(self.num_classes):
      self.train_img_files[i] = []
      self.test_img_files[i] = []
    self.num_train_images = 0
    self.num_test_images = 0
    # If not in test-only mode, load the path names for training images.
    if not test_only:
      self._load_train_image_paths(params['train_img_paths_file'])
    # Load the path names for test images.
    self._load_test_image_paths(params['test_img_paths_file'])
 
  @property
  def img_width(self):
    """Returns the width of the input images.

    Returns:
      The input image width.
    """
    return self.img_dimensions[0]
 
  @property
  def img_height(self):
    """Returns the height of the input images.

    Returns:
      The input image height.
    """
    return self.img_dimensions[1]

  @property
  def num_channels(self):
    """Returns the number of image channels the data is using.
    
    Returns:
      The number of image channels for the input data.
    """
    return self.img_dimensions[2]

  def _process_image_dimensions(self, dimensions):
    """Returns the training and testing image dimensions.
    
    All images fed into the neural network, for training and testing, will be
    formatted to match these dimensions.

    Args:
      dimensions: a tuple containing the following three values:
          width - a positive integer indicating the width of the images.
          height - a positive integer indicating the height of the images.
          num_channels - the number of channels to use. This number should be 1
              for grayscale training images or 3 to train on full RGB data.

    Returns:
      Valid image dimensions, as a (width, height, num_channels) tuple.
    """
    width, height, num_channels = dimensions
    # Make sure all the data is valid.
    if width <= 0:
      width = self._DEFAULT_IMG_DIMENSIONS[0]
    if height <= 0:
      height = self._DEFAULT_IMG_DIMENSIONS[1]
    if num_channels not in [1, 3]:
      num_channels = self._DEFAULT_IMG_DIMENSIONS[2]
    return (width, height, num_channels)

  def _load_image_classnames(self, fname):
    """Reads the classnames for the image data from the given file.
    
    Each class name should be on a separate line.

    Args:
      fname: the name of the file containing the list of class names.

    Returns:
      The list of class names.
    """
    classnames_file = open(fname, 'r')
    classnames = []
    for line in classnames_file:
      line = line.strip()
      if len(line) == 0 or line.startswith('#'):
        continue
      classnames.append(line)
    classnames_file.close()
    return classnames

  def _load_train_image_paths(self, fname):
    """Reads the image paths for the training data from the given file.
    
    Each file name (full directory path) should be on a separate line. The file
    paths must be ordered by their class label.

    Args:
      fname: the name of the file containing the training image paths.
    """
    self.num_train_images = self._read_file_data(fname, self.train_img_files)

  def _load_test_image_paths(self, fname):
    """Reads the image paths for the test data from the given file.
    
    Each file name (full directory path) should be on a separate line. The file
    paths must be ordered by their class label.

    Args:
      fname: the name of the file containing the test image paths.
    """
    self.num_test_images = self._read_file_data(fname, self.test_img_files)

  def _read_file_data(self, fname, destination):
    """Reads the data of the given file into the destination list.
    
    The file will be read line by line, and each line that doesn't start with a
    "#" or is not empty will be stored in the given list.

    Args:
      fname: the name of the file to read.
      destination: a dictionary into which the file's data will be put line by line.

    Returns:
      The number of image paths that were provided for this data batch.
    """
    paths_file = open(fname, 'r')
    num_images = 0
    for line in paths_file:
      line = line.strip()
      if len(line) == 0 or line.startswith('#'):
        continue
      imdata = line.split()
      impath, classnum = imdata[0], imdata[1]
      classnum = int(classnum)
      destination[classnum].append(impath)
      num_images += 1
    paths_file.close()
    return num_images
