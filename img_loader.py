# Provides a loader that reads and stores the training and test data in memory
# to be fed into a neural network.

from keras.utils import np_utils
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize


class ImageLoader(object):
  """Loads image data from provided image paths."""

  def __init__(self, image_info):
    """Reads in all of the images as defined in the given ImageInfo object.

    Args:
      image_info: an ImageInfo object that contains all of the image paths
          and data size values. These images will be loaded into memory and
          used for training and testing.
    """
    # Data size information:
    self._image_info = image_info
    num_classes = self._image_info.num_classes
    img_w = self._image_info.img_width
    img_h = self._image_info.img_height
    num_channels = self._image_info.num_channels
    # Initialize the empty train data arrays:
    num_train_imgs = self._image_info.num_train_images
    self.train_data = np.empty((num_train_imgs, num_channels, img_w, img_h),
                               dtype='float32')
    self.train_labels = np.empty((num_train_imgs,), dtype='uint8')
    # Initialize the empty test data arrays:
    num_test_imgs = self._image_info.num_test_images
    self.test_data = np.empty((num_test_imgs, num_channels, img_w, img_h),
                              dtype='float32')
    self.test_labels = np.empty((num_test_imgs,), dtype='uint8')

  def load_all_images(self):
    """Loads all images (training and test) into memory.

    All images are loaded based on the paths provided in the ImageInfo object.
    The image data is stored in the train_data/train_labels and test_data/
    test_labels numpy arrays, and formatted appropriately for the classifier.
    """
    self._load_train_images()
    self.load_test_images()

  def _load_train_images(self):
    """Loads all training images into memory and normalizes the data."""
    self._load_images(
        self._image_info.train_img_files,
        self._image_info.num_classes,
        self.train_data, self.train_labels, disp='train')
    self.train_data = self.train_data.astype('float32') / 255
    self.train_labels = np_utils.to_categorical(self.train_labels,
                                                self._image_info.num_classes)

  def load_test_images(self):
    """Loads all test images into memory and normalizes the data."""
    self._load_images(
        self._image_info.test_img_files,
        self._image_info.num_classes,
        self.test_data, self.test_labels, disp='test')
    self.test_data = self.test_data.astype('float32') / 255
    self.test_labels = np_utils.to_categorical(self.test_labels,
                                               self._image_info.num_classes)

  def _load_images(self, file_names, num_classes, data, labels,
                   disp='all'):
    """Loads the images from the given file names to the given arrays.
    
    No data normalization happens at this step.

    Args:
      file_names: a dictionary that maps each label ID to a list of file paths
          that points to the locations of the images on disk.
      num_classes: the number of images classes. The labels will be assigned
          between 0 and num_classes - 1.
      data: the pre-allocated numpy array into which the image data will be
          inserted.
      labels: the pre-allocated numpy array into which the image labels will be
          inserted.
      disp: a string (e.g. 'test' or 'train') to show what type of data is being
          loaded in the prints.
    """
    image_index = 0
    for label_id in range(num_classes):
      print 'Loading {} images for class "{}" ({})...'.format(
          disp, self._image_info.classnames[label_id], label_id)
      for impath in file_names[label_id]:
        img = Image.open(impath)
        img = img.resize((self._image_info.img_width,
                          self._image_info.img_height))
        if self._image_info.num_channels != 3:
          img = img.convert('L')
        img_arr = np.asarray(img, dtype='float32')
        if self._image_info.num_channels == 3:
          # If the raw image is grayscale but the classification is on RGB data,
          # replicate the grayscale intensities across the three channels.
          if img_arr.ndim == 2:
            data[image_index, 0, :, :] = img_arr
            data[image_index, 1, :, :] = img_arr
            data[image_index, 2, :, :] = img_arr
          # Otherwise, extract each channel and add it to the data array.
          else:
            data[image_index, 0, :, :] = img_arr[:, :, 0]
            data[image_index, 1, :, :] = img_arr[:, :, 1]
            data[image_index, 2, :, :] = img_arr[:, :, 2]
        else:
          data[image_index, 0, :, :] = img_arr
        labels[image_index] = label_id
        image_index += 1

  def assign_soft_labels(self, affinity_matrix):
    """Assigns soft labels, replacing the 1-hot labels for all training images.

    Each image will be assigned the label vector for its class.

    Args:
      affinity_matrix: an N by N nparray matrix where N is the number of classes
          in this data.
    """
    # Normalize in case it was not normalized already.
    affinity_matrix = normalize(affinity_matrix, axis=1)
    num_images = self._image_info.num_train_images
    for i in range(num_images):
      label_id = np.where(self.train_labels[i] == 1)[0][0]
      self.train_labels[i, :] = affinity_matrix[label_id, :]
