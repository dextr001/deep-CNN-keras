from img_loader import ImageInfo, ImageLoader
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
img_info.load_image_classnames('test/classnames.txt')
img_info.load_train_image_paths('test/trainImNames.txt')
img_info.load_test_image_paths('test/test1ImNames.txt')

start_time = time.time()
img_loader = ImageLoader(img_info)
img_loader.load_all_images()
elapsed = time.time() - start_time
print ('Data successfully loaded in {} seconds.'.format(elapsed))
