# Deep Convolutional Neural Networks with Keras

This code helps quickly set up and run a deep CNN project for image classification.

Getting Started
-----

Before you can run this code, you need to have all of the following Python modules installed:

<ul>
  <li> Keras </li>
  <li> Theano </li>
  <li> Pillow </li>
  <li> h5py </li>
</ul>

Here is how to do this with Anaconda 2 (for Python 2):

<ol>
  <li> Download Anaconda 2 here: https://www.continuum.io/downloads#_unix </li>
  <li> Run the installer bash script: <code>$ bash Anaconda2-2.5.0-Linux-x86_64.sh</code> </li>
  <li> Reload bashrc: <code>$ . ~/.bashrc</code> </li>
  <li> Create a new anaconda environment: <code>$ conda create -n py27x python=2.7</code>. "py27x" is the name of the environment. You can name it whatever you want. </li>
  <li> <code>$ source activate py27x</code>. </li>
  <li> <code>$ pip install --upgrade pip</code> </li>
  <li> <code>$ pip install keras</code> </li>
  <li> <code>$ pip install git+git://github.com/Theano/Theano.git</code> </li>
  <li> <code>$ pip install Pillow</code> </li>
  <li> <code>$ pip install h5py</code>. This will get an error along the way, but don't worry - it fixes itself. </li>
</ol>

Whenever you want to use this environment (which includes all of the things you just installed), simply run <code>$ source activate py27x</code>. Then run <code>source deactivate</code> when you're finished.

If you want to use the GPU (which is pretty much necessary if you want to train a deep CNN model), you need to install CUDA. You can download it here: https://developer.nvidia.com/cuda-downloads. If you download the runfile version, you can install it by running the provided bash script. If the appropriate Nvidea drivers are already installed, this will give you the option to install CUDA in a directory of your choosing (without root access).


Using the Code
-----

This code must be run in a Python 2 environment with all of the required modules listed above installed.

The <code>run.py</code> script is what allows you to train or test a module.

TODO


Preparing Your Data
-----

The data itself can consist of any RGB or Grayscale images saved on your computer. You need to specify three files that will tell the code where your data is and how it is organized:

<ul>
  <li> A file containing a list of class names. Each line of this file should be a string indicating a classname (e.g. "cat", "dog", "building", etc. </li>
  <li> A file containing a list of training images. Each line of this file should specify the <i>full path</i> of an image. The first <i>N</i> lines of the file are for images of the first class, the next <i>N</i> lines are for the next class, and so on. The order of classes should be the same as in the class names file. <b>At this time, you must have the same number of training images for each class.</b> This is still a TODO. </li>
  <li> A file containing a list of test images. As for the training images, each line must specify the full path of an image. The file is ordered by class, and each class must have the same number of test images. </li>
</ul>

In all of the above files, empty lines will be ignored. Also, lines starting with a "#" will be ignored (so you can add comments).

<b><u>Example Files</u></b>:

~~~~~
# classnames.txt
cat
dog
~~~~~

~~~~~
# train_images.txt
# 5 cat training images
/home/users/You/data/cats/img1.jpg
/home/users/You/data/cats/img2.jpg
/home/users/You/data/cats/img3.jpg
/home/users/You/data/cats/img4.jpg
/home/users/You/data/cats/img5.jpg
# 5 dog training images
/home/users/You/data/dogs/img1.jpg
/home/users/You/data/dogs/img2.jpg
/home/users/You/data/dogs/img3.jpg
/home/users/You/data/dogs/img4.jpg
/home/users/You/data/dogs/img5.jpg
~~~~~

~~~~~
# test_images.txt
# 2 cat test images
/home/users/You/data/cats/img6.jpg
/home/users/You/data/cats/img7.jpg
# 2 dog test images
/home/users/You/data/dogs/img6.jpg
/home/users/You/data/dogs/img7.jpg
~~~~~


Setting Up the Config File
-----

Edit the <code>params.config</code> file (or write your own) with your specific data information. This file is formatted as <code>key: value</code> pairs on each line. All of the values must be defined, otherwise the configuration file will not be accepted. The values must be explicity defined as valid Python 2 types (e.g. a string must be in quotes, a tuple must be of the form <code>(val1, val2, val3)</code>, etc. Here is what you need to add:

<ol>
  <li> Point to the data files you defined above. Specifically, set up the appropriate <code>classnames_file</code>, <code>train_img_paths_file</code>, and <code>test_img_paths_file</code>. Note that these file paths must be in quotes (i.e. they will be interpreted as Python strings). </li>
  <li> Set the data parameters: <code>number_of_classes</code>, <code>train_imgs_per_class</code>, and <code>test_imgs_per_class</code>. Following the example files above, these values would be 2, 5, and 2, respectively. </li>
  <li> Set the image dimensions (<code>img_dimensions</code>). All images will be resized to these dimensions (width, height, number of channels) before being fed into the CNN. <b>Currently, only 1 channel is supported (grayscale).</b> This is a TODO. </li>
  <li> Optionally, set the training hyperparameters as desired (<code>batch_size</code>, <code>num_epochs</code>, <code>learning_rate</code>, <code>decay</code>, and <code>momentum</code>). </li>
</ol>


Defining, Saving, and Loading a Module
-----

TODO
