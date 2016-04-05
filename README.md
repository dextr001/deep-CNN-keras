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
  <li> sklearn </li>
</ul>

Here is how to do this with Anaconda 2 (for Python 2):

<ol>
  <li> Download Anaconda 2 here: https://www.continuum.io/downloads#_unix </li>
  <li> Run the installer bash script: <code>$ bash Anaconda2-2.5.0-Linux-x86_64.sh</code> </li>
  <li> Reload bashrc: <code>$ . ~/.bashrc</code> </li>
  <li> Create a new anaconda environment: <code>$ conda create -n py27x python=2.7</code>. "py27x" is the name of the environment. You can name it whatever you want. </li>
  <li> <code>$ source activate py27x</code>. </li>
  <li> <code>$ pip install --upgrade pip</code> </li>
  <li> <code>$ pip install keras</code>
    <ul>
      <li> If this fails due to an error installing scipy, run <code>$ conda install scipy</code> first. </li>
      <li> This is a general fix for any packages it fails to install with pip. </li>
    </ul>
  </li>
  <li> <code>$ pip install git+git://github.com/Theano/Theano.git</code> </li>
  <li> <code>$ pip install Pillow</code> </li>
  <li> <code>$ pip install h5py</code>.
  <li> <code>$ pip install sklearn</code>.
    <ul>
      <li> This will get an error along the way, but don't worry - it fixes itself. </li>
      <li> If it doesn't work at all, <code>$ conda install h5py</code> instead. </li>
    </ul>
  </li?
</ol>

Whenever you want to use this environment (which includes all of the things you just installed), simply run <code>$ source activate py27x</code>. Then run <code>source deactivate</code> when you're finished.

If you want to use the GPU (which is pretty much necessary if you want to train a deep CNN model), you need to install CUDA. You can download it here: https://developer.nvidia.com/cuda-downloads. If you download the runfile version, you can install it by running the provided bash script. If the appropriate Nvidea drivers are already installed, this will give you the option to install CUDA in a directory of your choosing (without root access).


Using the Code
-----

This code must be run in a Python 2 environment with all of the required modules listed above installed. The <code>run.py</code> script is what allows you to train or test a model.

Run on CPU: <code>python run.py \<config-file\> [options]</code>

Run on GPU (requires CUDA): <code>./run_gpu \<config-file\> [options]</code>

The config file must be provided (see below). Other options are as follows:

<ul>
  <li> <code>-load-model _file_</code> Loads an existing model architecture from the given file (<code>_file_</code>). The saved architecture must match the input image size (specified by the config file). If this parameter is not provided, a new model will be built as defined in <code>model.py</code>. </li>
  <li> <code>-save-model _file_</code> Saves the model architecture used for training to the given file (training mode only). You may want to save the model after training to test it later. </li>
  <li> <code>-load-weights _file_</code> Loads existing weights from this file. The weights must match the architecture of the model being used. You can load weights for a training run to fine-tune the model's parameters. </li>
  <li> <code>-save-weights _file_</code> Saves the trained weights to this file (training mode only). You should save weights if you wish to test the trained model later. </li>
  <li> <code>--explicit-labels</code> Uses an explicitly-provided label distribution over the classes (i.e. soft labels) instead of just a 1-hot vector for labels. To use this option, the code must be formatted appropriately (see "Preparing Your Data" below). </li>
  <li> <code>--test</code> Runs a test on a model with existing weights instead of training. This option requires the <code>-load-weights</code> to be set. You may also want to explicity load the model (<code>-load-model</code>) that you used for training. </li>
  <li> <code>-confusion-matrix _file_</code> Saves a confusion matrix of the test results to the given file (test mode only). </li>
  <li> <code>-report-misclassified _file_</code> Saves a list of misclassified images to the given file (test mode only). Each line of this file will contain the path of a misclassified image, followed by its correct (ground truth) class, and then the class that was incorrectly predicted by the model. </li>
  <li> <code>-report-scores _file_</code> For each test image, writes the path of the image file followed by the prediction probabilities for each possible class to the file, one image per line (test mode only). The format is <i>/path/to/image prob0 prob1 ...</i>. </li>
</ul>


Preparing Your Data
-----

The data itself can consist of any RGB or Grayscale images saved on your computer. You need to specify three files that will tell the code where your data is and how it is organized:

<ul>
  <li> A file containing a list of class names. Each line of this file should be a string indicating a classname (e.g. "cat", "dog", "building", etc. </li>
  <li> A file containing a list of training images. Each line of this file should specify the <i>full path</i> of an image, followed by a space and their numerical class number (e.g. 0, 1, 2, ..., N-1) where N is the total number of classes.
  <li> A file containing a list of test images. As for the training images, each line must specify the full path of an image followed by the class number. </li>
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
/home/users/You/data/cats/img1.jpg 0
/home/users/You/data/cats/img2.jpg 0
/home/users/You/data/cats/img3.jpg 0
/home/users/You/data/cats/img4.jpg 0
/home/users/You/data/cats/img5.jpg 0
# 5 dog training images
/home/users/You/data/dogs/img1.jpg 1
/home/users/You/data/dogs/img2.jpg 1
/home/users/You/data/dogs/img3.jpg 1
/home/users/You/data/dogs/img4.jpg 1
/home/users/You/data/dogs/img5.jpg 1
~~~~~

~~~~~
# test_images.txt
# 2 cat and 3 dog test images
/home/users/You/data/cats/img6.jpg 0
/home/users/You/data/dogs/img6.jpg 1
/home/users/You/data/cats/img7.jpg 0
/home/users/You/data/dogs/img7.jpg 1
/home/users/You/data/dogs/img8.jpg 1
~~~~~

<b><u>Using Soft Labels</u></b>:

If you want to explicity provide label values for each image (instead of just using 1-hot label vectors for the model), you can add the soft label values (floats) on each line of the train and test data files, after the class ID. For each image, you have to specify a weight for each possible class. For example, we can augment the above data with soft labels:

~~~~~
# train_images.txt
# 5 cat training images
/home/users/You/data/cats/img1.jpg 0 0.9 0.1
/home/users/You/data/cats/img2.jpg 0 0.8 0.2
/home/users/You/data/cats/img3.jpg 0 0.5 0.5
/home/users/You/data/cats/img4.jpg 0 1 0
/home/users/You/data/cats/img5.jpg 0 0.7 0.3
# 5 dog training images
/home/users/You/data/dogs/img1.jpg 1 0.1 0.9
/home/users/You/data/dogs/img2.jpg 1 0 1
/home/users/You/data/dogs/img3.jpg 1 0.3 0.7
/home/users/You/data/dogs/img4.jpg 1 0.25 0.75
/home/users/You/data/dogs/img5.jpg 1 0.4 0.6
~~~~~

~~~~~
# test_images.txt
# 2 cat and 3 dog test images
/home/users/You/data/cats/img6.jpg 0 0.9 0.1
/home/users/You/data/dogs/img6.jpg 1 0.2 0.8
/home/users/You/data/cats/img7.jpg 0 0.5 0.5
/home/users/You/data/dogs/img7.jpg 1 0.0 1.0
/home/users/You/data/dogs/img8.jpg 1 0.1 0.9
~~~~~

Note that there are 2 classes, so we provide 2 weights for each image. If there were N classes, you have to specify N weights per line.

An example use case for this is if the images have some visual similarities to images of other classes, then you can provide a better distribution to model this similarity better.


Setting Up the Config File
-----

Edit the <code>params.config</code> file (or write your own) with your specific data information. This file is formatted as <code>key: value</code> pairs on each line. All of the values must be defined, otherwise the configuration file will not be accepted. The values must be explicity defined as valid Python 2 types (e.g. a string must be in quotes, a tuple must be of the form <code>(val1, val2, val3)</code>, etc. Here is what you need to add:

<ol>
  <li> Point to the data files you defined above. Specifically, set up the appropriate <code>classnames_file</code>, <code>train_img_paths_file</code>, and <code>test_img_paths_file</code>. Note that these file paths must be in quotes (i.e. they will be interpreted as Python strings). </li>
  <li> Set the number of classes parameter: <code>number_of_classes</code>. Following the example files above, this values would be 2. </li>
  <li> Set the image dimensions (<code>img_dimensions</code>). All images will be resized to these dimensions (width, height, number of channels) before being fed into the CNN. <b>Currently, only 1 channel is supported (grayscale).</b> This is a TODO. </li>
  <li> Optionally, set the training hyperparameters as desired (<code>batch_size</code>, <code>num_epochs</code>, <code>learning_rate</code>, <code>decay</code>, and <code>momentum</code>). </li>
</ol>


Defining, Saving, and Loading a Model
-----

TODO
