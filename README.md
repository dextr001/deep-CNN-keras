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

TODO
