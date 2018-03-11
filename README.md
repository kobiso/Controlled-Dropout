Controlled Dropout
=======
**"Controlled Dropout"** is a different dropout method which is for improving training speed on deep neural networks.
Basic idea and algorithm of controlled dropout are based on the paper "Controlled Dropout: a Different Dropout for Improving Training Speed on Deep Neural Network" which was presented in IEEE International Conference on Systems, Man, and Cybernetics (SMC) 2017.
Implementation of controlled dropout is done by revising [DeepNet](https://github.com/nitishsrivastava/deepnet) which is an implementation of conventional dropout.
DeepNet includes GPU-based python implementation of

1.  Feed-forward Neural Nets
2.  Restricted Boltzmann Machines
3.  Deep Belief Nets
4.  Autoencoders
5.  Deep Boltzmann Machines
6.  Convolutional Neural Nets

which is built on top of the [cudamat](http://code.google.com/p/cudamat/) library by Vlad Mnih and [cuda-convnet](http://code.google.com/p/cuda-convnet/) library by Alex Krizhevsky. We revised Feed-forward Neural Nets and Convolutional Neural Nets for Controlled Dropout.

- [**Presentation material in SMC 2017**](https://www.slideshare.net/ByungSooKo1/controlled-dropout-a-different-dropout-for-improving-training-speed-on-deep-neural-network)

## Research Summary
- **Controlled Dropout**
<p align="center"><img src="/images/controlled_dropout.png" height="250" width="700"></p>

- **Problem Statement**
    - Dropout takes longer time to train deep neural networks.
    - **Observation**: Dropout results in many zero element multiplications, which is redundant computation.

- **Research Objective**
    - To improve training speed of deep neural network while exhibiting the same generalization with dropout

- **Solution Proposed: Controlled Dropout**
    - Improve training speed by eliminating redundant computation of dropout
    - Drop units in a column-wise or row-wise manner, and train the network using compressed matrices

- **Contribution**
    - Training speed of FFNN and CNN is improved on both GPU and CPU.
    - Training speed improvement increases when the number of fully-connected layers increases.

## Installation

1. Dependencies
	- Numpy
	- Scipy
	- CUDA Toolkit and SDK
		- Install the toolkit and SDK. Set an environment variable CUDA_BIN to the path to the /bin directory of the cuda installation and CUDA_LIB to the path to the /lib64 (or /lib) directory. Also add them to PATH and LD_LIBARAY_PATH.

		- For example, add the following lines to your ~/.bashrc file
		export CUDA_BIN=/usr/local/cuda-5.0/bin
		export CUDA_LIB=/usr/local/cuda-5.0/lib64
		export PATH=\${CUDA_BIN}:\$PATH
		export LD_LIBRARY_PATH=\${CUDA_LIB}:\$LD_LIBRARY_PATH

    - [Protocol Buffers](http://code.google.com/p/protobuf/)
    	- Make sure that the PATH environment variable includes the directory that contains the protocol buffer compiler - protoc. For example, export PATH=/usr/local/bin:$PATH

2. Compiling Cudamat and Cudamat_conv
    - DeepNet uses Vlad Mnih's cudamat library and Alex Krizhevsky's cuda-convnet library. Some additional kernels have been added. To compile the library - run make in the cudamat dir.

3. Set environment variables
    - Add the path to cudamat to LD_LIBRARY_PATH. For example if DeepNet is located in the home dir, export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/deepnet/cudamat
    - Add the path to DeepNet to PYTHONPATH. For example, if DeepNet is located in the home dir, export PYTHONPATH=\$PYTHONPATH:\$HOME/deepnet

## Usage
  - [MNIST](http://www.cs.toronto.edu/~nitish/deepnet/mnist.tar.gz) dataset example
	  - This dataset consists of labelled images of handwritten digits as numpy files.
	  - cd to the deepnet/deepnet/examples dir
	  - run
	    \$ python setup_examples.py <path to mnist dataset> <output path>
	    This will modify the example models and trainers to use the specified

    - There are examples of different deep learning models. Go to any one and execute runall.sh. For example, cd to deepnet/deepnet/examples/rbm and execute \$ ./runall.sh  This should start training an RBM model.

  - [Cifar-10 and Cifar-100](https://www.cs.toronto.edu/~kriz/cifar.html) datasets
    - Cifar-10 and Cifar-100 are labeled subsets of the 80 million tiny images dataset.

  - [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset
    - SVHN is labeled subsets of house numbers in Google Street View.

## Reference
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
- [DeepNet](https://github.com/nitishsrivastava/deepnet)
  
## Author
Byung Soo Ko / kobiso62@gmail.com
