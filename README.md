## Binary Connect
### Experiments with binarizing the weights of Neural Networks as done in [BinaryConnect](https://arxiv.org/pdf/1511.00363.pdf)

## Authors
### Shubham Bansal, Aman Chahar, Nitesh Chauhan

## Prerequisites
[Theano](http://deeplearning.net/software/theano/library/tensor/basic.html) is required for redirecting processing to GPU.

## Instructions
### Data Directory
Get the [mnist](http://yann.lecun.com/exdb/mnist/) dataset and [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) matlab version dataset. use pylearn2 to preprocess the SHVN dataset as described in [orignal repo](https://github.com/MatthieuCourbariaux/BinaryConnect).

````
cd $(PROJ_DIR)
mkdir -p data
mkdir -p data/mnist
mkdir -p data/cifar-10-batches-mat
````
Put the cifar batch files and mnist ubytes files in their respective folder.