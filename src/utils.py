"""
Source Code for Homework 3 of ECBM E6040, Spring 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
"""
import os
import sys
import numpy
import scipy.io

import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from reading_mnist_imply import load_mnist


def shared_dataset(data_xy, borrow=True):
  """ Function that loads the dataset into shared variables

  The reason we store our dataset in shared variables is to allow
  Theano to copy it into the GPU memory (when code is run on GPU).
  Since copying data into the GPU is slow, copying a minibatch everytime
  is needed (the default behaviour if the data is not in a shared
  variable) would lead to a large decrease in performance.
  """
  data_x, data_y = data_xy
  shared_x = theano.shared(numpy.asarray(data_x,
                                         dtype=theano.config.floatX),
                           borrow=borrow)
  shared_y = theano.shared(numpy.asarray(data_y,
                                         dtype=theano.config.floatX),
                           borrow=borrow)
  # When storing data on the GPU it has to be stored as floats
  # therefore we will store the labels as ``floatX`` as well
  # (``shared_y`` does exactly that). But during our computations
  # we need them as ints (we use labels as index, and if they are
  # floats it doesn't make sense) therefore instead of returning
  # ``shared_y`` we will have to cast it to int. This little hack
  # lets ous get around this issue
  return shared_x, T.cast(shared_y, 'int32')


def load_data(ds_rate=None, theano_shared=True):
  ''' Loads the SVHN dataset

  :type ds_rate: float
  :param ds_rate: downsample rate; should be larger than 1, if provided.

  :type theano_shared: boolean
  :param theano_shared: If true, the function returns the dataset as Theano
  shared variables. Otherwise, the function returns raw data.
  '''
  if ds_rate is not None:
    assert(ds_rate > 1.)

  # Download the SVHN dataset if it is not present
  def check_dataset(dataset):
    # Check if dataset is in the data directory.
    new_path = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data",
        dataset
    )
    if (not os.path.isfile(new_path)):
      from six.moves import urllib
      origin = (
          'http://ufldl.stanford.edu/housenumbers/' + dataset
      )
      print('Downloading data from %s' % origin)
      urllib.request.urlretrieve(origin, new_path)
    return new_path

  train_dataset = check_dataset('extra_32x32.mat')
  test_dataset = check_dataset('test_32x32.mat')

  # Load the dataset
  train_set = scipy.io.loadmat(train_dataset)
  test_set = scipy.io.loadmat(test_dataset)

  # Convert data format
  def convert_data_format(data):
    X = numpy.reshape(data['X'],
                      (numpy.prod(data['X'].shape[:-1]), data['X'].shape[-1]),
                      order='C').T / 255.
    y = data['y'].flatten()
    y[y == 10] = 0
    return (X, y)
  train_set = convert_data_format(train_set)
  test_set = convert_data_format(test_set)

  # Downsample the training dataset if specified
  train_set_len = len(train_set[1])
  if ds_rate is not None:
    train_set_len = int(train_set_len // ds_rate)
    train_set = [x[:train_set_len] for x in train_set]

  # Extract validation dataset from train dataset
  valid_set = [x[-(train_set_len // 10):] for x in train_set]
  train_set = [x[:-(train_set_len // 10)] for x in train_set]

  # train_set, valid_set, test_set format: tuple(input, target)
  # input is a numpy.ndarray of 2 dimensions (a matrix)
  # where each row corresponds to an example. target is a
  # numpy.ndarray of 1 dimension (vector) that has the same length as
  # the number of rows in the input. It should give the target
  # to the example with the same index in the input.

  if theano_shared:
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
  else:
    rval = [train_set, valid_set, test_set]

  return rval


def load_data_cifar10(data_path='data/cifar-10-batches-mat/', theano_shared=True):
  '''
  Load Cifar10 Data (Matlab Version) to numpy arrays
  Accumulate Batches of data
  Create Train-Valid split
  Generate Theano Variable

  :type data_path: string
  :param data_path: path from where to laod the data

  :type theano_shared: boolean
  :param theano_shared: If true, the function returns the dataset as Theano
  shared variables. Otherwise, the function returns raw data.  
  '''

  train_data_x = numpy.zeros((40000, 3072))
  train_data_y = numpy.zeros(40000)

  valid_data_x = numpy.zeros((10000, 3072))
  valid_data_y = numpy.zeros(10000)

  test_data_x = numpy.zeros((10000, 3072))
  test_data_y = numpy.zeros(10000)

  file_name_list = ['data_batch_{}.mat'.format(x) for x in range(1, 5)]

  for i, file_name in enumerate(file_name_list):
    data_temp = scipy.io.loadmat(data_path + file_name)
    temp_cnt, _ = data_temp['data'].shape
    train_data_x[i * temp_cnt: (i + 1) * temp_cnt, :] = data_temp['data']
    train_data_y[i * temp_cnt: (i + 1) * temp_cnt] = data_temp['labels'].reshape(temp_cnt)

  data_temp = scipy.io.loadmat(data_path + 'data_batch_5.mat')
  valid_data_x[:, :] = data_temp['data']
  valid_data_y[:] = data_temp['labels'].reshape(temp_cnt)

  data_temp = scipy.io.loadmat(data_path + 'test_batch.mat')
  test_data_x[:, :] = data_temp['data']
  test_data_y[:] = data_temp['labels'].reshape(temp_cnt)

  if theano_shared:
    test_set_x, test_set_y = shared_dataset((test_data_x, test_data_y))
    valid_set_x, valid_set_y = shared_dataset((valid_data_x, valid_data_y))
    train_set_x, train_set_y = shared_dataset((train_data_x, train_data_y))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

  return [(train_data_x, train_data_y), (valid_data_x, valid_data_y), (test_data_x, test_data_y)]


def load_data_mnist(data_path='data/mnist/', theano_shared=True):
  '''
  Load Mnist Data (Matlab Version) to numpy arrays
  Accumulate Batches of data
  Create Train-Valid split
  Generate Theano Variable

  :type data_path: string
  :param data_path: path from where to laod the data

  :type theano_shared: boolean
  :param theano_shared: If true, the function returns the dataset as Theano
  shared variables. Otherwise, the function returns raw data.  
  '''

  train_data = load_mnist(dataset='training', path=data_path)
  test_data = load_mnist(dataset='testing', path=data_path)

  valid_data_x = numpy.zeros((test_data[0].shape[0], test_data[0].shape[1]))
  valid_data_y = numpy.zeros(test_data[0].shape[0])

  valid_data_x = train_data[0][-valid_data_x.shape[0]:, :]
  valid_data_y = train_data[1][-valid_data_x.shape[0]:, :].reshape(valid_data_x.shape[0])

  train_data_x = train_data[0][:train_data[0].shape[0] - valid_data_x.shape[0], :]
  train_data_y = train_data[1][:train_data[0].shape[0] - valid_data_x.shape[0], :].reshape(train_data_x.shape[0])

  test_data_y = test_data[1].reshape(test_data[1].shape[0])
  test_data_x = test_data[0]

  train_data_y_mat = numpy.float32(numpy.eye(10)[train_data_y])

  # for hinge loss
  train_data_y_mat = 2* train_data_y_mat - 1.

  if theano_shared:
    test_set_x, test_set_y = shared_dataset((test_data_x, test_data_y))
    valid_set_x, valid_set_y = shared_dataset((valid_data_x, valid_data_y))
    train_set_x, train_set_y = shared_dataset((train_data_x, train_data_y))

    borrow=True
    train_data_y_mat = theano.shared(numpy.asarray(train_data_y_mat, dtype=theano.config.floatX),borrow=borrow)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), train_data_y_mat]
    return rval

  return [(train_data_x, train_data_y), (valid_data_x, valid_data_y), (test_data_x, test_data_y)]


def plot_graph(arr, xlabel='Epochs', ylabel='Validation Error', file_name='mnist_validation_loss.jpg'):
  '''
  plot a graph and save it to disk

  :type arr: list of float/int
  :param arr: y axis parameters to plot

  :type xlabel: string
  :param xlabel: x label of the plot

  :type ylabel: string
  :param ylabel: y label of the plot

  :type file_name: string
  :param file_name: file name of the saved graph file
  '''
  
  x_range = range(1,len(arr)+1)
  plt.plot(x_range, arr)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(file_name)

if __name__ == '__main__':
  load_data_cifar10()
