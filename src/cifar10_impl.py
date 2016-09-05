import numpy

import theano, pudb
import theano.tensor as T
from theano.tensor.signal import downsample

from utils import shared_dataset, load_data, load_data_cifar10
from neural_network import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn

def test_cifar10(learning_rate=0.01, n_epochs=500, nkerns=[128, 256, 512], filter_shape=3,
            batch_size=200, verbose=False, normal=False, smaller_set=True, 
            std_normal=2, binary=True, normalization=True, eps=1e-4):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type binary: boolean
    :param binary: to binarize the output

    :type normalization: boolean
    :param normalization: normalization output or not

    :type eps: float
    :param eps: normalization variable    
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data_cifar10(theano_shared=True)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, filter_shape, filter_shape),
        poolsize=(1, 1),
        normal=normal,
        std_normal=std_normal,
        binary=binary,
        normalization=normalization,
        eps=eps
    )
    img_size = (32 - filter_shape + 1)
    layer00 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], img_size, img_size),
        filter_shape=(nkerns[0], nkerns[0], filter_shape, filter_shape),
        poolsize=(2, 2),
        normal=normal,
        std_normal=std_normal,
        binary=binary,
        normalization=normalization,
        eps=eps
    )

    # Construct the second convolutional pooling layer
    img_size = (img_size - filter_shape + 1)//2
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer00.output,
        image_shape=(batch_size, nkerns[0], img_size, img_size),
        filter_shape=(nkerns[1], nkerns[0], filter_shape, filter_shape),
        poolsize=(1, 1),
        normal=normal,
        std_normal=std_normal,
        binary=binary,
        normalization=normalization,
        eps=eps
    )

    img_size = (img_size - filter_shape + 1)
    layer11 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], img_size, img_size),
        filter_shape=(nkerns[1], nkerns[1], filter_shape, filter_shape),
        poolsize=(2, 2),
        normal=normal,
        std_normal=std_normal,
        binary=binary,
        normalization=normalization,
        eps=eps
    )

    img_size = (img_size - filter_shape + 1)//2
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer11.output,
        image_shape=(batch_size, nkerns[1], img_size, img_size),
        filter_shape=(nkerns[2], nkerns[1], filter_shape, filter_shape),
        poolsize=(2, 2),
        normal=normal,
        std_normal=std_normal,
        binary=binary,
        normalization=normalization,
        eps=eps
    )

    # img_size = (img_size - filter_shape + 1)
    # layer22 = LeNetConvPoolLayer(
    #     rng,
    #     input=layer2.output,
    #     image_shape=(batch_size, nkerns[2], img_size, img_size),
    #     filter_shape=(nkerns[2], nkerns[2], filter_shape, filter_shape),
    #     poolsize=(2, 2),
    #     normal=normal,
    #     std_normal=std_normal,
    #     binary=binary
    # )
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    img_size = (img_size - filter_shape + 1)//2
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * img_size * img_size,
        n_out=1024,
        activation=T.nnet.sigmoid,
        binary=binary,
        normalization=normalization,
        epsilon=eps
    )

    layer5 = HiddenLayer(
        rng,
        input=layer3.output,
        n_in=1024,
        n_out=1024,
        activation=T.nnet.sigmoid,
        binary=binary,
        normalization=normalization,
        epsilon=eps
    )

    layer6 = HiddenLayer(
        rng,
        input=layer5.output,
        n_in=1024,
        n_out=1024,
        activation=T.nnet.sigmoid,
        binary=binary,
        normalization=normalization,
        epsilon=eps
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer6.output, n_in=1024, n_out=10, binary=binary)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    #create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    params =  params + layer11.params + layer00.params# + layer22.params
    params =  params + layer5.params + layer6.params

    params_bin = layer4.params_bin + layer3.params_bin + layer2.params_bin + layer1.params_bin + layer0.params_bin
    params_bin =  params_bin + layer11.params_bin + layer00.params_bin# + layer22.params_bin
    params_bin =  params_bin + layer5.params_bin + layer6.params_bin    

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params_bin)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        # (param_i, param_i - learning_rate * grad_i)
        (param_i, T.cast(T.clip(param_i - learning_rate * grad_i, -1, 1), theano.config.floatX))
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    result = train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

    return result


if __name__ == "__main__":
    print test_cifar10(verbose=True)
