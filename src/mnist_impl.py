import numpy
import sys, time
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import shared_dataset, load_data_mnist, plot_graph
from neural_network import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn

def test_mnist(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=128, n_hidden=500, n_hiddenLayers=3, normalization=True, eps=1e-4,
             verbose=False, smaller_set=True, loss='norm', lr_decay=False, binary=True):
    """
    Wrapper function for training and testing MLP

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    :type loss: string
    :param loss: to use hinge loss or normal loss.

    :type lr_decay: boolean
    :param lr_decay: to use learning_rate decay

    :type binary: boolean
    :param binary: to binarize the output

    :type normalization: boolean
    :param normalization: normalization output or not

    :type eps: float
    :param eps: normalization variable
    """

    # load the dataset; download the dataset if it is not present
    datasets = load_data_mnist(theano_shared=True)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_data_y_mat = datasets[3]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    y_mat = T.matrix('y_mat')
    epoch = T.lscalar('epoch')

    rng = numpy.random.RandomState(1234)

    # construct a neural network, either MLP or CNN.
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=28*28,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10,
        binary=binary,
        normalization=normalization,
        eps=eps
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    # Loss can chosen as hinge loss or nll loss
    if loss == 'norm':
        cost = (
            classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
        )
    else:
        cost = classifier.logRegressionLayer.hinge(y_mat)

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size],
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size],
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams

    # According to paper gradient is calculated using binarized weights as 
    # same weights are used during forward propagation
    if binary:
        gparams = [T.grad(cost, param) for param in classifier.params_bin]
    else:
        gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    # change learning rate depending upon input
    # we are following exponential decay with
    # learning rate = learning_rate_start * (learning_rate_final / learning_rate_start) ** (epoch/ n_epochs)

    if lr_decay:
        lr_start = learning_rate
        lr_final = 1e-6
        updates = [
            (param_i, T.cast(T.clip(param_i - (lr_start * (lr_final/lr_start)**(epoch/25)) * grad_i, -1, 1), theano.config.floatX))
            for param_i, grad_i in zip(classifier.params, gparams)
        ]
    else:
        updates = [
            (param_i, T.cast(T.clip(param_i - learning_rate * grad_i, -1, 1), theano.config.floatX))
            for param_i, grad_i in zip(classifier.params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`

    # Dependin upon lr_decay and loss function calculation we need to pass different
    # parameters to training model
    if loss=='norm':
        if lr_decay:
            train_model = theano.function(
                inputs=[index, epoch],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y: train_set_y[index * batch_size: (index + 1) * batch_size],
                }
            )
        else:
            train_model = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y: train_set_y[index * batch_size: (index + 1) * batch_size],
                }
            )
    else:
        if lr_decay:
            train_model = theano.function(
                inputs=[index, epoch],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y_mat: train_set_y[index * batch_size: (index + 1) * batch_size],
                }
            )
        else:
            train_model = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y_mat: train_set_y[index * batch_size: (index + 1) * batch_size],
                }
            )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    result = train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose, lr_decay)

    # plot_graph(result[2])
    return result


if __name__ == '__main__':
    test_mnist(verbose=True, learning_rate=3e-6, batch_size=128, n_hidden=[2048, 2048, 2048], n_hiddenLayers=3)
