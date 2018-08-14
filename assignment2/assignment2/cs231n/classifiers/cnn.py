from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        #---------------------------------------------------------------------------
        # Layer 1
        #---------------------------------------------------------------------------
        # CONV input dimensions
        #---------------------------------------------------------------------------
        F = num_filters
        C = input_dim[0]
        HH = WW = filter_size
        self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
        self.params['b1'] = np.zeros(F,)
        #---------------------------------------------------------------------------
        # Layer 2
        #---------------------------------------------------------------------------
        # CONV output dimensions
        #---------------------------------------------------------------------------
        HH = WW = filter_size
        P = (filter_size - 1) // 2                 # conv_param['pad']
        S = 1                                      # conv_param['stride']
        W1 = input_dim[1]
        H1 = input_dim[2]
        K = num_filters
        H2 = int(1 + (H1 + 2 * P - HH) / S)
        W2 = int(1 + (W1 + 2 * P - WW) / S)
        D2 = K
        #---------------------------------------------------------------------------
        # max_pool output dimensions
        #---------------------------------------------------------------------------
        HH = 2                                     # pool_param['pool_height']
        WW = 2                                     # pool_param['pool_width']
        S = 2                                      # pool_param['stride']
        H3 = int(1 + (H2 - HH) / S)
        W3 = int(1 + (W2 - WW) / S)
        D3 = D2                                    # D remains unchanged with max pooling
        fan_in = W3 * H3 * D3                      # max_pool output dimensions
        fan_out = hidden_dim
        self.params['W2'] = weight_scale * np.random.randn(fan_in, fan_out)
        self.params['b2'] = np.zeros((1, fan_out))
        #---------------------------------------------------------------------------
        # Layer 3
        #---------------------------------------------------------------------------
        # Affine output layer
        #---------------------------------------------------------------------------
        fan_in = hidden_dim
        fan_out = num_classes
        self.params['W3'] = weight_scale * np.random.randn(fan_in, fan_out)
        self.params['b3'] = np.zeros((1, fan_out))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        CNN, CNN_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        H1, H1_cache = affine_relu_forward(CNN, W2, b2)
        scores, OUT_cache = affine_forward(H1, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        #-----------------------------------------------------------
        # Compute the loss
        #-----------------------------------------------------------
        # dscores: Derivative of the Softmax/CE layer:  dCE/dP * dP/dscores
        data_loss, dscores = softmax_loss(scores, y)
        # L2 regularization includes the convenience factor of 0.5
        reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2) + 0.5 * self.reg * np.sum(W3 * W3)
        loss = data_loss + reg_loss
        #-----------------------------------------------------------
        # Backpropate the gradient to the parameters
        #-----------------------------------------------------------
        dH1, dW3, db3 = affine_backward(dscores, OUT_cache)
        dCNN, dW2, db2 = affine_relu_backward(dH1, H1_cache)
        _, dW1, db1 = conv_relu_pool_backward(dCNN, CNN_cache)
        #-----------------------------------------------------------
        # Add regularization gradient contribution
        #-----------------------------------------------------------
        # Derivatives of the reg_loss with convenience factor of 0.5
        dW3 += self.reg * W3
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        #-----------------------------------------------------------
        # Store the Gradients
        #-----------------------------------------------------------
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
