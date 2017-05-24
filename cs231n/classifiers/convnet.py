import numpy as np
# -*- coding:utf8 -*-
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
  
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, n=1, m=2):
        """
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
        self.filter_size = filter_size
        self.n = n
        self.m = m

        # Initialize weights and biases
        C, H, W = input_dim

        # [conv-relu-conv-relu-pool] * n
        num_layers = 0


        for i in xrange(n):
            ## conv 1 ##
            ##  W1:(F1,C,F1H,F1W) ##
            ##  b1:(1,F1)  ##
            if i == 0:
                num_layers += 1
                self.params['W' + str(num_layers)] = weight_scale * np.random.randn(num_filters, C, filter_size,
                                                                        filter_size)
                self.params['b' + str(num_layers)] = np.zeros((1,num_filters))
            else:
                num_layers += 1
                self.params['W' + str(num_layers)] = weight_scale * np.random.randn(num_filters, num_filters, filter_size,
                                                                                    filter_size)
                self.params['b' + str(num_layers)] = np.zeros((1, num_filters))
            ## conv 2 ##
            ## W2:(F2,F1,F2H,F2W) ##
            ## b2:(1,F2) ##
            num_layers += 1
            self.params['W' + str(num_layers)] = weight_scale * np.random.randn(num_filters, num_filters, filter_size,
                                                                                filter_size)
            self.params['b' + str(num_layers)] = np.zeros((1,num_filters))

            ## max_pool causes height and width divided by 2 respectively ##
            H /= 2
            W /= 2

        ## affine 1 ##
        num_layers += 1
        self.params['W' + str(num_layers)] = weight_scale * np.random.randn(num_filters * H * W, hidden_dim)
        self.params['b' + str(num_layers)] = np.zeros((1,hidden_dim))

        ## rest affine layers - 2 ##
        for i in xrange(m - 2):
            num_layers += 1
            self.params['W' + str(num_layers)] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
            self.params['b' + str(num_layers)] = np.zeros((1,hidden_dim))

        ## last affine ##
        num_layers += 1
        self.params['W' + str(num_layers)] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b' + str(num_layers)] = np.zeros((1,num_classes))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # Compute the FP
        conv_relu_out, conv_relu_pool_out, affine_relu_out = {}, {}, {}
        cache1, cache2, cache3 = {}, {}, {}
        conv_relu_pool_out[0] = X

        num_layers = 0
        for i in xrange(self.n):
            ##  conv 1 ##
            num_layers += 1
            W1, b1 = self.params['W' + str(num_layers)], self.params['b' + str(num_layers)]
            conv_relu_out[i], cache1[i] = conv_relu_forward(conv_relu_pool_out[i], W1, b1, conv_param)
            ## conv 2 ##
            num_layers += 1
            W2, b2 = self.params['W' + str(num_layers)], self.params['b' + str(num_layers)]
            conv_relu_pool_out[i + 1], cache2[i] = conv_relu_pool_forward(conv_relu_out[i], W2, b2, conv_param,
                                                                          pool_param)

        ## affine 0 ##
        num_layers += 1
        W3, b3 = self.params['W' + str(num_layers)], self.params['b' + str(num_layers)]
        affine_relu_out[0], cache3[0] = affine_relu_forward(conv_relu_pool_out[self.n], W3, b3)

        ## m-2 affine ##
        if self.m > 2:
            for i in xrange(self.m - 2):
                num_layers += 1
                W4, b4 = self.params['W' + str(num_layers)], self.params['b' + str(num_layers)]
                affine_relu_out[i + 1], cache3[i + 1] = affine_relu_forward(affine_relu_out[i], W4, b4)

        ##  affine 1 ##
        num_layers += 1
        W5,b5 = self.params['W' + str(num_layers)], self.params['b' + str(num_layers)]
        scores, cache4 = affine_forward(affine_relu_out[self.m - 2],W5,b5)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, reg_loss, grads = 0.0, 0.0, {}

        # compute the BP
        data_loss, dscores = softmax_loss(scores, y)
        for i in xrange(num_layers):
            reg_loss += 0.5 * self.reg * np.sum(self.params['W' + str(i+1)]**2)
        loss = data_loss + reg_loss

        dconv_relu_in, dconv_relu_pool_in, daffine_in = {}, {}, {}
        ## affine 1 ##
        daffine_in[self.m],grads['W'+str(num_layers)],grads['b'+str(num_layers)] = affine_backward(dscores,cache4)
        num_layers -= 1
        ## rest m-1 affines ##
        for i in xrange(self.m - 1):
            daffine_in[self.m - 1 - i],grads['W'+str(num_layers)],grads['b'+str(num_layers)] = affine_relu_backward(daffine_in[self.m-i],cache3[self.m-2-i])
            num_layers -= 1

        dit_upstream = daffine_in[1]
        for i in xrange(self.n):
            dconv_relu_pool_in[self.n-1-i],grads['W'+str(num_layers)],grads['b'+str(num_layers)] = conv_relu_pool_backward(dit_upstream,cache2[self.n-1-i])
            num_layers -= 1
            dconv_relu_in[self.n-1-i],grads['W'+str(num_layers)],grads['b'+str(num_layers)] = conv_relu_backward(dconv_relu_pool_in[self.n-1-i],cache1[self.n-1-i])
            num_layers -= 1
            dit_upstream = dconv_relu_in[self.n-1-i]

        # Add regularization gradient contribution
        for i in xrange(num_layers):
            grads['W' + str(i + 1)] += self.reg * self.params['W' + str(i + 1)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


pass
