from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    M = x.shape[0]                            # number of images in mini-batch
    D = w.shape[0]                            # number of image features
    x_reshaped = np.reshape(x, (M, D))        # x_reshaped shape of (M, D)
    out = np.dot(x_reshaped, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.dot(dout, w.T).reshape(x.shape)  # make dx and reshape into original x shape

    M = x.shape[0]
    D = w.shape[0]
    x_reshaped = np.reshape(x, (M,D))
    dw = np.dot(x_reshaped.T, dout)

    db = np.sum(dout, axis=0, keepdims=True)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.array(dout, copy=True)      # copying dout 
    dx[x <= 0] = 0                      # derivative of ReLU: dout / dx
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, Dim = x.shape  # Note (Timo): renamed D to Dim because D is a graph node.
    running_mean = bn_param.get('running_mean', np.zeros(Dim, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(Dim, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        A = 1. / N * np.sum(x, axis=0, keepdims=True)  # (1) A: mini-batch mean
        B = x - A                                      # (2)
        C = B ** 2                                     # (3)
        D = 1. / N * np.sum(C, axis=0, keepdims=True)  # (4) D: mini-batch variance
        E = np.sqrt(D + eps)                           # (5)
        F = 1. / E                                     # (6)
        G = F * B                                      # (7) G: normalize (x-hat)
        H = G * gamma                                  # (8) scale
        I = H + beta                                   # (9) I: out (y), shift

        x_mean = A
        x_var = D
        x_norm = G
        out = I

        cache = (B, D, E, F, G, eps, gamma, beta)
        
        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var

        # Debugging: For the bprop implementation, print the forward input 
        # shapes for each node of the computational graph. The derivative
        # of a node must return a local gradient of the same shape.
        """
        print('------- Graph nodes forward shape -------')
        graph_nodes = {'beta':beta, 'gamma':gamma, 
                       'A':A, 'B':B, 'C':C, 'D':D, 'E':E, 
                       'F':F, 'G':G, 'H':H, 'I':I}
        for n in graph_nodes.keys():
            print('%s: %s' % (n, str(graph_nodes[n].shape)))
        """
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x - running_mean) / np.sqrt(running_var + eps) # normalize
        y = gamma * x_norm + beta                                # scale and shift

        out = y
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    B, D, E, F, G, eps, gamma, beta = cache

    N, Dim = dout.shape  # Note: renamed D to Dim because D is a graph node.

    dbeta = np.sum(dout, axis=0)                          # (9)
    dH = dout                                             # (9)
    dgamma = np.sum(G * dH, axis=0)                       # (8)
    dG = gamma * dH                                       # (8)
    dF = np.sum(B * dG, axis=0, keepdims=True)            # (7)
    dB = F * dG                                           # (7)
    dE = -1. / E**2 * dF                                  # (6)
    dD = 1. / (2.0 * np.sqrt(D + eps)) * dE               # (5)
    dC = 1. / N * np.ones((N, Dim)) * dD                  # (4)
    dB += 2. * B * dC                                     # (3) sum dB gradients
    dA = -1. * np.sum(dB, axis=0, keepdims=True)          # (2)
    dx = 1. * dB                                          # (2)
    dx += 1. / N * np.ones((N, Dim)) * dA                 # (1) sum dx gradients

    """
    print('------- Graph nodes derivative shape -------')
    print('dbeta: %s' % str(dbeta.shape))
    print('dH: %s' % str(dH.shape))
    print('dgamma: %s' % str(dgamma.shape))
    print('dG: %s' % str(dG.shape))
    print('dF: %s' % str(dF.shape))
    print('dE: %s' % str(dE.shape))
    print('dD: %s' % str(dD.shape))
    print('dC: %s' % str(dC.shape))
    print('dB: %s' % str(dB.shape))
    print('dA: %s' % str(dA.shape))
    print('dx: %s' % str(dx.shape))
    """
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    B, D, E, F, G, eps, gamma, beta = cache
    N, Dim = dout.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(G * dout, axis=0)
    dx = (1. / N) * gamma * (D + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0) - B * (D + eps)**(-1.0) * np.sum(dout * B, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
        (higher p = less dropout)
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p  # Dropout mask. Notice /p (Inverted Dropout to omit scaling in predict)!
        out = x * mask                             # Drop!
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x   # Test time is unchanged. No scaling necessary because of /p!
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = mask * dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H1, W1 = x.shape       # C: Number of Channels (Depth of the image)
    F, C, HH, WW = w.shape       # F: Number of Filters (also called K; D2 = K)
    S = conv_param['stride']     # Receptive Field (C, HH, WW); size: (HH, WW)
    P = conv_param['pad']

    H2 = int(1 + (H1 + 2 * P - HH) / S)
    W2 = int(1 + (W1 + 2 * P - WW) / S)
    out = np.zeros((N, F, H2, W2))

    zero_pad = ((0,0), (0,0), (P,P), (P,P))
    #  x:         N      C      H1     W1        Zero-pad only (H1, W1) of x
    X_pad = np.pad(x, zero_pad, 'constant', constant_values=0)

    for n in range(N):
        for f in range(F):
            for h in range(H2):
                h_sta = h * S
                h_end = h_sta + HH
                for v in range(W2):
                    v_sta = v * S
                    v_end = v_sta + WW
                    receptive_field = X_pad[n,:,h_sta:h_end,v_sta:v_end]
                    out[n,f,h,v] = np.sum(receptive_field * w[f,:,:,:]) + b[f]
                    #print('n:%d  filter:%d  receptive_field:%s  filter_w%d:%s' % 
                    #    (n, f, str(receptive_field.shape), f, str(w[f,:,:,:].shape)))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H1, W1 = x.shape
    F, C, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    H2 = int(1 + (H1 + 2 * P - HH) / S)
    W2 = int(1 + (W1 + 2 * P - WW) / S)

    zero_pad = ((0,0), (0,0), (P,P), (P,P))
    X_pad = np.pad(x, zero_pad, 'constant', constant_values=0)

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx_pad = np.zeros_like(X_pad)   # Gradient of x including the pooling zero's.

    for n in range(N):
        for f in range(F):
            for h in range(H2):
                h_sta = h * S
                h_end = h_sta + HH
                for v in range(W2):
                    v_sta = v * S
                    v_end = v_sta + WW
                    dx_pad[n,:,h_sta:h_end,v_sta:v_end] += 1. * w[f,:,:,:] * dout[n,f,h,v]  # dout/dx
                    receptive_field = X_pad[n,:,h_sta:h_end,v_sta:v_end]
                    dw[f,:,:,:] += receptive_field * 1. * dout[n,f,h,v]   # dout/dw
                    db[f] += 1. * dout[n,f,h,v]  # dout/db

    dx = dx_pad[:,:,P:-P,P:-P]   # Remove the pooling zero's in dout/dx.
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H1, W1 = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    S = pool_param['stride']

    H2 = int(1 + (H1 - HH) / S)
    W2 = int(1 + (W1 - WW) / S)
    F = D = C   # F: number of filters; D remains unchanged with max pooling

    out = np.zeros((N, F, H2, W2))  # Downsampling the image to H2 x W2 (not D)
    for n in range(N):
        for f in range(F):
            for h in range(H2):
                h_sta = h * S
                h_end = h_sta + HH
                for v in range(W2):
                    v_sta = v * S
                    v_end = v_sta + WW
                    receptive_field = x[n,f,h_sta:h_end,v_sta:v_end]
                    max_pool = np.max(receptive_field)                    # (1)
                    out[n,f,h,v] = max_pool
                    #print('n:%d  filter:%d  h_sta:%d  h_end:%d  v_sta:%d  v_end:%d  receptive_field:%s  max_pool:%f' % 
                    #    (n, f, h_sta, h_end, v_sta, v_end, str(receptive_field.shape), max_pool))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H1, W1 = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    S = pool_param['stride']

    H2 = int(1 + (H1 - HH) / S)
    W2 = int(1 + (W1 - WW) / S)
    F = D = C

    dx = np.zeros_like(x)

    for n in range(N):
        for f in range(F):
            for h in range(H2):
                h_sta = h * S
                h_end = h_sta + HH
                for v in range(W2):
                    v_sta = v * S
                    v_end = v_sta + WW
                    receptive_field = x[n,f,h_sta:h_end,v_sta:v_end]
                    #-------------------------------------------------------------------------
                    # (1) Backward pass for the np.max() operation:
                    #     Routing the gradient to the input that had the highest value in the 
                    #     forward pass.
                    #-------------------------------------------------------------------------
                    # (1.1) Get the index of the max_pool element in the receptive_field matrix.
                    #       (sometimes also called "the switch")
                    i, j = np.unravel_index(np.argmax(receptive_field, axis=None), receptive_field.shape)
                    # (1.2) Create a all zeros matrix like receptive_field. 
                    dxx = np.zeros_like(receptive_field)
                    # (1.3) Route (store) the gradient to the index (i, j) of the max_pool element of the forwarding path.
                    dxx[i,j] = dout[n,f,h,v]
                    # (1.4) Store the gradient matrix dxx in dx.
                    dx[n,f,h_sta:h_end,v_sta:v_end] = dxx
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # NOTE: Hint from Course Instructor: Need to reshape the spatial dimensions.
    # https://www.reddit.com/r/cs231n/comments/443y2g/hints_for_a2/?st=j7ycdnkr&sh=a4775a0c
    N, C, H, W = x.shape
    xx = x.transpose(0, 2, 3, 1).reshape((N*H*W), C)          # after transpose: (N, H, W, C); xx: ((N*H*W), C)
    a, cache = batchnorm_forward(xx, gamma, beta, bn_param)   # a: ((N*H*W), C)
    out = np.reshape(a, (N, H, W, C)).transpose(0, 3, 1, 2)   # after reshape: (N, H, W, C); after transpose: (N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # TIMO: Same transpose/reshape dimension as used in spatial_batchnorm_forward.
    N, C, H, W = dout.shape
    xx = dout.transpose(0, 2, 3, 1).reshape((N*H*W), C)
    a, dgamma, dbeta = batchnorm_backward(xx, cache)
    dx = np.reshape(a, (N, H, W, C)).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
