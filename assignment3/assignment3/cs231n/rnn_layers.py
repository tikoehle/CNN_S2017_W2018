from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # Note: dot multiplication is not commutative: A.dot(B) != B.dot(A)
    Wxx = np.dot(x, Wx)                     # (1)
    Whh = np.dot(prev_h, Wh)                # (2)
    tanh = Wxx + Whh + b                    # (3)
    next_h = np.tanh(tanh)                  # (4)

    cache = (x, tanh, prev_h, Wx, Wh)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, tanh, prev_h, Wx, Wh = cache

    dtanh = (1 - np.square(np.tanh(tanh))) * dnext_h  # (4)   dnext_h/dtanh
    db = np.sum(dtanh, axis=0)              # (3.3) dtanh/db
    dWhh = dtanh                            # (3.2) dtanh/dWhh
    dWxx = dtanh                            # (3.1) dtanh/dWxx
    dWh = np.dot(prev_h.T, dWhh)            # (2.2) dWhh/dWh
    dprev_h = np.dot(dWhh, Wh.T)            # (2.1) dWhh/dprev_h <-- the "expensive" gradient highway: multiplies by Wh.T
    dWx = np.dot(x.T, dWxx)                 # (1.2) dWxx/dWx
    dx = np.dot(dWxx, Wx.T)                 # (1.1) dWxx/dx
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    cache = {}
    N, T, H = x.shape[0], x.shape[1], h0.shape[1]
    h = np.zeros((N,T,H))
    h_prev = h0
    for t in range(T):
        h[:,t,:], cache[t] = rnn_step_forward(x[:,t,:], h_prev, Wx, Wh, b)
        h_prev = h[:,t,:]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape

    # Initial derivative of ht is the upstream gradient of the last time step dh[:,-1,:].
    dx, dprev_h, dWx, dWh, db = rnn_step_backward(dh[:,-1,:], cache[list(cache.keys())[-1]])

    # Next, walk the computational graph from the second last time step t to h0.
    for t in reversed(range(T-1)):

        # The gradient input for h at time step t is the sum of the gradients of
        # the individual loss functions (dh) and the computed upstream gradient 
        # of h (dprev_h).
        dupstream = dh[:,t,:] + dprev_h
        _dx, dprev_h, _dWx, _dWh, _db = rnn_step_backward(dupstream, cache[t])

        # Stack each time step dx in a sequence along a 3rd dimension.
        dx  = np.dstack((dx, _dx))

        # Sum up these gradients.
        dWx += _dWx
        dWh += _dWh
        db  += _db

    # Reverse the stacked sequence of dx along the sequence axis.
    dx = dx[:,:,::-1]

    # And transpose the result from (N, D, T) to (N, T, D).
    dx = dx.transpose(0,2,1)

    # rnn_step_backward() returns the final gradient for h at time step t0.
    dh0 = dprev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # Index into W's V by x's T. This works across the N dimension of x. The
    # values it pulls from W are in D. So the out result is (N, T, D).
    out = W[x[:,:],:]
    cache = (x, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    dW = np.zeros_like(W)
    # Add the values in the D dimension of x to the D dimension of dout.
    # Finally, the result in the D dimension of dout is broadcasted to dW.
    # dout: (N,T,D) -- bcast+sum(N-axis)--> dW: (V,D). Note that T.shape != V.shape.
    # The values are summed up across the N dimension.
    np.add.at(dW, x[:,:], dout[:,:,:])
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    a = np.dot(x, Wx)+ np.dot(prev_h, Wh) + b     # (1) Activation vector a [4H]
    a_i, a_f, a_o, a_g = np.split(a, 4, axis=1)   # (2) Split a into 4 vectors of size [H]
    i = sigmoid(a_i)                              # (3) Compute the 4 gates of size [H]
    f = sigmoid(a_f)
    o = sigmoid(a_o)
    g = np.tanh(a_g)
    next_c = f * prev_c + i * g                   # (4) Next cell state
    tanh = np.tanh(next_c)                        # (5)
    next_h = o * tanh                             # (6) Next hidden state

    cache = (x, i, f, o, g, a_i, a_f, a_o, a_g, tanh, prev_h, prev_c, next_c, Wx, Wh)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, i, f, o, g, a_i, a_f, a_o, a_g, tanh, prev_h, prev_c, next_c, Wx, Wh = cache

    dtanh = o * dnext_h                                # (6) dnext_h/dtanh
    do = tanh * dnext_h                                #     dnext_h/do
    dnext_c += (1 - np.square(np.tanh(next_c))) * dtanh# (5) dtanh/dnext_c <-- Note: sum up the gradients for ct
    df = prev_c * dnext_c                              # (4) dnext_c/df
    dprev_c = f * dnext_c                              #     dnext_c/dprev_c <-- the "fast" gradient highway: multiply by f
    di = g * dnext_c                                   #     dnext_c/di
    dg = i * dnext_c                                   #     dnext_c/dg
    da_g = (1 - np.square(np.tanh(a_g))) * dg          # (3) dg/da_g
    da_o = o * (1 - o) * do                            #     do/da_o
    da_f = f * (1 - f) * df                            #     df/da_f
    da_i = i * (1 - i) * di                            #     di/da_i
    da = np.hstack((da_i, da_f, da_o, da_g))           # (2) da = hstack(da_i, da_f, da_o, da_g)
    dx = np.dot(da, Wx.T)                              # (1) da/dx
    dWx = np.dot(x.T, da)                              #     da/dWx
    dprev_h = np.dot(da, Wh.T)                         #     da/dprev_h
    dWh = np.dot(prev_h.T, da)                         #     da/dWh
    db = np.sum(da, axis=0)                            #     da/db
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    cache = {}
    N, T, H = x.shape[0], x.shape[1], h0.shape[1]
    h = np.zeros((N,T,H))
    prev_h = h0
    prev_c = np.zeros((N,H))       # Note, the initial cell state is set to zero.
    for t in range(T):
        h[:,t,:], next_c, cache[t] = lstm_step_forward(x[:,t,:], prev_h, prev_c, Wx, Wh, b)
        prev_h = h[:,t,:]
        prev_c = next_c            # Also note that the cell state is internal.
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape

    # This is almost the same as rnn_backward() except for the cell state dprev_c
    # which must be initialized to zero also in the backward step.
    dprev_c = np.zeros((N,H))
    dx, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dh[:,T-1,:], dprev_c, cache[T-1])

    # Next, walk the computational graph from the second last time step t to t0.
    for t in reversed(range(T-1)):

        # The gradient input for h at time step t is the sum of the gradients of
        # the individual loss functions (dh) and the computed upstream gradient
        # of h (dprev_h).
        dupstream = dh[:,t,:] + dprev_h
        _dx, dprev_h, dprev_c, _dWx, _dWh, _db = lstm_step_backward(dupstream, dprev_c, cache[t])

        # Stack each time step dx in a sequence along a 3rd dimension.
        dx  = np.dstack((dx, _dx))

        # Sum up these gradients.
        dWx += _dWx
        dWh += _dWh
        db  += _db

    # Reverse the stacked sequence of dx along the sequence axis.
    dx = dx[:,:,::-1]

    # And transpose the result from (N, D, T) to (N, T, D).
    dx = dx.transpose(0,2,1)

    # rnn_step_backward() returns the final gradient for h at time step t0.
    dh0 = dprev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
