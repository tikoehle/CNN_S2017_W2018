import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = W.shape[1]                                     # Number of classes (C)
  N = X.shape[0]                                     # Number of training records (N)

  for i in xrange(N):
    scores = X[i].dot(W)                             # (N, C)
    scores -= np.max(scores)                         # Numeric stability
    P = np.exp(scores) / np.sum(np.exp(scores))      # P: Output of the Softmax function. (N, C)
    CE = -1.0 * np.log(P[y[i]])                      # Cross-entropy of a single element X[i] 
    loss += CE

    for c in xrange(C):
      if c == y[i]:                                  # correct class, i == j
        dW[:,c] += (P[c] - 1.0) * X[i,:]             # Derivative of the Cross-entropy (complete layer including Softmax), case i == j
      else:                                          # incorrect classes, i != j
        dW[:,c] += P[c] * X[i,:]                     # Derivative of the Cross-entropy (complete layer including Softmax), case i != j

  dW /= N                                            # Average out the gradient
  dW += 2 * reg * np.sum(dW)                         # Derivative of the regularization for the gradient

  loss /= N                                          # Average out the loss
  loss += reg * np.sum(W * W)                        # Regularization of the loss
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  N_enum = np.arange(X.shape[0])
  scores = X.dot(W)
  scores -= np.max(scores)                                 # Numeric stability
  P = np.exp(scores).T / np.sum(np.exp(scores), axis=1)    # P[C,N] <---- Softmax output
  CE = -1.0 * np.log(P[y,N_enum])                          # CE[N,] <---- Cross-entropy of the Softmax
  loss = np.sum(CE)
  loss /= N
  loss += reg * np.sum(W * W)                              # Regularization of the loss

  kroneker_delta = np.zeros_like(P)
  kroneker_delta[y,N_enum] = 1
  dW = X.T.dot((P - kroneker_delta).T)                     # Derivative of the Cross-entropy
  dW /= N
  dW += 2 * reg * np.sum(dW)                               # Derivative of the regularization for the gradient
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

