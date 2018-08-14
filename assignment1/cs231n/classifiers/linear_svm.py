import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero. dW need to return same shape as W.shape in order to make the gradient update of W.
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    incorrect_class_count = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        incorrect_class_count += 1  # it counts the classes which are contributing to loss (incorrect classes with margin > 0)
        dW[:,j] += X[i]                              # Derivative dWj which corresponds to ihe incorrect classes
    dW[:,y[i]] += -incorrect_class_count * X[i]      # Derivative dWyi which corresponds to the correct class
  dW /= num_train                                    # Average out the gradients dW/m
  dW += 2 * reg * np.sum(dW)                 # Derivative of the regularization term because the loss function also uses regularization

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  delta = 1
  num_train = X.shape[0]
  scores = X.dot(W)
  i = np.arange(y.shape[0])
  D = (scores.T - scores[i,y]).T + delta
  D[D < 0] = 0
  D[i,y] = 0
  loss = np.sum(D)
  loss /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # I am re-using D from the hinge-loss implementation above!
  dWj = X.T.dot((D>0))
  XX = X.T * -1.0 * (D>0).sum(axis=1)
  correct_class_scores = (scores.T - scores[i,y]).T
  dWyi = XX.dot((correct_class_scores == 0))
  dW = dWj + dWyi
  dW /= num_train
  dW += 2.0 * reg * np.sum(dW)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
