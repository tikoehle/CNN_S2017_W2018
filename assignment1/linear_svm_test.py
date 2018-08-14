#!/usr/bin/env python

# -----------------------------------------------------------------------------------------
# Debugging and testing for:
# assignment1 / linear_svm.py: svm_loss_vectorized
# Nov 18, 2017
#-----------------------------------------------------------------------------------------
import random
import numpy as np
from past.builtins import xrange

W = np.array([[0.01, -0.05, 0.1, 0.05],[0.7, 0.2, 0.05, 0.16],[0, -0.45, -0.2, 0.03]])
X = np.array([[-15,-18,-3],[22,28,15],[-44,-48,-12],[56,58,99]])
B = np.array([[0.0,0.0,0.0],[0.2,0.2,0.2],[-0.3,-0.3,-0.3]])
y = np.array([0,2,0])
#--------------------------------------------
# Transpose according to given inputs shape
#--------------------------------------------
W = W.T
X = X.T
B = B.T

print('--- X----')
print X
reg = 0.000005
dW = np.zeros(W.shape)
num_classes = W.shape[1]
num_train = X.shape[0]
loss = 0.0
for i in xrange(num_train):
  incorrect_class_count = 0
  scores = X[i].dot(W)
  print('-------- class scores for image: %s' % (X[i]))
  print(scores)

  correct_class_score = scores[y[i]]
  print('correct class score:%f' % (correct_class_score))

  for j in xrange(num_classes):
  
    if j == y[i]:
    	print('-------- skip correct class %d' % (j))
    	continue
    margin = scores[j] - correct_class_score + 1 # note delta = 1

    if margin > 0:
      loss += margin
      print('-------- class j:%d, margin:%f, scores[j]:%f, correct_class_score:%f ----' % (j, margin, scores[j], correct_class_score))
      incorrect_class_count += 1
      dW[:,j] += X[i]
      print('-------- dWj incorrect class update: X[i]: %d, class j: %d ----' % (i, j))
      print(dW)
    else:
    	print('-------- margin < 0, skipping class %d' % (j))

  dW[:,y[i]] += -incorrect_class_count * X[i]
  print('-------- dWyi correct class update: X[i]: %d, class y[i]: %d, incorrect_class_count: %d ----' % (i, y[i], incorrect_class_count))
  print(dW)

dW /= num_train
print('naive gradient w/o reg:')
print dW
dW += 2 * reg * np.sum(dW)
loss /= num_train
loss += reg * np.sum(W * W)
print('naive loss: %f' % (loss))
print('naive gradient w/ reg:')
print dW
grad_naive = dW

#--------------------------------------------------------------
loss = 0.0
dW = np.zeros(W.shape) # initialize the gradient as zero
del(scores)
#############################################################################
# TODO:                                                                     #
# Implement a vectorized version of the structured SVM loss, storing the    #
# result in loss.                                                           #
#############################################################################
delta = 1
num_train = X.shape[0]
scores = X.dot(W)                                       # with Bias: scores = X.dot(W) + B
i = np.arange(y.shape[0])
D = (scores.T - scores[i,y]).T + delta
D[D < 0] = 0
D[i,y] = 0
loss = np.sum(D)
loss /= num_train
loss += reg * np.sum(W * W)
print('vectorized loss: %f' % (loss))
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
#print dWj

XX = X.T * -1.0 * (D>0).sum(axis=1)                         # MUST -1.0 not -1; ALL IMAGES HAVE TO BE SCALED BEFORE DOING THE DOT PRODUCT
correct_class_scores = (scores.T - scores[i,y]).T           #    OTHERWISE THE _SUM_ OF THE IMAGES IN ONE CLASS WILL BE * -incorrect_class_count
dWyi = XX.dot((correct_class_scores == 0))              # THE the dot product with the bool mask is the ultimative dWyi += .. replacement
print('---dWyi vectorized----')
print dWyi

dW = dWj + dWyi

dW /= num_train
print('vectorized gradient w/o reg:')
print dW
dW += 2.0 * reg * np.sum(dW)                                  # MUST must use 2.0 and not 2 
print('vectorized gradient w/ reg:')
print dW

grad_vectorized = dW
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('difference: %f' % difference)










