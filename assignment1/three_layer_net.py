'''
This Neural Network implementation is
Copyright of Timo Koehler - tikoehle@googlemail.com.

This file is part of the Stanford Class CS231n/assignment1
and contains my own code to solve this class assignment.
Therefore this code must not be published anywhere because
it contains the correct and working solutions.
'''
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    Input - H1 - ReLU - H2 - ReLU - Output Layer - Softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size_1: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        # FIXME:
        # In practice, the current recommendation is to use ReLU units and use
        # the w = np.random.randn(n) * sqrt(2.0/n); n  is the number of inputs
        # to the neuron
        self.params = {}   # Initializing the parameters
        self.params['W1'] = std * np.random.randn(input_size, hidden_size_1)
        self.params['b1'] = np.zeros((1, hidden_size_1))
        self.params['W2'] = std * np.random.randn(hidden_size_1, hidden_size_2)
        self.params['b2'] = np.zeros((1, hidden_size_2))
        self.params['W3'] = std * np.random.randn(hidden_size_2, output_size)
        self.params['b3'] = np.zeros((1, output_size))

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape
        
        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        hidden_layer_1 = np.maximum(0, np.dot(X, W1) + b1)          # ReLU activation
        hidden_layer_2 = np.maximum(0, np.dot(hidden_layer_1, W2) + b2)
        scores = np.dot(hidden_layer_2, W3) + b3
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores
        
        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        scores -= np.max(scores)
        theta = 1.0
        scores *= float(theta)
        
        exp_scores = np.exp(scores)
        P = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # P: Softmax output  [N x K]
        
        epsilon = 1e-15
        L = P[range(N), y]
        L[L == 0] = epsilon
        CE = -1.0 * np.log(L)
        # CE = -1.0 * np.log(P[range(N),y])      # Cross-entropy
        data_loss = np.sum(CE) / N               # Average cross-entropy
        reg_loss = reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2) + reg * np.sum(W3 * W3)
        loss = data_loss + reg_loss
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        #-----------------------------------------------------------
        # Derivative of the Softmax/CE layer:  dCE/dP * dP/dscores
        #-----------------------------------------------------------
        dscores = P
        dscores[range(N), y] -= 1     # dscores = P - K;    K: Kroneker Delta
        dscores /= N
        
        #-----------------------------------------------------------
        # Backpropate the gradient to the parameters
        #-----------------------------------------------------------
        # first backprop into parameters W3 and b3
        dW3 = np.dot(hidden_layer_2.T, dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer 2
        dhidden2 = np.dot(dscores, W3.T)
        # backprop the ReLU non-linearity
        dhidden2[hidden_layer_2 <= 0] = 0

        # backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer_1.T, dhidden2)
        db2 = np.sum(dhidden2, axis=0, keepdims=True)
        # backprop into hidden layer 1
        dhidden1 = np.dot(dhidden2, W2.T)
        # backprop the ReLU non-linearity
        dhidden1[hidden_layer_1 <= 0] = 0
        
        # finally into W1,b1
        dW1 = np.dot(X.T, dhidden1)
        db1 = np.sum(dhidden1, axis=0, keepdims=True)
        
        #-----------------------------------------------------------
        # Add regularization gradient contribution
        #-----------------------------------------------------------
        dW3 += 2 * reg * W3
        dW2 += 2 * reg * W2    # derivative of the reg_loss
        dW1 += 2 * reg * W1
        
        #-----------------------------------------------------------
        # Store the Gradients
        #-----------------------------------------------------------
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        return loss, grads
    
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        
        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        weight_update_history_W1 = []    # Plotting the weight / update ratios
        weight_update_history_W2 = []
        weight_update_history_W3 = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None
            
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            random_indices = np.random.choice(num_train, batch_size)
            X_batch = X[random_indices]
            y_batch = y[random_indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            
            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            #-----------------------------------------------------------
            # Perform the parameter updates
            #-----------------------------------------------------------
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']
            self.params['W3'] += -learning_rate * grads['W3']
            self.params['b3'] += -learning_rate * grads['b3']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            
            #########################################################################
            # TIMO: Ratio of weights:updates
            #########################################################################
            param_scale = np.linalg.norm(grads['W1'].ravel())
            update = -learning_rate * grads['W1']          # simple SGD update for W1
            update_scale = np.linalg.norm(update.ravel())
            ratio = update_scale / param_scale             # want ~1e-3
            weight_update_history_W1.append(ratio)
            
            param_scale = np.linalg.norm(grads['W2'].ravel())
            update = -learning_rate * grads['W2']          # simple SGD update for W2
            update_scale = np.linalg.norm(update.ravel())
            ratio = update_scale / param_scale             # want ~1e-3
            weight_update_history_W2.append(ratio)

            param_scale = np.linalg.norm(grads['W3'].ravel())
            update = -learning_rate * grads['W3']          # simple SGD update for W2
            update_scale = np.linalg.norm(update.ravel())
            ratio = update_scale / param_scale             # want ~1e-3
            weight_update_history_W3.append(ratio)
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                
            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
                # Decay learning rate
                learning_rate *= learning_rate_decay
                
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            'weight_update_history_W1': weight_update_history_W1,
            'weight_update_history_W2': weight_update_history_W2,
            'weight_update_history_W3': weight_update_history_W3,
            'learning_rate': learning_rate,
            'reg': reg
        }
    
    def predict(self, X):
        """
        Use the trained weights of this three-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None
        
        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        hidden_layer_1 = np.maximum(0, np.dot(X, self.params['W1']) + self.params['b1'])
        hidden_layer_2 = np.maximum(0, np.dot(hidden_layer_1, self.params['W2']) + self.params['b2'])
        scores = np.dot(hidden_layer_2, self.params['W3']) + self.params['b3']
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################
        
        return y_pred

    def predict_softmax(self, X, y):
        """
        Use the trained weights of this three-layer network to predict Softmax 
        probabilities and the Log Loss for data points. The Log Loss implementation
        works for two classes only (binary classifier).

        Returns:
        - probs: A numpy array of shape (N,) giving the Softmax probability of the 
          predicted classes in y_pred.
        - logloss_bin: Scalar value of the Logarithmic Loss of the predicted 
          Softmax probabilities.
        """
        probs = None

        hidden_layer_1 = np.maximum(0, np.dot(X, self.params['W1']) + self.params['b1'])
        hidden_layer_2 = np.maximum(0, np.dot(hidden_layer_1, self.params['W2']) + self.params['b2'])
        scores = np.dot(hidden_layer_2, self.params['W3']) + self.params['b3']
        y_pred = np.argmax(scores, axis=1)

        # Softmax probabilities for y_pred
        scores -= np.max(scores)
        #theta = 1.0
        #scores *= float(theta)
        exp_scores = np.exp(scores)
        sm_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax output  [N x K]
        #print(sm_probs[:20,:])
        probs = sm_probs[range(X.shape[0]), y_pred]

        # Log Loss for a binary classifier
        correct_class = y_pred == y
        logloss_bin = -np.sum((correct_class * np.log(probs)) + \
            ((1 - correct_class) * np.log(1 - probs))) / X.shape[0]

        return probs, logloss_bin

    def predict_softmax_p(self, X):
        """
        Use the trained weights of this three-layer network to predict Softmax 
        probabilities.

        Returns:
        - probs: A numpy array of shape (N,) giving the Softmax probability of the 
          predicted classes in y_pred.
        """
        probs = None
        
        hidden_layer_1 = np.maximum(0, np.dot(X, self.params['W1']) + self.params['b1'])
        hidden_layer_2 = np.maximum(0, np.dot(hidden_layer_1, self.params['W2']) + self.params['b2'])
        scores = np.dot(hidden_layer_2, self.params['W3']) + self.params['b3']
        y_pred = np.argmax(scores, axis=1)

        scores -= np.max(scores)
        #theta = 1.0
        #scores *= float(theta)
        exp_scores = np.exp(scores)
        sm_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax output  [N x K]
        probs = sm_probs[range(X.shape[0]), y_pred]

        return probs
