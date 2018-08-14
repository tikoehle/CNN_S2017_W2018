
# coding: utf-8

# # Implementing a Neural Network
# In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.

# In[57]:


# A bit of setup

import numpy as np
import matplotlib.pyplot as plt
import time

from cs231n.classifiers.neural_net import TwoLayerNet

from __future__ import print_function

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (12.0, 8.0)        # Set default size of plots. Needed to load this cell twice.
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# We will use the class `TwoLayerNet` in the file `cs231n/classifiers/neural_net.py` to represent instances of our network. The network parameters are stored in the instance variable `self.params` where keys are string parameter names and values are numpy arrays. Below, we initialize toy data and a toy model that we will use to develop your implementation.

# In[58]:


# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4       # dimensionality D
hidden_size = 10     # hidden layer number of neurons H
num_classes = 3      # output_size number of classes C
num_inputs = 5       # number of training records N
#
# input (4) - fully connected hidden layer (10) - ReLU - fully connected output layer (3) - softmax
#

def init_toy_model():
    np.random.seed(0)    # np.random.seed() makes the random numbers predictable. Same numbers are generated.
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)   # X: Input data of shape (N, D)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()


# # Forward pass: compute scores
# Open the file `cs231n/classifiers/neural_net.py` and look at the method `TwoLayerNet.loss`. This function is very similar to the loss functions you have written for the SVM and Softmax exercises: It takes the data and weights and computes the class scores, the loss, and the gradients on the parameters. 
# 
# Implement the first part of the forward pass which uses the weights and biases to compute the scores for all inputs.

# In[59]:


scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. We get < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))


# # Forward pass: compute loss
# In the same function, implement the second part that computes the data and regularizaion loss.

# In[60]:


loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133
print('loss %f' % (loss))

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))


# # Backward pass
# Implement the rest of the function. This will compute the gradient of the loss with respect to the variables `W1`, `b1`, `W2`, and `b2`. Now that you (hopefully!) have a correctly implemented forward pass, you can debug your backward pass using a numeric gradient check:

# In[61]:


from cs231n.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.05)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


# # Train the network
# To train the network we will use stochastic gradient descent (SGD), similar to the SVM and Softmax classifiers. Look at the function `TwoLayerNet.train` and fill in the missing sections to implement the training procedure. This should be very similar to the training procedure you used for the SVM and Softmax classifiers. You will also have to implement `TwoLayerNet.predict`, as the training process periodically performs prediction to keep track of accuracy over time while the network trains.
# 
# Once you have implemented the method, run the code below to train a two-layer network on toy data. You should achieve a training loss less than 0.2.

# In[62]:


net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()


# # Load the data
# Now that you have implemented a two-layer network that passes gradient checks and works on toy data, it's time to load up our favorite CIFAR-10 data so we can use it to train a classifier on a real dataset.

# In[63]:


from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# # Train a network
# To train our network we will use SGD with momentum. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.

# In[64]:


input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

t1 = time.time()

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

t2 = time.time()

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: %f, computed in %fs' % (val_acc, t2 - t1))


# # Debug the training
# With the default parameters we provided above, you should get a validation accuracy of about 0.29 on the validation set. This isn't very good.
# 
# One strategy for getting insight into what's wrong is to plot the loss function and the accuracies on the training and validation sets during optimization.
# 
# Another strategy is to visualize the weights that were learned in the first layer of the network. In most neural networks trained on visual data, the first layer weights typically show some visible structure when visualized.

# In[65]:


# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'], label='train')
plt.legend()
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.legend()
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.tight_layout()                 # control extra padding around the figure border to avoid overlapping labels
plt.show()


# ## Debug the Ratio of weights:updates
# The last quantity you might want to track is the ratio of the update magnitudes to the value magnitudes. Note: updates, not the raw gradients (e.g. in vanilla sgd this would be the gradient multiplied by the learning rate). You might want to evaluate and track this ratio for every set of parameters independently. A rough heuristic is that this ratio should be somewhere around 1e-3. If it is lower than this then the learning rate might be too low. If it is higher then the learning rate is likely too high.
# ### NOTE: The step function is because of Decay learning rate
#             learning_rate *= learning_rate_decay

# In[66]:


plt.subplot(2, 1, 2)
plt.plot(stats['weight_update_history_W1'], label='W1 weights:updates')
plt.plot(stats['weight_update_history_W2'], label='W2 weights:updates')
plt.legend()
plt.title('Ratio of weights:updates history')
plt.xlabel('Epoch')
plt.ylabel('update_scale / param_scale')
plt.show()


# In[67]:


from cs231n.vis_utils import visualize_grid

# Visualize the weights of the network

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)


# # Tune your hyperparameters
# 
# **What's wrong?**. Looking at the visualizations above, we see that the loss is decreasing more or less linearly, which seems to suggest that the learning rate may be too low. Moreover, there is no gap between the training and validation accuracy, suggesting that the model we used has low capacity, and that we should increase its size. On the other hand, with a very large model we would expect to see more overfitting, which would manifest itself as a very large gap between the training and validation accuracy.
# 
# **Tuning**. Tuning the hyperparameters and developing intuition for how they affect the final performance is a large part of using Neural Networks, so we want you to get a lot of practice. Below, you should experiment with different values of the various hyperparameters, including hidden layer size, learning rate, numer of training epochs, and regularization strength. You might also consider tuning the learning rate decay, but you should be able to get good performance using the default value.
# 
# **Approximate results**. You should be aim to achieve a classification accuracy of greater than 48% on the validation set. Our best network gets over 52% on the validation set.
# 
# **Experiment**: You goal in this exercise is to get as good of a result on CIFAR-10 as you can, with a fully-connected Neural Network. For every 1% above 52% on the Test set we will award you with one extra bonus point. Feel free implement your own techniques (e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).

# In[70]:


best_net = None     # store the best model into this 
best_stats = None   # training stats of the best model
best_val = -1       # The highest validation accuracy that we have seen so far.
results = {}        # History buffer for plotting. The other buffers are in stats returned from net.train().

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
input_size = 32 * 32 * 3
num_classes = 10
std=1e-6

# Hyperparameters
hidden_size = [100]
np.random.seed(2)
learning_rates = 10 ** np.random.uniform(-3.1, -2.7, 10)           # searching on log scale. The size is last par.
np.random.seed(3)
regularization_strengths = 10 ** np.random.uniform(-3, -1, 10)     # searching on log scale. The size is last par.
# dropout = np.random.uniform(0, 1, 10)                            # TODO. Usually searched in the original scale.
num_iters = 5000                                                   # Number of steps to take when optimizing.
batch_size = 400                                                   # Number of training examples to use per step.

# Calculating the number of epochs. 
# Staging the search from coarse (wide hyperparameter ranges, training only for 1-5 epochs), 
# to fine (narrower rangers, training for many more epochs).
num_train = X_train.shape[0]
iterations_per_epoch = max(num_train / batch_size, 1)
epochs = num_iters / iterations_per_epoch
print('Number of epochs:%.2f, iterations per epoch:%.2f' % (epochs, iterations_per_epoch))

# Prefer one validation fold to cross-validation
for h_size in hidden_size:
    for l_rate in learning_rates:
        for reg_strength in regularization_strengths:
            # Create the network
            net = TwoLayerNet(input_size, h_size, num_classes, std)
            # Train the network.
            t1 = time.time()
            stats = net.train(X_train, y_train, X_val, y_val,
                              num_iters=num_iters, batch_size=batch_size,
                              learning_rate=l_rate, learning_rate_decay=0.95,
                              reg=reg_strength, verbose=False)
            # Evaluate. Predict on the training set
            train_accuracy = (net.predict(X_train) == y_train).mean()
            # Predict on the validation set
            val_accuracy = (net.predict(X_val) == y_val).mean()
            if val_accuracy > best_val:
                t2 = time.time()
                best_val = val_accuracy
                best_net = net
                best_stats = stats
                print('hidden:%d batch:%d learn:%f reg:%f train_accuracy:%f val_accuracy:%f took:%.1fs' 
                      % (h_size, batch_size, l_rate, reg_strength, train_accuracy, val_accuracy, t2 - t1))
                
            results[(l_rate, reg_strength)] = (train_accuracy, val_accuracy)
                    
print('Best validation accuracy achieved during cross-validation: %f' % best_val)
#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################


# In[42]:


# Visualize the cross-validation results of the val_accuracy in results
import math
from matplotlib import cm

x_scatter = [math.log10(x[0]) for x in results]     # log scale; learning_rate
y_scatter = [math.log10(x[1]) for x in results]     # log scale; regularization_strength

# Plot validation accuracy
colors = [results[x][1] for x in results]           # results[x][1] is val_accuracy which determines the color
plt.subplot(2, 1, 2)
p = plt.scatter(x_scatter, y_scatter, c=colors, cmap=cm.coolwarm)

# Add a color bar which maps values to colors.
cbar = plt.colorbar(p, aspect=10)
cbar.set_label('validation accurracy')

# Labels
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 Validation Accuracy, Random Layout')
plt.show()


# In[48]:


# Visualize the cross-validation results in 3D. Note the other history buffers returned from train() in stats.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get the X, Y, Z data. 
X = [math.log10(x[0]) for x in results]     # log scale; learning_rate
Y = [math.log10(x[1]) for x in results]     # log scale; regularization_strength
Z = [results[x][1] for x in results]        # normal scale; val_accuracy

# Scatter-3D plot
p = ax.scatter(X, Y, Z, c=Z, s=50, cmap=cm.coolwarm)   # s is the dot size

# Add a color bar which maps values to colors.
cbar = plt.colorbar(p, shrink=0.65, aspect=10)
cbar.set_label('validation accurracy')

# Labels
ax.set_xlabel('log learning rate')
ax.set_ylabel('log regularization strength')
#ax.set_zlabel('validation accurracy')
ax.text2D(0.05, 0.95, "CIFAR-10 Cross-Validation, Random Cube Search Layout", fontsize=12, transform=ax.transAxes)
plt.show()


# In[44]:


# As a sanity check, make sure your initial loss is reasonable
print('Initial training loss (Softmax/CE expect: -ln(1/num_classes) = -ln(0.1) = 2.302): %f' 
      % (best_stats['loss_history'][0]))

print('Final training loss: %f' % (best_stats['loss_history'][-1]))

# plot the loss history
plt.plot(best_stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()


# In[45]:


# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(best_stats['loss_history'], label='train')
plt.legend()
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(best_stats['train_acc_history'], label='train')
plt.plot(best_stats['val_acc_history'], label='val')
plt.legend()
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.tight_layout()                 # control extra padding around the figure border to avoid overlapping labels
plt.show()


# In[46]:


# visualize the weights of the best network
show_net_weights(best_net)


# # Run on the test set
# When you are done experimenting, you should evaluate your final trained network on the test set; you should get above 48%.
# 
# **We will give you extra bonus point for every 1% of accuracy above 52%.**

# In[47]:


test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)

