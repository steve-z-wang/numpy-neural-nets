import numpy as np

def sigmoid(Z):
    """
    Sigmoid activation function

    Arguments:
    Z - numpy array of any shape

    Returns:
    A - output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    """
    The backward propagation for Sigmoid function

    Arguments:
    dA - post-activation gradient, of any shape
    cache - 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ - Gradient of the cost with respect to Z
    """
    Z = cache 
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu(Z):
    """
    Relu activation function

    Arguments:
    Z - Output of the linear layer, of any shape

    Returns:
    A - Post-activation parameter, of the same shape as Z
    cache - Z, stored for computing the backward pass efficiently
    """
    A = np.maximum(0,Z)  
    cache = Z 
    return A, cache

def relu_backward(dA, cache):
    """
    The backward propagation for Relu function

    Arguments:
    dA - post-activation gradient, of any shape
    cache - 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ - Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    """ 
    Softmax activation function 

    Arugments: 
    Z - Output of the linear layer 

    Returns: 
    A - Post-activation parameter
    cache - Z, stored for computing the backward pass efficiently
    """ 

    expo = np.exp(Z)
    expo_sum = np.sum(expo)
    A = expo/expo_sum 
    cache = Z

    return A, cache


    