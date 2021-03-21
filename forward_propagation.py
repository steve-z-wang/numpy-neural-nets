import numpy as np
from activation_functions import sigmoid, relu 

def linear_forward(A, W, b):

    Z = np.dot(W,A)+b

    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".       
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)      
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".      
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache

def n_model_forward(X, parameters):
    caches = []
    A = X

    # number of layers in the neural network
    L = len(parameters) // 2
    
    # Using a for loop to replicate [LINEAR->RELU] (L-1) times
    for l in range(1, L):
        A_prev = A 

        # Implementation of LINEAR -> RELU.
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")

        # Adding "cache" to the "caches" list.
        caches.append(cache)

    
    # Implementation of LINEAR -> SIGMOID.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    
    # Adding "cache" to the "caches" list.
    caches.append(cache)
            
    return AL, caches