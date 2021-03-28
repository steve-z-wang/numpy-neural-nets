import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(layer_dims): 
    parameters = {}
    for l in range(1, len(layer_dims)): 
        bound = np.sqrt(2/layer_dims[l-1])
        parameters['W' + str(l)] = np.random.uniform(-bound, bound, (layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b): 
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def relu_activation(Z): 
    return np.maximum(0, Z)

def softmax_activation(Z):
    exp = np.exp(Z)
    return exp / exp.sum(axis=0)

def forward_propagation(X, parameters): 
    """ (LINEAR -> RELU) * (L-1) -> LINEAR -> SOFTMAX """ 

    L = len(parameters) // 2
    caches = [] 
    A = X 
    for l in range(1, L): 
        A_prev = A
        Z, linear_cache = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        A = relu_activation(Z)
        cache = (linear_cache, Z)
        caches.append(cache)

    ZL, linear_cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
    AL = softmax_activation(ZL)
    cache = (linear_cache, ZL)
    caches.append(cache)

    return AL, caches

def compute_cost(AL, y): 
    y_hat = np.clip(AL, 0.00001, 0.99999)
    loss = - np.sum(y * np.log(y_hat), axis=0)
    cost = np.squeeze(np.mean(loss))
    return cost

def relu_activation_backward(dA, Z): 
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    return dZ

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches): 

    grads = {}
    L = len(caches) 

    dZL = AL - Y
    linear_cache, ZL = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZL, linear_cache)

    for l in reversed(range(1, L)):
        linear_cache, Z = caches[l-1]
        dZ = relu_activation_backward(grads["dA" + str(l)], Z)
        grads["dA" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_backward(dZ, linear_cache)

    return grads

def update_parameters(parameters, grads, learning_rate): 
    L = len(parameters) // 2
    
    for l in range(1, L+1): 
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    
    return parameters

def L_layer_model(X, Y, layers_dimensions, num_iterations, learning_rate=0.1, print_cost=False): 

    costs = []

    # initialize model
    parameters = initialize_parameters(layers_dimensions)

    for i in range(0, num_iterations):
        # forward propagation
        AL, caches = forward_propagation(X, parameters)

        # compute cost
        cost = compute_cost(AL, Y)

        # backward propatation
        grads = backward_propagation(AL, Y, caches)

        # updata parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # print and save cost value
        if print_cost and i % 100 == 0: 
            print ("Cost after iteration %i: %f" %(i, cost))
        costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def predict(X, parameters): 
    AL, caches = forward_propagation(X, parameters)
    return np.argmax(AL, axis=0)