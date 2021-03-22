import pickle 
import numpy as np
import matplotlib.pyplot as plt

from forward_propagation import n_model_forward
from backward_propagation import n_model_backward

def compute_cost(AL, Y):
    # number of examples
    m = Y.shape[1]

    # Compute loss from AL and y.
    cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m

    # To make sure our cost's shape is what we expect (e.g. this turns [[23]] into 23).
    cost = np.squeeze(cost)
    
    return cost

def update_parameters(parameters, grads, learning_rate):
    # number of layers in the neural network
    L = len(parameters) // 2 

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

    return parameters

def train(X, Y, parameters, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    # keep track of cost
    costs = []

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = n_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = n_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Every 100 iterations: 
        if print_cost and i % 10 == 0:

            # print cost value 
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

            # save parameters
            pickle.dump(parameters, open('parameters.pkl', 'wb'))            

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, parameters):
    m = X.shape[1]

    # number of layers in the neural network
    n = len(parameters) // 2
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = n_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0   
        
    return p

