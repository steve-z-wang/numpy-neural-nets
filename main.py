#%%
import os
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from keras.datasets import mnist 

from initialization import initialize_parameters
from n_layer_model import train, predict

#%% 
def load_dataset(): 
    (train_x_orig, train_y_orig), (test_x_orig, test_y_orig) = mnist.load_data()

    # reshape training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # standardize data to values between 0 and 1. 
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255. 

    train_y = np.zeros((10, train_y_orig.shape[0]))
    for count, y in enumerate(train_y_orig):
        train_y[y][count] = 1
    
    test_y = np.zeros((10, test_y_orig.shape[0]))
    for count, y in enumerate(test_y_orig):
        test_y[y][count] = 1

    return train_x, train_y, test_x, test_y

#%%
def main(): 

    # load training data 
    train_x, train_y, test_x, test_y = load_dataset()

    # initialize model 
    f = "parameters.pkl" 
    if os.path.exists(f): 
        parameters = pickle.load(open(f, 'rb'))
    else: 
        parameters = initialize_parameters(layer_dimension=[784, 20, 10])

    # train the model
    train(train_x, train_y, parameters, learning_rate = 0.2, num_iterations = 50, print_cost = True)

    #
    print("train accuracy: {} %".format(100 - np.mean(np.abs(predict(train_x, parameters) - train_y)) * 100))

    # from forward_propagation import n_model_forward

    # def predict(x, parameters):

    #     # Forward propagation
    #     probas, caches = n_model_forward(x, parameters)

    #     return probas

main()

# %%
