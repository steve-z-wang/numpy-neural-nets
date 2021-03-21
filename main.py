#%%
import numpy as np 
import matplotlib.pyplot as plt
from keras.datasets import mnist 

from n_layer_model import n_layer_model, predict

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

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plt.scatter(X[0, :], X[1, :], c=y[0], cmap=plt.cm.Spectral)
    plt.show()

#%%
train_x, train_y, test_x, test_y = load_dataset()

layers_dims = [784, 20, 10]

parameters = n_layer_model(train_x, train_y, layers_dims, learning_rate = 0.2, num_iterations = 300, print_cost = True)

print("train accuracy: {} %".format(100 - np.mean(np.abs(predict(train_x, parameters) - train_y)) * 100))
plot_decision_boundary(lambda x: predict(x.T, parameters), train_x, train_y)

#%%

from forward_propagation import n_model_forward

def predict(x, parameters):

    # Forward propagation
    probas, caches = n_model_forward(x, parameters)

    return probas