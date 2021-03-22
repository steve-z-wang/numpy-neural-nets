import numpy as np

def initialize_parameters(layer_dimension):
    parameters = {}

    L = len(layer_dimension)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dimension[l], layer_dimension[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dimension[l], 1))

    return parameters

