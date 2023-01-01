import numpy as np

def identity_function(x):
    return x

def step_function(x):
    return np.array(x>0, dtype=np.int)

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x))*sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad