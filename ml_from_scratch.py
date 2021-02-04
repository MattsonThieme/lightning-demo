
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

############################################
############################################

# Load data
boston = load_boston()
data, target = boston.data, boston.target

# Split data
# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=24)

# Fit model
'''
lm = LinearRegression(fit_intercept=True)
lm.fit(x_train, y_train)

y_hat = lm.predict(x_train)
plt.scatter(y_hat, y_train)
plt.plot([0,55],[0,55])

y_hat = lm.predict(x_test)
plt.scatter(y_hat, y_test)
plt.plot([0,55],[0,55])
'''
############################################
############################################

# Data preparation

split_ratio = 0.7
cut_point = int(boston.data.shape[0]*split_ratio)

def shuffle(x, y):
    perm = np.random.permutation(x.shape[0])
    return x[perm], y[perm]

def split(X, y, split_ratio):
    X, y = shuffle(X, y)
    cut_point = int(X.shape[0] * split_ratio)
    X_train, X_test = X[:cut_point], X[cut_point:]
    y_train, y_test = y[:cut_point].reshape(-1,1), y[cut_point:].reshape(-1,1)
    return X_train, X_test, y_train, y_test

x_train, x_test, y_train, y_test = split(boston.data, boston.target, 0.7)



# Utilities
# We want to use types to give people an idea what you expect in and out of a function
from typing import Dict, Tuple, Callable, List

# Initialization function

def initialize(ndim: int) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    weights['W'] = np.random.randn(ndim, 1)
    weights['B'] = np.random.randn(1, 1)
    return weights

# Batch Generator

Batch = Tuple[np.ndarray, np.ndarray]
def batch_generator(X: np.ndarray,
                    y: np.ndarray,
                    start: int=0,
                    batch_size: int=3) -> Batch:

    end = min(start + batch_size, X.shape[0])
    return X[start:end], y[start:end]

# Neural Network Models

def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.max(0.2 * x, x)

def identity(x: np.ndarray) -> np.ndarray:
    return x

# Function derivative
Func = Callable[[np.ndarray], np.ndarray]

# Super cool trick to get the derivative of some function
def deriv(func: Func,
          input_: np.ndarray) -> np.ndarray:
    delta = 0.001
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


# Implement forward function
Cache = Dict[str, np.ndarray]
def forward(X_batch: np.ndarray,
            y_batch: np.ndarray,
            weights: Cache) -> Tuple[float, Cache]:
    cache: Cache = {}
    cache['X'] = X_batch
    cache['Z'] = cache['X'] @ weights ['W'] + weights['B']
    cache['A'] = identity(cache['Z'])
    cache['Y'] = y_batch
    cache['E'] = cache['Y'] - cache['A']
    cache['Loss'] = np.mean(np.power(cache['E'], 2))
    return cache

# Implement backward function

def backward(cache: Cache,
             func: Func,
             weights: Dict[str, np.ndarray]) -> Cache:

    m = cache['E'].shape[0]
    dJ_dE = 2 / m * cache['E'].T
    dE_dA = -1 * np.eye(3)
    dJ_dA = dJ_dE @ dE_dA
    dA_dZ = np.diag(deriv(identity, cache['Z'].T)[0])
    dJ_dZ = dJ_dA @ dA_dZ
    dZ_dB = np.ones((m,1))
    dJ_dB = dJ_dZ @ dZ_dB
    dZ_dW = cache['X']
    dJ_dW = dJ_dZ @ dZ_dW

    grad: Cache = {}
    grad['W'] = dJ_dW.T
    grad['B'] = dJ_dB.T

    return grad

X_batch, y_batch = batch_generator(x_train, y_train)
weights = initialize(X_batch.shape[1])
cache = forward(X_batch, y_batch, weights)
back = backward(cache, identity, weights)

# Implement training module

def train(X_train: np.ndarray,
          y_train: np.ndarray,
          iter_nums: int=100,
          batch_size: int=3,
          learning_rate: float=0.001) -> Cache:
    np.random.seed(2)
    weights = initialize(X_train.shape[1])
    start = 0
    result = {}
    X, y = X_train, y_train
    for i in range(iter_nums):
        if start >= X_train.shape[0]:
            X, y = shuffle(X_train, y_train)
            start = 0

        # Generate minibatch
        X_batch, y_batch = batch_generator(X, y, start, batch_size)
        start += batch_size

        cache = forward(X_batch, y_batch, weights)
        grad = backward(cache, identity, weights)

        # Gradient update
        for key in weights.keys():
            weights[key] -= learning_rate * grad[key]

        result[str(i)] = forward(X_train, y_train, weights)['Loss']
    return result

x = train(X_batch, y_batch)

print(" ")