import numpy as np
import tensorflow tf

import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)


def main():
    X,Y = get_normalized_data()

    max_iter = 20
    print_period = 10

    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:,]
    Ytest  = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N / batch_sz

    M1 = 300
    M2 = 100
    K = 10
    W1_init = np.random.randn(D, M1) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(K)
    W3_init = np.random.randn(M2, k) / np.sqrt(M2)
    b3_init = np.zeros(K)
