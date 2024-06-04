import numpy as np


def LeastSquareLoss(W, X, y):
    loss = np.sum((X.T @ W - y)**2)
    return loss