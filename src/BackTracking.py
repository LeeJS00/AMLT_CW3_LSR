import numpy as np

from Loss import LeastSquareLoss


def BackTracking(X:np.ndarray, y:np.ndarray, W:np.ndarray, grad:np.ndarray, direction:np.ndarray, tau:float = 0.1, c1:float = 0.1):
    t = 1.0
    # print(LeastSquareLoss(W + t * direction, X, y))
    # print(LeastSquareLoss(W, X, y) - c1 * t * np.sum((grad.T @ direction)**2))
        
    while (LeastSquareLoss(W + t * direction, X, y) > LeastSquareLoss(W, X, y) + c1 * t * grad.T@direction).all():
        # print(LeastSquareLoss(W + t * direction, X, y))
        # print(LeastSquareLoss(W, X, y) - c1 * t * np.sum((grad.T @ direction)**2))
        t = tau * t

    return t