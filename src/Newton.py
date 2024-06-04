import numpy as np

from Loss import LeastSquareLoss
from BackTracking import BackTracking


def Newton(X:np.ndarray, y:np.ndarray, W_init:np.ndarray, threshold:float = 1e-6, max_epoch:int = 1000 ): 
    W = W_init
    losses = []
    for epoch in range(max_epoch):
        grad_f = 2 * X @ X.T @ W - 2 * X @ y
        if np.linalg.norm(grad_f) < threshold:
            loss = LeastSquareLoss(W, X, y)
            losses.append(loss)
            print(f"Final EPOCH : {epoch}\tLoss : {loss:.3f}")
            break

        Hessian_f = 2* X @ X.T + 1e-6 * np.eye(X.shape[0])
        
        direction = - np.linalg.inv(Hessian_f) @ grad_f
        alpha = BackTracking(X, y, W, grad_f, direction) 
        W = W + alpha * direction

        if epoch % 100 == 0:            
            loss = LeastSquareLoss(W, X, y)
            losses.append(loss)
            print(f"EPOCH : {epoch}\tLoss : {loss:.3f}")

    return W, losses

