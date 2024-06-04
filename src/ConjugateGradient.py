import numpy as np

from Loss import LeastSquareLoss
from BackTracking import BackTracking


def ConjugateGradient(X:np.ndarray, y:np.ndarray, W_init:np.ndarray, threshold:float = 1e-6, max_epoch:int = 1000 ): 
    W = W_init
    losses = []
    for epoch in range(max_epoch):
        residual = y - X.T @ W
        J = X.T
        grad_f = -2 * J.T @ residual
        if np.linalg.norm(grad_f) < threshold:
            loss = LeastSquareLoss(W, X, y)
            losses.append(loss)
            print(f"Final EPOCH : {epoch}\tLoss : {loss:.3f}")
            break

        Hessian_f = 2 * J.T @ J        
        direction = np.linalg.inv(Hessian_f) @ grad_f
        alpha = BackTracking(X, y, W, grad_f, direction) 
        W = W + alpha * direction

        if epoch % 100 == 0:            
            loss = LeastSquareLoss(W, X, y)
            losses.append(loss)
            print(f"EPOCH : {epoch}\tLoss : {loss:.3f}")

    return W, losses
