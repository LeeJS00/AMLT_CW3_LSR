import numpy as np

from Loss import LeastSquareLoss

def Analytic(X:np.ndarray, y:np.ndarray, W_init:np.ndarray, alpha:float = 1e-6, max_epoch:int = 1000 ): 
    W = W_init
    losses = []
    K = np.linalg.cholesky(X @ X.T + alpha * np.eye(X.shape[0]))
    W = np.linalg.solve(K, X @ y)

    loss = LeastSquareLoss(W, X, y)
    losses.append(loss)
    print(f"EPOCH : 0\tLoss : {loss:3f}")

    return W, losses
