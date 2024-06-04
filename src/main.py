import numpy as np
import time


from Loss import LeastSquareLoss

from readData import readData
from Analytic import Analytic
from SteepestDescent import SteepestDescent
from Newton import Newton
from GaussNewton import GaussNewton

if __name__ == '__main__':
    np.random.seed(42)

    X_train, y_train = readData('../ICVLResnet101FeaturesandLabels/Trn.csv')
    X_test, target = readData('../ICVLResnet101FeaturesandLabels/Tst.csv')
    # print(X_train.shape, y_train.shape)
    l, n, m = X_train.shape[1], X_train.shape[0], y_train.shape[1]
    max_iter = 1000
    W_init = np.zeros([n, m])

    mode = 'Analytic'
    start = time.time()
    W, losses = Analytic(X_train, y_train, W_init, 1e-3, max_iter)    
    runtime = time.time() - start
    loss = LeastSquareLoss(W, X_test, target)
    print(f'{mode} Loss : {loss:3f}')
    print(f'{mode} Runtime: ', runtime)

    mode = 'Steepest Descent W_init zeros'
    W_init = np.zeros([n, m])
    start = time.time()
    W, losses = SteepestDescent(X_train, y_train, W_init)
    runtime = time.time() - start
    loss = LeastSquareLoss(W, X_test, target)
    print(f'{mode} Loss : {loss:3f}')
    print(f'{mode} Runtime: ', runtime)
    
    mode = 'Steepest Descent W_init ones'
    W_init = np.ones([n, m])
    start = time.time()
    W, losses = SteepestDescent(X_train, y_train, W_init)
    runtime = time.time() - start
    loss = LeastSquareLoss(W, X_test, target)
    print(f'{mode} Loss : {loss:3f}')
    print(f'{mode} Runtime: ', runtime)
    
    mode = 'Newton W_init zeros'
    W_init = np.zeros([n, m])
    start = time.time()
    W, losses = Newton(X_train, y_train, W_init)
    runtime = time.time() - start
    loss = LeastSquareLoss(W, X_test, target)
    print(f'{mode} Loss : {loss:3f}')
    print(f'{mode} Runtime: ', runtime)
    
    mode = 'Newton W_init ones'
    W_init = np.ones([n, m])
    start = time.time()
    W, losses = Newton(X_train, y_train, W_init)
    runtime = time.time() - start
    loss = LeastSquareLoss(W, X_test, target)
    print(f'{mode} Loss : {loss:3f}')
    print(f'{mode} Runtime: ', runtime)

    mode = 'Gauss Newton W_init zeros'
    W_init = np.zeros([n, m])
    start = time.time()
    W, losses = GaussNewton(X_train, y_train, W_init)
    runtime = time.time() - start
    loss = LeastSquareLoss(W, X_test, target)
    print(f'{mode} Loss : {loss:3f}')
    print(f'{mode} Runtime: ', runtime)
    
    mode = 'Gauss Newton W_init ones'
    W_init = np.ones([n, m])
    start = time.time()
    W, losses = GaussNewton(X_train, y_train, W_init)
    runtime = time.time() - start
    loss = LeastSquareLoss(W, X_test, target)
    print(f'{mode} Loss : {loss:3f}')
    print(f'{mode} Runtime: ', runtime)