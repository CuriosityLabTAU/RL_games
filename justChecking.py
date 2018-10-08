import numpy as np
def normalize(X, Nstates, Nactions):
    for i in range(0,Nstates):
        for j in range(0,Nactions):
            probSum = np.sum(X[i, j, :])
            X[i, j, :] = X[i, j, :] / probSum
    return X

W = np.random.rand(3,2,3)
W = normalize(W, 3, 2)
nextState = np.random.choice([0,1,2], p=[0.1, 0.2,0.7])#W[0, 0,:])

