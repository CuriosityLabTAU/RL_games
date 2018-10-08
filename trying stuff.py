import numpy as np


def arr2int(X):
    b = 0
    for i in range(len(X)):
        b += X[i, 0] * np.power(2, i)
    return b

nodes= 10
edgePerNode = 2
Rvector = np.random.randint(edgePerNode, size=nodes)
R = np.zeros((nodes, edgePerNode))
for i in range(0, nodes):
    R[i, Rvector[i]] = 1
#print R

B = arr2int(R)
#print B

np.random.seed(0)

Rvector = np.random.randint(10, size=2)
#print Rvector
np.random.seed(0)

Rvector = np.random.randint(10, size=2)
#print Rvector

print(np.random.randint(5, size=(2, 4, 2)))