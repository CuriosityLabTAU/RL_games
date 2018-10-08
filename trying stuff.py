import numpy as np
import pickle
import time
import random
timestr = time.strftime("Log-%H%M-%d-%m-%Y")
#print timestr


dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
#print dict

filename = timestr
with open(filename, 'wb') as f:
    pickle.dump(dict, f)

#with open(filename, 'rb') as f:
#    dict1 = pickle.load(f)
#print dict1

a = np.random.randint(5, size=(2, 4))
print a
b = a.flatten()
print b
c = b.reshape((2,4))
print c

Rvector = random.sample(range(20), 2)
print Rvector





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

#print(np.random.randint(5, size=(2, 4, 2)))