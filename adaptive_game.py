import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import time
import datetime
import pickle
import random
import math
import itertools

class MDP:

    def __init__(self, nodes, edgePerNode, Nsparse):

        #np.random.seed(0)
        self.nodes = nodes
        self.edgePerNode= edgePerNode
        self.Nsparse = Nsparse
        self.W = np.exp(10.0 * np.random.rand(nodes, edgePerNode, nodes))
        #plt.plot(self.W[0,1,:])
        #plt.show()
        self.W = self.normalize(self.W, nodes, edgePerNode)
        #print self.W
        #self.W = (self.W > 0.5)
        #print self.W
        Rvector = np.random.randint(edgePerNode, size=nodes)
        Rvector = random.sample(range(nodes*edgePerNode), Nsparse)
        self.R = np.zeros((nodes, edgePerNode))
        Rindex = self.R.flatten()
        for i in range(0, Nsparse):
            Rindex[Rvector[i]] = 1
        self.R = Rindex.reshape((nodes, edgePerNode))

    def normalize(self, X, nodes, edgePerNode):
        for i in range(0, nodes):
            for j in range(0, edgePerNode):
                probSum = np.sum(X[i, j, :])
                X[i, j, :] = X[i, j, :] / probSum
        return X

    def reset(self):
        #np.random.seed(0)
        #self.W = np.exp(10.0 * np.random.rand(nodes, edgePerNode, nodes))
        self.W = self.normalize(self.W, self.nodes, self.edgePerNode)
        Rvector = np.random.randint(self.edgePerNode, size=self.nodes)
        self.R = np.zeros((self.nodes, self.edgePerNode))
        for i in range(0, self.nodes):
            self.R[i, Rvector[i]] = 1

    def createR(self, Rvector):
        #self.R = np.zeros((self.nodes, self.edgePerNode))
        self.R = np.full((self.nodes, self.edgePerNode), 0.0)
        Rindex = self.R.flatten()
        #print Rvector
        for i in range(0, self.Nsparse):
            Rindex[Rvector[i]] += 1
        self.R = Rindex.reshape((self.nodes, self.edgePerNode))
        #for i in range(len(self.R[:,0])):
        #    if (1 in self.R[i,:]) == True:
        #        self.R[i,:] = 1
        #print self.R
        return self.R


class Player:

    def __init__(self, Nstates=10, Nactions=2, eps=0.2, gamma=0.95, Nepisodes=200, MaxEpiSteps=10, MinAlpha=0.01 ):
        self.Nstates = Nstates
        self.Nactions = Nactions
        self.Q = np.zeros((self.Nstates,self.Nactions))
        self.eps = eps
        self.gamma = gamma
        self.Nepisodes = Nepisodes
        self.MaxEpiSteps = MaxEpiSteps
        self.MinAlpha = MinAlpha
        self.alphas = np.linspace(1.0, self.MinAlpha, self.Nepisodes)

    def reset(self):
        self.Q = np.zeros((self.Nstates, self.Nactions))

    def chooseAction(self, state):
        if np.random.uniform(0, 1) < self.eps:
            return np.random.choice(np.arange(self.Nactions))
        else:
            return np.argmax(self.Q[state,:])

    def act(self, state, action, W, R):
        nextState = np.random.choice(np.arange(self.Nstates), p=W[state, action,:])
        #print ['state = ', state]
        #print ['action = ', action]
        reward = R[state, action]
        return nextState, reward

    def runStep(self, state, alpha, W, R):
        action = self.chooseAction(state)
        nextState, reward = self.act(state, action, W, R)
        TDe = reward + self.gamma*np.max(self.Q[nextState,:])-self.Q[state, action]
        self.Q[state, action] = self.Q[state, action]+alpha*(reward + self.gamma*np.max(self.Q[nextState,:])-self.Q[state, action])
        return TDe, reward, nextState

    def runEpisode(self, episode, W, R):
        alpha = self.alphas[episode]
        state = 0
        totalReward = 0
        totalTDe = 0
        for step in range(0, self.MaxEpiSteps):
            #action = self.chooseAction(state)
            TDe, reward, nextState = self.runStep(state, alpha, W, R)
            state = nextState
            totalReward += reward
            totalTDe += np.abs(TDe)
        return totalReward, totalTDe

    def runEpoch(self, W, R):
        logR = []
        logTDe = []
        for episode in range(0,self.Nepisodes):
            totalReward, totalTDe = self.runEpisode(episode, W, R)
            logR.append(totalReward)
            logTDe.append(totalTDe)
        return logR, logTDe


class Master:

    def __init__(self, Nstates=10, Nactions=10, eps=0.2, gamma=0.95, Nepisodes=50, MaxEpiSteps=10, MinAlpha=0.01 ):
        self.Nstates = Nstates  # number of possible R, 2^nodes
        self.Nactions = Nactions # the number of nodes
        self.Q = np.zeros((self.Nstates,self.Nstates*self.Nactions,self.Nstates*self.Nactions ))
        self.eps = eps
        self.gamma = gamma
        self.Nepisodes = Nepisodes
        self.MaxEpiSteps = MaxEpiSteps
        self.MinAlpha = MinAlpha
        self.alphas = np.linspace(1.0, self.MinAlpha, self.Nepisodes)

    def translateR(self, R, Nsparse):
        #take the R matrix from the game and translates in to a Nsparse index vector
        indexVector = np.zeros(Nsparse)
        Rflat = R.flatten()
        j = 0
        for i in range(len(Rflat)):
            if Rflat[i]==1:
                indexVector[j] = i
                j += 1
        return indexVector

    def computeReward(self, logTDe):
        avg_length = 10
        TDepoch = np.average(logTDe[0:avg_length - 1]) - np.average(logTDe[-avg_length:])
        return TDepoch

    def reset(self):
        self.Q = np.zeros((self.Nstates, self.Nactions))

    def chooseAction(self, R):
        indexVector = self.translateR(R)
        if np.random.uniform(0, 1) < self.eps:
            return np.random.choice(np.arange(self.Nactions))
        else:
            return np.argmax(self.Q[state_int,:])


    def act(self, state, action, W, R):
        nextState = state
        temp = nextState[action, 0]
        nextState[action, 0] = nextState[action, 1]
        nextState[action, 1] = temp
        return nextState

    def actSparse(self, state, action, W, R):
        nextState = state


    def runStep(self, state, alpha, W, R):
        action = self.chooseAction(state)
        nextState = self.act(state, action, W, R)
        TDe = reward + self.gamma*np.max(self.Q[nextState,:])-self.Q[state, action]
        self.Q[state, action] = self.Q[state, action]+alpha*(reward + self.gamma*np.max(self.Q[nextState,:])-self.Q[state, action])
        return TDe, reward, nextState

    def runEpisode(self, episode, W, R):
        alpha = self.alphas[episode]
        state = 0
        totalReward = 0
        totalTDe = 0
        for step in range(0, self.MaxEpiSteps):
            #action = self.chooseAction(state)
            TDe, reward, nextState = self.runStep(state, alpha, W, R)
            state = nextState
            totalReward += reward
            totalTDe += TDe
        return totalReward, totalTDe

    def runEpoch(self, W, R):
        logR = []
        logTDe = []
        for episode in range(0,self.Nepisodes):
            totalReward, totalTDe = self.runEpisode(episode, W, R)
            logR.append(totalReward)
            logTDe.append(totalTDe)
        return logR, logTDe

    def arr2int(self, X):
        b = 0
        for i in range(len(X)):
            b += X[i, 0] * np.power(2, i)
        return int(b)

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def sparseVector(index, Nsparse, nodes, edgePerNode):
    Ngames = np.power(nodes * edgePerNode, Nsparse)
    sparseMat = np.zeros((Ngames ,Nsparse))

def averageLogs(log):
    nlog = np.asarray(log)
    averaged = sum(nlog)/len(nlog)