import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import time
import pickle
import random
outfile = TemporaryFile()
#RL agent learns policy for maximum accumulated reward
#master changes the game each player epoch
#R is converted from matrix to vector
#fixed bug that the game didn't really update afte the master acted
#all parameters save in file
#sparse R matrix

class MDP:

    def __init__(self, nodes, edgePerNode, Nsparse):

        np.random.seed(0)
        self.nodes = nodes
        self.edgePerNode= edgePerNode
        self.Nsparse = Nsparse
        self.W = np.exp(10.0 * np.random.rand(nodes, edgePerNode, nodes))
        self.W = self.normalize(self.W, nodes, edgePerNode)
        #Rvector = np.random.randint(edgePerNode, size=nodes)
        Rvector = random.sample(range(nodes*edgePerNode), Nsparse)
        print Rvector
        self.R = np.zeros((nodes, edgePerNode))
        print self.R
        Rindex = self.R.flatten()
        for i in range(0, Nsparse):
            Rindex[Rvector[i]] = 1
        self.R = Rindex.reshape((nodes, edgePerNode))
        print self.R

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
        self.R = np.zeros((self.nodes, self.edgePerNode))
        print self.R
        Rindex = self.R.flatten()
        for i in range(0, self.Nsparse):
            Rindex[Rvector[i]] = 1
        self.R = Rindex.reshape((self.nodes, self.edgePerNode))
        print self.R


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


class Master:

    def __init__(self, Nstates=10, Nactions=10, eps=0.2, gamma=0.95, Nepisodes=50, MaxEpiSteps=10, MinAlpha=0.01 ):
        self.Nstates = Nstates # number of possible R, 2^nodes
        self.Nactions = Nactions # the number of nodes
        self.Q = np.zeros((self.Nstates,self.Nactions))
        self.eps = eps
        self.gamma = gamma
        self.Nepisodes = Nepisodes
        self.MaxEpiSteps = MaxEpiSteps
        self.MinAlpha = MinAlpha
        self.alphas = np.linspace(1.0, self.MinAlpha, self.Nepisodes)

    def computeReward(self, logTDe):
        avg_length = 30
        TDepoch = np.average(logTDe[0:avg_length - 1]) - np.average(logTDe[-avg_length:])
        return TDepoch

    def reset(self):
        self.Q = np.zeros((self.Nstates, self.Nactions))

    def chooseAction(self, state_int):
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



###parameters###
##world##
W_nodes = 10
W_edgePerNode = 2
W_Nsparse = 2

##player##
P_Nstates = W_nodes
P_Nactions = W_edgePerNode
P_eps = 0.2
P_gamma = 0.95
P_Nepisodes = 100
P_MaxEpiSteps = 20
P_MinAlpha = 0.01

##game master##

M_Nstates = np.power(W_edgePerNode,W_nodes)
M_Nactions = W_nodes
M_eps = 0.2
M_gamma = 0.95
M_Nepisodes = 5
M_MaxEpiSteps = 10
M_MinAlpha = 0.01


game = MDP(W_nodes, W_edgePerNode, W_Nsparse)
player = Player(W_nodes, W_edgePerNode, P_eps, P_gamma, P_Nepisodes, P_MaxEpiSteps, P_MinAlpha)
master = Master(M_Nstates, M_Nactions, M_eps, M_gamma, M_Nepisodes, M_MaxEpiSteps, M_MinAlpha)

logR = []
logTDe = []
logBestScore = []
for i in range(0, W_nodes*W_edgePerNode):
    for j in range(0, W_nodes*W_edgePerNode):
        if i!=j and i>j:
            game.R = game.createR([i, j])
            player.reset()
            PlogR, PlogTDe = player.runEpoch(game.W, game.R)
            bestScore = np.sum(PlogR[-1])
            logR.append(PlogR)
            logTDe.append(PlogTDe)
            logBestScore.append(bestScore)

plt.hist(logBestScore, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
















