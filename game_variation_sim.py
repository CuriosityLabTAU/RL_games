import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import time
import datetime
import pickle
import random
import math
import itertools
outfile = TemporaryFile()
#RL agent learns policy for maximum accumulated reward
#master changes the game each player epoch
#R is converted from matrix to vector
#fixed bug that the game didn't really update afte the master acted
#all parameters save in file
#sparse R matrix
#N sparse = N
#averaging over epochs

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
        avg_length = 10
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


###parameters###

#analysis parameters
Nepochs = 100

##world##
W_nodes = 10
W_edgePerNode = 2
W_Nsparse = 2
#
##player##
P_Nstates = W_nodes
P_Nactions = W_edgePerNode
P_eps = 0.1
P_gamma = 0.95
P_Nepisodes = 500
P_MaxEpiSteps = 30
P_MinAlpha = 0.01

##game master##

M_Nstates = np.power(W_edgePerNode,W_nodes)
M_Nactions = W_nodes
M_eps = 0.1
M_gamma = 0.95
M_Nepisodes = 5
M_MaxEpiSteps = 30
M_MinAlpha = 0.01


game = MDP(W_nodes, W_edgePerNode, W_Nsparse)
#print game.W
player = Player(W_nodes, W_edgePerNode, P_eps, P_gamma, P_Nepisodes, P_MaxEpiSteps, P_MinAlpha)
master = Master(M_Nstates, M_Nactions, M_eps, M_gamma, M_Nepisodes, M_MaxEpiSteps, M_MinAlpha)

logR = []
logTDe = []
logLastScore = []
logLastTDe = []
logTDdiff = []
logTotalTDe = []
logGame = []

tot_logR = []
tot_logTDe = []
tot_logLastScore = []
tot_logLastTDe = []
tot_logTDdiff = []
tot_logTotalTDe = []
tot_logGame = []

K = math.factorial(W_nodes*W_edgePerNode)/(math.factorial(W_Nsparse)*math.factorial(W_nodes*W_edgePerNode-W_Nsparse))
#gameDiff = np.zeros((K, W_Nsparse+1))
gameDiff = np.zeros((W_nodes*W_edgePerNode,W_nodes*W_edgePerNode))
accumuR = np.zeros((W_nodes, W_edgePerNode))
faccumuR = accumuR.flatten()

#create all the combinations of sparse vector indeces of length W_Nsparse
sparseList = list(itertools.product(range(W_nodes * W_edgePerNode),repeat = W_Nsparse))

Ngames = len(sparseList)

#k = 0
#for t in range(Ngames):
t1 = datetime.datetime.now()
for i in range(Nepochs):
    t2 = datetime.datetime.now()
    time_diff = t2 - t1
    print time_diff
    print i
    k = 0
    logR = []
    logTDe = []
    logLastScore = []
    logLastTDe = []
    logTDdiff = []
    logTotalTDe = []
    logGame = []
    while k<=Ngames-1:
        #if len(np.unique(sparseList[k]))!= W_Nsparse or np.all(np.diff(sparseList[k])<=0)==True:
        #print ["k =", k, "before loop", np.diff(sparseList[k]), sparseList[k]]
        if  np.all(np.diff(sparseList[k])>0)==False:
            #print [" in loop", k]
            #print sparseList[k]
            #print np.diff(sparseList[k])
            #print "k += 1"
            k += 1
            #print ['k =', k]
        else:
            #print ['k = ', k]
            #game.R = game.createR([1,2,3,4,5])
            game.R = game.createR(sparseList[k])
            #game.R = game.createR(np.random.choice(sparseList,1))
            player.reset()
            PlogR, PlogTDe = player.runEpoch(game.W, game.R)
            gameMat = [player.Q, game.R, game.W]
            logGame.append(gameMat)
            logR.append(PlogR)
            logLastScore.append(PlogR[-1])
            logTDe.append(PlogTDe)
            logLastTDe.append(PlogTDe[-1])
            TDdiff = master.computeReward(logTDe)
            logTDdiff.append(TDdiff)
            totalTDe = np.sum(PlogTDe)
            logTotalTDe.append(totalTDe)
            sparseGame = sparseList[k]
            for j in range(len(sparseGame)):
                faccumuR[sparseGame[j]] += PlogR[-1]
            #k_old = k
            #while len(np.unique(sparseList[k])) != W_Nsparse:
            #    print ['unique = ', len(np.unique(sparseList[k])), W_Nsparse, 'k = ', k]
            #    k += 1
            #if k == k_old:
            #    k += 1
            k += 1
            #plt.figure(3)
            #plt.plot(PlogR)
            #plt.show()
            #smooth_box = 20
            #smoothed = smooth(PlogR, smooth_box)
            #plt.figure(4)
            #plt.plot(smoothed[smooth_box-1:-smooth_box])
            #plt.figure(5)
            #plt.plot(smoothed)
            #plt.show()
    tot_logR.append(logR)
    tot_logLastScore.append(logLastScore)
    tot_logTDe.append(logTDe)
    tot_logLastTDe.append(logLastTDe)
    tot_logTDdiff.append(logTDdiff)
    tot_logTotalTDe.append(logTotalTDe)
    tot_logGame.append(logGame)

Ngames = tot_logGame[0]

print ['len', len(tot_logGame[0])]
av_logR = np.zeros((P_Nepisodes,1))
gameNum = 100
for i in range(Nepochs):
    for j in range(P_Nepisodes):
        temp = (tot_logR[i][gameNum][j])/Nepochs
        #print av_logR
        av_logR[j] += temp

#plt.plot(av_logR)
#plt.show()

parameters = {'W_nodes':W_nodes, 'W_edgePerNode':W_edgePerNode, 'W_Nsparse':W_Nsparse,
              'P_Nstates':P_Nstates, 'P_Nactions':P_Nactions,'P_eps':P_eps, 'P_gamma':P_gamma, 'P_Nepisodes':P_Nepisodes,
              'P_MaxEpiSteps':P_MaxEpiSteps, 'P_MinAlpha':P_MinAlpha,
              'M_Nstates':M_Nstates, 'M_Nactions':M_Nactions, 'M_eps':M_eps, 'M_gamma':M_gamma, 'M_MaxEpiSteps':M_MaxEpiSteps,
              'M_MinAlpha':M_MinAlpha, 'M_Nepisodes':M_Nepisodes, 'Nepochs':Nepochs, 'Ngames':Ngames}

save_variables = [parameters,tot_logR,tot_logLastScore,tot_logTDe,tot_logLastTDe, tot_logTDdiff,tot_logTotalTDe, tot_logGame]
fileName = time.strftime("Log-%H%M-%d-%m-%Y.p")
with open(fileName, 'wb') as f:
    pickle.dump(save_variables, f)

plt.figure(1)
plt.subplot(231)
plt.hist(logLastScore, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
plt.xlabel("accumulated reward of last episode played")
#plt.figure(2)
plt.subplot(232)
plt.hist(logLastTDe, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
plt.xlabel("accumulated TD error of last episode played")
#plt.figure(3)
plt.subplot(233)
plt.hist(logTDdiff, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
plt.xlabel(" TD error diff of 10 first and last episodes of epoch")
#plt.figure(4)
plt.subplot(234)
plt.hist(logTotalTDe, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
plt.xlabel(" total accumulated TDe in 100 episodes")

plt.subplot(235)
smooth_box = 20
smoothed = smooth(PlogR, smooth_box)
plt.plot(smoothed[smooth_box-1:-smooth_box])
#plt.plot(PlogR)
plt.xlabel("episode number (of the last game simulated)")
plt.ylabel("accumulated reward in episode ")

plt.subplot(236)
plt.text(0.6, 0.5, parameters, size=12, rotation=0.,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ),
         wrap=True)

#plt.matshow(gameDiff)
plt.figure(2)
accumuR = faccumuR.reshape((W_nodes, W_edgePerNode))
plt.matshow(accumuR)

plt.show()
















