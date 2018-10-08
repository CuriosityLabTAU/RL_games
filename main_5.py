import numpy as np
import matplotlib.pyplot as plt

#RL agent learns policy for maximum accumulated reward
#master changes the game each player epoch
#R is converted from matrix to vector

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
        self.Nstates = np.power(2,Nstates) # number of possible R, 2^nodes
        self.Nactions = Nactions # the number of nodes
        self.Q = np.zeros((self.Nstates,self.Nactions))
        self.eps = eps
        self.gamma = gamma
        self.Nepisodes = Nepisodes
        self.MaxEpiSteps = MaxEpiSteps
        self.MinAlpha = MinAlpha
        self.alphas = np.linspace(1.0, self.MinAlpha, self.Nepisodes)

    def computeReward(self, logTDe):
        avg_length = 20
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
            b += np.power(2, X[i])
        return b

def normalize(X, Nstates, Nactions):
    for i in range(0,Nstates):
        for j in range(0,Nactions):
            probSum = np.sum(X[i, j, :])
            X[i, j, :] = X[i, j, :] / probSum
    return X


def arr2int(X):
    b = 0
    for i in range(len(X[:,0])):
        b += np.power(2, X[i,0])
    return int(b)


Nstates = 20
Nactions = 2
Nepisodes = 200
MaxEpiSteps = 10
MinAlpha = 0.01
alphas = np.linspace(1.0, MinAlpha, Nepisodes)
gamma = 0.95

# np.random.seed(0)
W = np.exp(10.0*np.random.rand(Nstates,Nactions, Nstates))
W = normalize(W, Nstates, Nactions)
Rvector = np.random.randint(Nactions, size=Nstates)
R = np.zeros((Nstates,Nactions))
for i  in range(0, Nstates):
    R[i,Rvector[i]] = 1


player = Player(Nstates, Nactions, eps=0.2, gamma=0.95, Nepisodes=200, MaxEpiSteps=10, MinAlpha=0.01)
master = Master(Nstates, Nactions=10, eps=0.2, gamma=0.95, Nepisodes=600, MaxEpiSteps=20, MinAlpha=0.01)

logR = []
logTDe = []
logAction  = []
logPTDe = []
for episode in range(0, master.Nepisodes):
    alpha = master.alphas[episode]
    state_int = arr2int(R)
    state_arr = R
    totalReward = 0
    totalTDe = 0
    player.reset()
    print 'episode = ', episode
    for step in range(0, master.MaxEpiSteps):
        player.reset()
        action = master.chooseAction(state_int)
        print 'action = ', action
        R = master.act(state_arr, action, W, R)
        nextState_int = arr2int(R)
        PlogR, PlogTDe = player.runEpoch(W, R)
        reward = master.computeReward(PlogTDe)
        TDe = reward + gamma*np.max(master.Q[nextState_int,:])-master.Q[state_int, action]
        master.Q[state_int, action] = master.Q[state_int, action]+alpha*(TDe)
        totalReward += reward
        totalTDe += np.abs(TDe)
        logAction.append(action)
    logR.append(totalReward)
    logTDe.append(totalTDe)
    logPTDe.append(np.average(PlogTDe[-10:]))


#print TDepoch

# print ['w', W]
# print ['R', R]
# print ['Q', Q]

plt.figure(1)
plt.plot(logR)
plt.xlabel('Episodes')
plt.ylabel('Accumulated reward')
plt.figure(2)
plt.plot(logTDe)
plt.xlabel('Episodes')
plt.ylabel('Accumulated TD error')
plt.figure(3)
plt.plot(logAction)
plt.xlabel('Episodes')
plt.ylabel('actions')
plt.figure(4)
plt.plot(logPTDe)
plt.xlabel('Episodes')
plt.ylabel('P TDe')
plt.show()














