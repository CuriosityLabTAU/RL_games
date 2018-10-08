import numpy as np
import matplotlib.pyplot as plt

#RL agent learns policy for maximum accumulated reward

class player:

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


    def chooseAction(self, state):
        if np.random.uniform(0, 1) < self.eps:
            return np.random.choice(np.arange(self.Nactions))
        else:
            return np.argmax(self.Q[state, :])

    def act(self, state, action, W, R):
        nextState = np.random.choice(np.arange(self.Nstates), p=W[state, action,:])
        reward = R[state, action]
        return nextState, reward

    def runStep(self, state, action, alpha, W, R):
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
            TDe, reward, nextState = self.runStep(self, state, action, alpha)
            state = nextState
            totalReward += reward
            totalTDe += TDe
        return totalReward, totalTDe

    def runEpoch(self, W, R):
        logR = []
        logTDe = []
        for episode in range(0,self.Nepisodes):
            totalReward, totalTDe = self.runEpisode(episode)
        logR.append(totalReward)
        logTDe.append(totalTDe)
        return logR, logTDe


def normalize(X, Nstates, Nactions):
    for i in range(0,Nstates):
        for j in range(0,Nactions):
            probSum = np.sum(X[i, j, :])
            X[i, j, :] = X[i, j, :] / probSum
    return X

Nstates = 10
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

print R

logR = []
logTDe = []

for e in range(Nepisodes):

    state = 0
    totalReward = 0
    totalTDe = 0
    alpha = alphas[e]
    for step in range(MaxEpiSteps):
        action = chooseAction(state, Q, eps)
        nextState, reward = act(state, action, W, R, Q)
        TDe = reward + gamma*np.max(Q[nextState,:])-Q[state, action]
        totalReward += reward
        totalTDe += TDe
        Q[state, action] = Q[state, action]+alpha*(reward + gamma*np.max(Q[nextState,:])-Q[state, action])
        if e == Nepisodes-1:
            print(state, action, nextState, reward)
        state = nextState

    logR.append(totalReward)
    logTDe.append(totalTDe)

avg_length = 20
TDepoch = np.average(logTDe[0:avg_length-1]) - np.average(logTDe[-avg_length:])
print TDepoch

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
plt.show()














