import numpy as np
import matplotlib.pyplot as plt


def chooseAction(state, Q, eps):
    if np.random.uniform(0, 1) < eps:
        return np.random.choice(np.arange(Nactions))
    else:
        return np.argmax(Q[state, :])

def act(state, action, W, R, Q):
    nextState = W[state, action]
    reward = R[state, action]
    return nextState, reward


Nstates = 10
Nactions = 2
Nepisodes = 500
MaxEpiSteps = 10
MinAlpha = 0.02
alphas = np.linspace(1.0, MinAlpha, Nepisodes)
gamma = 0.9
eps = 0.2
np.random.seed(42)
W = np.random.randint(Nstates, size=(Nstates,Nactions))
R = np.random.rand(Nstates,Nactions)
Q = np.zeros((Nstates,Nactions))
logR = []
logTDe = []

#Qg =

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
        state = nextState

    logR.append(totalReward)
    logTDe.append(totalTDe)

plt.figure(1)
plt.plot(logR)
plt.xlabel('Episodes')
plt.ylabel('Accumulated reward')
plt.figure(2)
plt.plot(logTDe)
plt.xlabel('Episodes')
plt.ylabel('Accumulated TD error')
plt.show()














