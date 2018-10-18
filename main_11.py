from adaptive_game import *

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

M_Nstates = np.power(W_edgePerNode*W_nodes,2)
M_Nactions = W_nodes * W_edgePerNode
M_eps = 0.1
M_gamma = 0.95
M_Nepisodes = 5
M_MaxEpiSteps = 30
M_MinAlpha = 0.01


game = MDP(W_nodes, W_edgePerNode, W_Nsparse)
player = Player(W_nodes, W_edgePerNode, P_eps, P_gamma, P_Nepisodes, P_MaxEpiSteps, P_MinAlpha)
master = Master(M_Nstates, M_Nactions, M_eps, M_gamma, M_Nepisodes, M_MaxEpiSteps, M_MinAlpha)


logFinalR = []
logR = []
logr = []
logTDe = []
logAction  = []
logPTDe = []
for episode in range(0, master.Nepisodes):
    #game.reset()
    #game = MDP(nodes, edgePerNode)
    alpha = master.alphas[episode]
    totalReward = 0
    totalTDe = 0
    player.reset()
    print 'episode = ', episode
    for step in range(0, master.MaxEpiSteps):
        player.reset()
        action = master.chooseAction(game.R)
        #print 'action = ', action
        game.R = master.act(state_arr, action, game.W, game.R)
        nextState_int = master.arr2int(game.R)
        PlogR, PlogTDe = player.runEpoch(game.W, game.R)
        if episode==master.Nepisodes-1 and step==master.MaxEpiSteps-1:
            plt.figure(6)
            plt.plot(PlogR)
            plt.figure(7)
            plt.plot(PlogTDe)
        reward = master.computeReward(PlogTDe)
        #reward = -master.computeReward(PlogR)
        TDe = reward + master.gamma*np.max(master.Q[nextState_int,:])-master.Q[state_int, action]
        master.Q[state_int, action] = master.Q[state_int, action]+alpha*(TDe)
        totalReward += reward
        totalTDe += np.abs(TDe)
        logAction.append(action)
    logr.append(reward)
    logR.append(totalReward)
    logTDe.append(totalTDe)
    logPTDe.append(np.average(PlogTDe[-10:]))