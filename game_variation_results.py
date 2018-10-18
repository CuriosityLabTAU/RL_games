from adaptive_game import *
import pickle
import datetime

t1 = datetime.datetime.now()

filename = 'Log-2305-17-10-2018.p'
with open(filename, 'rb') as f:
    save_variables = pickle.load(f)

t2 = datetime.datetime.now()
time_diff = t2 - t1
print time_diff
print 'file loaded'
parameters = save_variables[0]
tot_logR = save_variables[1]
tot_logLastScore = save_variables[2]
tot_logTDe = save_variables[3]
tot_logLastTDe = save_variables[4]
tot_logTDdiff = save_variables[5]
tot_logTotalTDe = save_variables[6]
tot_logGame = save_variables[7]

W_nodes = parameters['W_nodes']
W_edgePerNode = parameters['W_edgePerNode']
W_Nsparse = parameters['W_Nsparse']
P_Nstates = parameters['P_Nstates']
P_Nactions = parameters['P_Nactions']
P_eps = parameters['P_eps']
P_gamma = parameters['P_gamma']
P_Nepisodes = parameters['P_Nepisodes']
P_MaxEpiSteps = parameters['P_MaxEpiSteps']
P_MinAlpha = parameters['P_MinAlpha']
M_Nstates = parameters['M_Nstates']
M_Nactions = parameters['M_Nactions']
M_eps = parameters['M_eps']
M_gamma = parameters['M_gamma']
#M_Nepisodes = parameters['M_Nepisodes']
M_MaxEpiSteps = parameters['M_MaxEpiSteps']
M_MinAlpha = parameters['M_MinAlpha']
Nepochs =parameters['Nepochs']
Ngames = parameters['Ngames']
Ngames = 190
M_Nepisodes = 5
#Nepochs = 100

t2 = datetime.datetime.now()
time_diff = t2 - t1
print time_diff
print ' variables extracted'
print ['len(tot_logGame)',len(tot_logGame)]
###progeress graph of a specific game

av_logR = np.zeros(parameters['P_Nepisodes'])
gameNum = 99

for j in range(parameters['P_Nepisodes']):
    for i in range(Nepochs):
        print ['progress', i]
        temp = (tot_logR[i][gameNum][j])/Nepochs
        #print av_logR
        print temp
        av_logR[j] += temp
        #plt.plot((tot_logR[i][gameNum][j])/Nepochs)
        #plt.show()
        #print ['i = ', i, ' j = ', j]
plt.subplot(221)
plt.plot(av_logR)
plt.xlabel("episode number ")
plt.ylabel("accumulated reward in episode (of a specific game)")
#plt.show()


### histogram of average best score in each game
P_eps = 0
#Ngames = len(tot_logGame[0])
bestPlayer = np.zeros(Ngames)
Player = Player(W_nodes, W_edgePerNode, P_eps, P_gamma, P_Nepisodes, P_MaxEpiSteps, P_MinAlpha)
master = Master(M_Nstates, M_Nactions, M_eps, M_gamma, M_Nepisodes, M_MaxEpiSteps, M_MinAlpha)
for j in range(Ngames):
    print ['best', j]
    print ['len(tot_logGame[0])', len(tot_logGame)]
    for i in range(Nepochs):
        Q = tot_logGame[i][j][0]
        R = tot_logGame[i][j][1]
        W = tot_logGame[i][j][2]
        Player.Q = Q
        totalReward, totalTDe = Player.runEpisode(P_Nepisodes-1, W, R)
        bestPlayer[j] += totalReward / Nepochs
plt.subplot(222)
plt.hist(bestPlayer, bins='auto')
plt.xlabel("accumulated reward of last episode played, eps = 0")
#plt.show()

### histogram of average TD difference between first 10 and last 10 episodes
TDdiff = np.zeros(Ngames)
for j in range(Ngames):
    print ['TD diff', j]
    for i in range(Nepochs):
        TDdiff[j] += tot_logTDdiff[i][j] / Nepochs
plt.subplot(223)
plt.hist(TDdiff, bins='auto')
plt.xlabel(" TD error diff of 10 first and last episodes of epoch")
#plt.show()


###histogram of total accumulated TD error in a player's epoch
totTDe = np.zeros(Ngames)
for j in range(Ngames):
    print ['total TDe', j]
    for i in range(Nepochs):
        totTDe[j] += tot_logTotalTDe[i][j] / Nepochs
plt.subplot(224)
plt.hist(totTDe, bins='auto')
plt.xlabel(" total accumulated TDe in 100 episodes")
plt.show()







