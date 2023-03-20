import numpy as np
import time
from vis import *
from game_mgmt import *
from agent import *
import timeit
import pickle




def playGame(agents, gridSize=4, visType='end', maxTurns = 250, seed=None, batch_size=8, fileNamePrefix="", noLearning=False):
   
    # game setup
    if seed != None:
        np.random.seed(seed)
    nPlayers = len(agents)
    maxAttacksPerTurn = 10
    S = startGame(gridSize, nPlayers) # state of board is shape(size, size, 2). [x,y,0] is player owner, [x,y,1] is nTroops of cell (x,y)
    A = generateActionSet(gridSize)

    # housekeeping
    if visType == 'end':
        clearAllPlots()
    turnCount = 0
    S_history = []
    r_history = np.zeros((1, nPlayers))
    figs = []
    startt = timeit.default_timer()
    winner = None
    
    for turnCount in range(maxTurns):
        runTime = (timeit.default_timer() - startt) / 60
        print(f'turn {turnCount} ({round((turnCount/runTime),2)} turns/min)', end="\r")
        player = np.mod(turnCount, nPlayers)
                
        figs = visState(S, nPlayers, visType, figs, title=f'Turn {turnCount}: Start of player {player}\'s turn')
        
        nTroopsToPlace = getResupplyCount(S, player)
        for t in range(nTroopsToPlace):
            S = chooseTroopPlacement(S, player, agents[player]['place'])
        figs = visState(S, nPlayers, visType, figs, title=f'Turn {turnCount}: Player {player} troops placed')

        for _ in range(maxAttacksPerTurn): 
            S_history.append(copy.deepcopy(S))
            S_orig = copy.deepcopy(S)
            a, a_idx = chooseAction(S, player, A, agents[player]['attack'])
            # if player hasn't passed their turn
            if a[0] != -1:
                S, rolls = doAction(S, player, a)
                r = getReward(S_orig, S, player)

                # visualize action and outcome
                title = f'Turn {turnCount}: Player {player} attacks {a[2]} from {a[0], a[1]} \n Attacker rolls {rolls[0]}, Defender rolls {rolls[1]}'
                figs = visState(S_orig, nPlayers, visType, figs, title=title, action=a)
                figs = visState(S, nPlayers, visType, figs, title=f'Turn {turnCount}: Player {player}\'s attack outcome (reward = {r})')

            # if player has passed their turn
            else: 
                r = -.05
                figs = visState(S_orig, nPlayers, visType, figs,title=f'Turn {turnCount}: Player {player} ends their turn')

            # store experience in memory
            r_history = trackRewards(r, player, r_history)
            terminated = True if np.all(S[:,:,0] == S[0,0,0]) else False
            if agents[player]['attack'] != None and not noLearning:
                agents[player]['attack'].store(get1DState(S_orig), a_idx, r, get1DState(S), terminated)
            
            if terminated or a[0] != -1:
                break

        # 25% of the time, retrain the model using random memories
        doRetrain = True if np.random.randint(0,100) < 25 else False
        if agents[player]['attack'] != None and len(agents[player]['attack'].expirience_replay) > batch_size and doRetrain and not noLearning:
            # print('\n retrain')
            agents[player]['attack'].retrain(batch_size)
        
        # Check if game is complete
        if np.all(S[:,:,0] == S[0,0,0]):
            winner = S[0,0,0]
            title=f'Turn {turnCount}: End of game! Player {winner} wins!'
            figs = visState(S, nPlayers, visType, figs, title=title)
            break

    if visType == 'end':
        visGameFlow(figs, fileNamePrefix)
    plotGameProgress(S_history, r_history, nPlayers, fileNamePrefix)
    hist = [S_history, r_history]
    return [agents, winner, hist]

    
def main():
    maxTime_mins = 60*1
    visType = 'none' # 'end', 'each', or 'none'
    gridSize = 3
    seed = 1
    maxTurns = 250
    retraining = False

    agents = {
        0: {
            'place': None,
            'attack': None
        },
        1: {
            'place': None,
            'attack': None
        }
    }

    nPlayers = len(agents)
    optimizer = Adam(learning_rate=0.1)
    # agents[1]['place'] = Agent(gridSize, nPlayers, optimizer, 'place')
    if retraining:
        agents[0]['attack'] = Agent(gridSize, nPlayers, optimizer, 'attack')
    else:
        agents[0]['attack'] = Agent(gridSize, nPlayers, optimizer, 'attack', './attackModel_3x3_5mins.h5')
    agents[0]['attack'].q_network.summary()

    startt = timeit.default_timer()
    count = 0
    history_all = []
    winners = []
    r_all = {'random':[], 'agent':[]}
    while (timeit.default_timer() - startt) / 60 < maxTime_mins:
        if retraining:
            agents, winner, hist = playGame(agents, gridSize, visType, maxTurns, seed=count, fileNamePrefix=str(count))
            agents[0]['attack'].q_network.save('attackModel_3x3_5mins.h5')
        else:
            _, winner, hist = playGame(agents, gridSize, visType, maxTurns, seed=count, fileNamePrefix=str(count), noLearning=True)

        history_all.append(hist)
        winners.append(winner)
        winRate = np.round(1-(sum(winners)/len(winners)), 2)
        print(f'\n\n*****\n END OF RUN {count}. Winner was player {winner}. Runtime {(timeit.default_timer() - startt) / 60}' + 
              f' of {maxTime_mins} mins. Agent reward was {round(history_all[-1][1][-1,0],2)}, random was {round(history_all[-1][1][-1,1],2)}. Agent winrate {winRate} \n******\n\n')
        r_all['agent'].append(history_all[-1][1][-1,0])
        r_all['random'].append(history_all[-1][1][-1,1])
        count += 1

        if len(winners) > 99:
            break
    
    _, winner, history_all = playGame(agents, gridSize, 'end', maxTurns, seed=count, fileNamePrefix=str(count))

    # plot total reward vs run
    x = range(len(r_all['agent']))
    plt.figure()
    plt.plot(x, r_all['agent'], color='b', label='agent')
    plt.plot(x, r_all['random'], color='r', label='random')
    plt.legend()
    plt.xlabel('run #'); plt.ylabel('total reward')
    plt.savefig('./gameGraphs/learning.png')

    # save model and training history
    agents[0]['attack'].q_network.save('attackModel_3x3_5mins.h5')
    fileObj = open('train_history.pkl', 'wb')
    pickle.dump(history_all,fileObj)

    plt.figure()
    plt.title('Total accumulated reward over 100 games')
    plt.plot(range(len(r_all['agent'])), np.cumsum(r_all['agent']), color='blue', label='agent')
    plt.plot(range(len(r_all['random'])), np.cumsum(r_all['random']), color='red', label='random')
    plt.xlabel('games played'); plt.ylabel('total reward')
    plt.legend()
    plt.savefig('total reward, all games.png')
    
    a = 1+1
    return

if __name__ == '__main__':
    main()

# todo
#X1. figure out verbose
#X2. add accumulated reward
# 3. plot reward vs. run
