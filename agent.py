import numpy as np
import random
from game_mgmt import *
import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar

from tensorflow import keras
from keras import Model, Sequential
from keras.layers import Dense, Embedding, Reshape, Flatten
from keras.optimizers import Adam

# choose an action (x, y, direction) from action set A in board state S in using random or deep q-learning
def chooseAction(S, p, A, agent=None):
    gridSize = np.shape(S)[0]
    randomAgentLikelihoodOfPassing = 10 # percent
    
    if agent == None: # choose randomly
        # the random agent has a 10% change of just passing and not attacking
        if np.random.randint(100) < randomAgentLikelihoodOfPassing:
            return [A[0], 0]
        else:
            actionList = np.arange(len(A))
            random.shuffle(actionList)
    else:  # use deep Q-learning
        S_flat = get1DState(S)
        actionList = agent.act(S_flat)

    # iterate through prioritized action list, pick the first valid action
    for a_idx in actionList:
        if checkValidAction(S, A[a_idx], p):
            return [A[a_idx], a_idx]
    assert False, f'ERROR actionList = {actionList}'


def chooseTroopPlacement(S, p, agent=None):
    gridSize = np.shape(S)[0]
    if agent == None or True: # debug always do this
        placementList = np.arange(gridSize**2)
        random.shuffle(placementList)
    else:  # use deep Q-learning
        S_flat = get1DState(S)
        placementList = agent.act(S_flat)
        pass
    
    for i in range(len(placementList)):
        cellToPlace = placementList[i]
        cellxy = np.unravel_index(cellToPlace, (gridSize, gridSize))

        # if player owns the territory
        if p == S[cellxy[0], cellxy[1], 0]:
            break
    
    S[cellxy[0], cellxy[1], 1] += 1
    return S

def getReward(S_orig, S, p):
    gridSize = np.shape(S)[0]

    r_defeatEnemies = 1
    r_loseFriendlies = -1.1
    r_nCells = 20
    r_takeCell = 100
    r_winGame = 10000

    r = 0

    for x in range(gridSize):
        for y in range(gridSize):
            # if we originally owned the cell, check if we've lost troops
            if S_orig[x,y,0] == p:
                r += (S_orig[x,y,1] - S[x,y,1]) * r_loseFriendlies

            # if we didn't originally own the cell
            else:
                # if we've taken the cell
                if S[x,y,0] == p:
                    r += r_takeCell
                # if we still don't own the cell, check if we've defeated some of its troops
                else:
                    r += (S_orig[x,y,1] - S[x,y,1]) * r_defeatEnemies

    # number of cells owned
    # r += r_nCells * np.sum(S[:,:,0] == p)

    # check if game was won
    if np.all(S[:,:,0] == S[0,0,0]) and S[0,0,0] == p:
        r += r_winGame

    return r

def trackRewards(r, player, r_history):
    newRow = r_history[-1,:]
    newRow[player] += r
    r_history = np.vstack((r_history, newRow))
    return r_history

    
    

    
#################################################################################################################################
 
class Agent:
    def __init__(self, gridSize, nPlayers, optimizer, agentType, loadPath=None):
        
        # Initialize atributes
        nTroopBins = 6
        self._grid_size = gridSize
        self._state_size = nTroopBins*nPlayers # gridSize**2
        if agentType == 'place':
            self._action_size = gridSize**2
        elif agentType == 'attack':
            self._action_size = 4*(gridSize-2)**2 + 3*4*(gridSize-2) + 2*4 + 1 # can attack in 4, 3, and 2 directions in various cells, plus one "pass" action
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        
        # Build or load networks
        if loadPath == None:
            self.q_network = self._build_compile_model()
            self.target_network = self._build_compile_model()
        else:
            self.q_network = keras.models.load_model(loadPath)
            self.target_network = keras.models.load_model(loadPath)
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model = Sequential()
        # model.add(Embedding(self._state_size, 10, input_length=self._grid_size**2))
        # model.add(Reshape((10*(self._grid_size**2),)))
        # model.add(Reshape((10,)))
        # model.add(Reshape((-1,1)))

        # model.add(Flatten())
        model.add(Dense(20, input_shape=(self._grid_size**2,), activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            preferredActions = np.arange(self._action_size)
            np.random.shuffle(preferredActions)
            return preferredActions
        q_values = self.q_network.predict(state, verbose=0)
        preferredActions = np.argsort(q_values[0])
        return preferredActions

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state, terminated in minibatch:
            
            target = self.q_network.predict(state, verbose=0)
            
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state, verbose=0) 
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.q_network.fit(state, target, epochs=1, verbose=0)