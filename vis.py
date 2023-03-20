import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
# import io
# from PIL import Image
import imageio.v3 as iio
import os
import glob


def visState(S, nPlayers, visualize, figs, title=None, action=None):
    if visualize == 'none':
        return
    
    gridSize = np.shape(S)[0]

    alpha = 0.5
    cmap = {0:[0.1,0.1,1.0,alpha],1:[1.0,0.1,0.1,alpha],2:[1.0,0.5,0.1,alpha], 3:[0.5,0.5,0.1,alpha]}
    labels = {0:'Player 0',1:'Player 1',2:'Player 2',3:'Player 3'}
    S = np.swapaxes(S, 0, 1)
    arrayShow = np.array([[cmap[i] for i in j] for j in S[:,:,0]])    
    ## create patches as legend
    patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in range(nPlayers)]

    fig = plt.figure()

    plt.imshow(arrayShow)
    plt.legend(handles=patches, bbox_to_anchor=(1.02, 0.2), loc='upper left', borderaxespad=0.)

    baseFontSz = 5
    scaler = 1.5
    for x in range(gridSize):
        for y in range(gridSize):
            troopsBin = S[x,y,1] if S[x,y,1] < 6 else 6
            plt.text(y,x,S[x,y,1], fontsize=baseFontSz + troopsBin*scaler)
    if visualize == 'end':
        plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    if title is not None:
        plt.title(title)
    if action is not None:
        orig_x = action[0]
        orig_y = action[1]
        if action[2] == 'n':
            dy = 0.6
            dx = 0
            orig_y += 0.2
        if action[2] == 'e':
            dy = 0
            dx = 0.6
            orig_x += 0.2
        if action[2] == 's':
            dy = -0.6
            dx = 0
            orig_y -= 0.2
        if action[2] == 'w':
            dy = 0
            dx = -0.6
            orig_x -= 0.2
        plt.arrow(orig_x, orig_y, dx, dy, length_includes_head=True, width=0.05, color='black')

    # print(f'VIS S: owner \n {S[:,:,0]} \n troops \n {S[:,:,1]}')
    # print(f'S[0,0:4] {S[0,0,0]}, {S[0,1,0]}, {S[0,2,0]}, {S[0,3,0]}')
    if visualize == 'each':
        plt.show()
    elif visualize == 'end':
        figs.append(fig)
        plt.close()
        return figs
    else:
        return
    
def visGameFlow(figs, fileNamePrefix=""):
    path = 'gameFlow.gif'
    nFrames = len(figs)
    imgs = np.zeros(nFrames).tolist()
    for i in range(nFrames):
        print(f'Saving game flow gif, {np.round(100*i/nFrames,1)}% done', end='\r')
        figs[i].savefig(f'./plots/{i:05d}')

    frames = np.stack([iio.imread(f'./plots/{i:05d}.png') for i in range(nFrames)], axis=0)

    iio.imwrite(fileNamePrefix+'_'+path, frames)
    # optimize(path)
    print('\n Game flow saved to ', fileNamePrefix+'_'+path)
    return

def clearAllPlots():
    path = './plots/*'
    files = glob.glob(path)
    for f in files:
        os.remove(f)

def plotGameProgress(S_hist, r_hist, nPlayers, fileNamePrefix=''):
    nTurns = len(S_hist)

    nCells = np.zeros((nTurns, nPlayers))
    nTroops = np.zeros((nTurns, nPlayers))

    # get troops and cells vs action number
    for t in range(nTurns):
        S = S_hist[t]
        for p in range(nPlayers):
            doesPlayerOwnCell = S[:,:,0]==p
            nCells[t, p] = np.sum(doesPlayerOwnCell)
            nTroops[t, p] = np.sum(S[:,:,1][doesPlayerOwnCell])

    # plot
    alpha = 1
    cmap = {0:[0.1,0.1,1.0,alpha],1:[1.0,0.1,0.1,alpha],2:[1.0,0.5,0.1,alpha], 3:[0.5,0.5,0.1,alpha]}
    plt.figure(dpi=300, figsize=(10,8))

    plt.subplot(3,1,1)
    plt.title('Number of territories owned per player after every action')
    plt.xlabel('Action number'); plt.ylabel('Number of territories owned'); #plt.ylim(bottom=0)
    plt.stackplot(range(nTurns), nCells.T, colors=np.array(list(cmap.values())))

    plt.subplot(3,1,2)
    plt.title('Number of troops per player after every action')
    plt.xlabel('Action number'); plt.ylabel('Number of troops'); 
    for p in range(nPlayers):
        plt.plot(range(nTurns), nTroops[:, p], c=cmap[p], label=str(p))
    plt.legend(title='Player #:');plt.ylim(bottom=0)
    
    plt.subplot(3,1,3)
    plt.title('Accumulated reward')
    plt.ylabel('Cumulative reward'); plt.xlabel('Action number')
    for p in range(nPlayers):
        x = range(0, np.shape(r_hist)[0])
        plt.plot(x, r_hist[:,p], c=cmap[p])

    
    plt.tight_layout()
    plt.savefig('./gameGraphs/'+fileNamePrefix+'_'+'gameGraphs.png')
    print(fileNamePrefix+'_'+'gameGraphs.png saved')
    return
    

