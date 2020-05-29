import pandas as pd
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import pylab as plt
import numpy as np
import seaborn as sns 
import itertools as it

from src.MDPChasing.envNoPhysics import Reset
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, ApproximateValue, restoreVariables

def evaluateValue(modelDf, valueFunction, wolvesState):
    sheepState = np.asarray(modelDf.index.values)
    state = np.concatenate([list(sheepState)] + [wolvesState])
    stateValue = max(0.0, valueFunction(state))
    resultSe = pd.Series({'stateValue': stateValue})
    return resultSe

def drawHeatmapPlot(plotDf, ax):
    plotDf = plotDf.reset_index().pivot(columns= 'sheepXPosition', index = 'sheepYPosition', values = 'stateValue')
    sns.heatmap(plotDf, ax = ax)
    ax.set_xlabel('sheepX')
    ax.set_ylabel('sheepY')

def drawLinePlot(plotDf, ax):
    for sheepYPosition, subDf in plotDf.groupby('sheepYPosition'):
        subDf = subDf.droplevel('sheepYPosition')
        subDf.plot.line(ax = ax, label = 'sheepY = {}'.format(sheepYPosition), y = 'reward', marker = 'o')
    ax.set_xlabel('sheepX')

def main():
    numWolves = 2
    numSheep = 1
    numWolvesStateSpaces = [2 * (numInWe + 1) 
            for numInWe in range(2, numWolves + 1)]
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    #actionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10)]
    predatorPowerRatio = 8
    wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolvesCentralControlActionSpaces = [list(it.product(wolfIndividualActionSpace, repeat = numInWe)) 
            for numInWe in range(2, numWolves + 1)]
    numWolvesCentralControlActionSpaces = [len(wolvesCentralControlActionSpace)
            for wolvesCentralControlActionSpace in wolvesCentralControlActionSpaces]
    regularizationFactor = 1e-4
    generateWolvesCentralControlModels = [GenerateModel(numStateSpace, numActionSpace, regularizationFactor) 
        for numStateSpace, numActionSpace in zip(numWolvesStateSpaces, numWolvesCentralControlActionSpaces)]
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    wolfNNDepth = 9
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    initWolvesCentralControlModels = [generateWolvesCentralControlModel(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths, 
            resBlockSize, initializationMethod, dropoutRate) for generateWolvesCentralControlModel in generateWolvesCentralControlModels] 
    NNNumSimulations = 250
    wolvesModelPaths = [os.path.join('..', '..', 'data', 'preTrainModel', 
            'agentId='+str(len(actionSpace) * np.sum([10**_ for _ in
            range(numInWe)]))+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations='+str(NNNumSimulations)+'_trainSteps=50000') 
            for numInWe in range(2, numWolves + 1)]
    print(wolvesModelPaths)
    wolvesCentralControlNNModels = [restoreVariables(initWolvesCentralControlModel, wolvesModelPath) 
            for initWolvesCentralControlModel, wolvesModelPath in zip(initWolvesCentralControlModels, wolvesModelPaths)]
    wolvesValueFunctionListBasedOnNumAgentsInWe = [ApproximateValue(NNModel)
            for NNModel in wolvesCentralControlNNModels] 
    valueFunction = wolvesValueFunctionListBasedOnNumAgentsInWe[numWolves - 2]


    xBoundary = [0,600]
    yBoundary = [0,600]
    reset = Reset(xBoundary, yBoundary, numWolves)

    numGridX = 120
    numGridY = 120
    xInterval = (xBoundary[1] - xBoundary[0])/numGridX
    yInterval = (yBoundary[1] - yBoundary[0])/numGridY
    sheepXPosition = [(gridIndex + 0.5) * xInterval for gridIndex in range(numGridX)]
    sheepYPosition = [(gridIndex + 0.5) * yInterval for gridIndex in range(numGridY)]
    
    
    wolvesState = reset()
    wolvesState = np.array([[300, 350], [550, 400]])
    print(wolvesState)
    levelValues = [sheepXPosition, sheepYPosition]
    levelNames = ["sheepXPosition", "sheepYPosition"]

    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)

    toSplitFrame = pd.DataFrame(index = modelIndex)

    evaluate = lambda df: evaluateValue(df, valueFunction, wolvesState)
    valueResultDf = toSplitFrame.groupby(levelNames).apply(evaluate)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    drawHeatmapPlot(valueResultDf, ax)

    fig.savefig('valueMap2', dpi = 300)
    plt.show()

if __name__ == "__main__":
    main()
