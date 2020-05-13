import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,IsCollided
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from exec.evaluationFunctions import ComputeStatistics
from src.constrainedChasingEscapingEnv.envNoPhysics import  TransiteForNoPhysics, Reset,IsTerminal,StayInBoundaryByReflectVelocity


def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['dataSize'] =  [1000,2000,3000]
    manipulatedVariables['depth'] =   [5,9]
    manipulatedVariables['trainSteps'] = list(range(0,50001,10000))


    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    killzoneRadius = 25
    maxRunningSteps = 100
    numSimulations = 200
    # accumulate rewards for trajectories
    numOfAgent=3
    sheepId = 0
    wolvesId = 1

    wolfOneId = 1
    wolfTwoId = 2
    xPosIndex = [0, 1]
    xBoundary = [0,600]
    yBoundary = [0,600]

    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
    getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)

    isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
    isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
    playIsTerminal=lambda state:isTerminalOne(state) or isTerminalTwo(state)

    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)


    playAliveBonus = -1/maxRunningSteps
    playDeathPenalty = 1
    playKillzoneRadius = killzoneRadius
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

# generate trajectory parallel
    generateTrajectoriesCodeName = 'generateCenterControlTrajectoryByCondition.py'
    evalNumTrials = 500
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName,evalNumTrials, numCmdList)

    # run all trials and save trajectories
    generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    # toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..','..', '..', 'data','evaluateSupervisedLearning', 'multiMCTSAgentResNetNoPhysicsCenterControl','evaluateCenterControlTrajByCondition')

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    trajectoryExtension = '.pickle'


    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    getTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories
    fuzzySearchParameterNames = []
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)

    def calculateSuriveRatio(trajectory):
           lenght = np.array(len(trajectory))
           count = np.array([lenght<50, lenght>=50 and lenght<100,lenght>=100])
           return count
    computeNumbers = ComputeStatistics(loadTrajectoriesFromDf, calculateSuriveRatio)
    df = toSplitFrame.groupby(levelNames).apply(computeNumbers)
    print(df)

    fig = plt.figure()
    numRows = 1
    numColumns = 1
    plotCounter = 1
    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    xlabel = ['0-50','50-100','100-150']
    x = np.arange(len(xlabel))

    numTrials = 500
    yMean = df['mean'].tolist()
    yRrr = np.array(df['std'].tolist()) / (np.sqrt(numTrials) -1)

    totalWidth, n = 0.6, 3
    width = totalWidth / n

    x = x - (totalWidth - width) / 2
    plt.bar(x, yMean[0], yerr=yRrr[0],   width=width, label='trainStep0')
    plt.bar(x + width, yMean[1], yerr=yRrr[1], width=width, label='trainStep10000')
    plt.bar(x + width * 2, yMean[2], yerr=yRrr[2],width=width, label='trainStep30000')
    plt.bar(x + width * 3, yMean[3], yerr=yRrr[3],width=width, label='trainStep50000')
    plt.suptitle('dataSize 3000')
    plt.xticks(x, xlabel)
    plt.ylim(0, 1)
    plt.xlabel('living steps')
    plt.legend(loc='best')
    # plt.show()


    # plot the results
    fig = plt.figure()
    numRows = len(manipulatedVariables['depth'])
    numColumns = len(manipulatedVariables['dataSize'])
    plotCounter = 1
    print(statisticsDf)
    for depth, grp in statisticsDf.groupby('depth'):
        grp.index = grp.index.droplevel('depth')

        for dataSize, group in grp.groupby('dataSize'):
            group.index = group.index.droplevel('dataSize')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if plotCounter % numColumns == 1:
                axForDraw.set_ylabel('depth: {}'.format(depth))
            if plotCounter <= numColumns:
                axForDraw.set_title('dataSize: {}'.format(dataSize))

            axForDraw.set_ylim(-1, 1)
            # plt.ylabel('Accumulated rewards')
            maxTrainSteps = manipulatedVariables['trainSteps'][-1]

            plt.plot([0, maxTrainSteps], [0.354]*2, '--m', color="#1C2833", label='pure MCTS')

            group.plot(ax=axForDraw, y='mean', yerr='std',marker='o', logx=False)

            plotCounter += 1

    plt.suptitle('center control wolves')
    plt.legend(loc='best')
    plt.show()

def drawPerformanceLine(dataDf, axForDraw, agentId):
    for key, grp in dataDf.groupby('otherIteration'):
        grp.index = grp.index.droplevel('otherIteration')
        grp['agentMean'] = np.array([value[agentId] for value in grp['mean'].values])
        grp['agentStd'] = np.array([value[agentId] for value in grp['std'].values])
        grp.plot(ax=axForDraw, title='agentId={}'.format(agentId), y='agentMean', yerr='agentStd', marker='o', label='otherIteration={}'.format(key))

if __name__ == '__main__':
    main()
