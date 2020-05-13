import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict
import itertools as it
import pathos.multiprocessing as mp
import pandas as pd
from matplotlib import pyplot as plt


from src.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, backup, InitializeChildren, Expand, RollOut
import src.constrainedChasingEscapingEnv.envNoPhysics as env
import src.constrainedChasingEscapingEnv.reward as reward
from src.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors
from src.episode import chooseGreedyAction, SampleTrajectory
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset, StayInBoundaryByReflectVelocity
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
import time
from exec.trajectoriesSaveLoad import GetSavePath, saveToPickle
from exec.evaluationFunctions import ComputeStatistics


class SampleTrajectoryFixRet:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction

    def __call__(self, policy, trialIndex):
        state = self.reset(trialIndex)
        while self.isTerminal(state):
            state = self.reset(trialIndex)
        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [self.chooseAction(actionDist) for actionDist in actionDists]
            trajectory.append((state, action, actionDists))
            actionFortransit = [action[0], action[1][0], action[1][1]]
            nextState = self.transit(state, actionFortransit)
            state = nextState

        return trajectory


def generateOneCondition(parameters):
    print(parameters)
    numTrials = 7
    numSimulations = int(parameters['numSimulations'])

    killzoneRadius = 30
    maxRunningSteps = 100

    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius, 'numTrials': numTrials}

    trajectorySaveExtension = '.pickle'
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeSingleChasingNoPhysics', 'evaluateMCTSTBaseLineTajectories')

    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parameters)

    numOfAgent = 3
    sheepId = 0
    wolvesId = 1

    wolfOneId = 1
    wolfTwoId = 2
    xPosIndex = [0, 1]
    xBoundary = [0, 600]
    yBoundary = [0, 600]

    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
    getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)

    isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
    isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
    isTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 3
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

    predatorPowerRatio = 2
    wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolvesActionSpace = list(it.product(wolfActionOneSpace, wolfActionTwoSpace))

    numSheepActionSpace = len(sheepActionSpace)
    numWolvesActionSpace = len(wolvesActionSpace)

    numStateSpace = 6
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

    # load save dir
    NNModelSaveExtension = ''
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeMultiChasingNoPhysics', 'trainedResNNModelsMultiStillAction')
    NNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 150, 'numSimulations': 200, 'miniBatchSize': 256, 'learningRate': 0.0001, }
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)
    depth = 5
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
    sheepTrainedModelPath = getNNModelSavePath({'trainSteps': 50000, 'depth': depth})
    sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
    sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

    # sheepPolicy = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    # select child
    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)

    # prior
    getActionPrior = lambda state: {action: 1 / len(wolvesActionSpace) for action in wolvesActionSpace}

# load chase nn policy

    def wolvesTransit(state, action): return transit(
        state, [chooseGreedyAction(sheepPolicy(state)), action[0], action[1]])

    # reward function
    aliveBonus = -1 / maxRunningSteps
    deathPenalty = 1
    rewardFunction = reward.RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)

    # initialize children; expand
    initializeChildren = InitializeChildren(
        wolvesActionSpace, wolvesTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    # random rollout policy
    def rolloutPolicy(
        state): return wolvesActionSpace[np.random.choice(range(numWolvesActionSpace))]

    # rollout
    rolloutHeuristicWeight = 0.1
    rolloutHeuristic1 = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getWolfOneXPos, getSheepXPos)
    rolloutHeuristic2 = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getWolfTwoXPos, getSheepXPos)
    rolloutHeuristic = lambda state: (rolloutHeuristic1(state) + rolloutHeuristic2(state)) / 2

    maxRolloutSteps = 10

    rollout = RollOut(rolloutPolicy, maxRolloutSteps, wolvesTransit, rewardFunction, isTerminal, rolloutHeuristic)

    wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

    # All agents' policies
    policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]

    np.random.seed(1447)
    initPositionList = [[env.samplePosition(xBoundary, yBoundary) for j in range(numOfAgent)] for i in range(numTrials)]
    reset = env.FixedReset(initPositionList)
    sampleTrajectory = SampleTrajectoryFixRet(maxRunningSteps, transit, isTerminal, reset, chooseGreedyAction)

    startTime = time.time()
    trajectories = [sampleTrajectory(policy, trial) for trial in range(numTrials)]
    finshedTime = time.time() - startTime

    saveToPickle(trajectories, trajectorySavePath)

    print(parameters)
    print('lenght:', np.mean([len(tra) for tra in trajectories]))
    print('timeTaken:', finshedTime)


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSimulations'] = [50,100, 200, 400]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    trainPool = mp.Pool(numCpuToUse)
    # trainPool.map(generateOneCondition, parametersAllCondtion)

    # load data
    dirName = os.path.dirname(__file__)
    # trajectoryDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateEscapeSingleChasingNoPhysics', 'evaluateMCTSTBaseLineTajectories')

    trajectoryDirectory = os.path.join(dirName, '..', '..', '..', 'data','evaluateSupervisedLearning', 'multiMCTSAgentResNetNoPhysicsCenterControl', 'trajectories')


    trajectoryExtension = '.pickle'
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)


    numTrials = 7
    killzoneRadius = 30
    maxRunningSteps = 100

    # trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius, 'numTrials': numTrials}
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'killzoneRadius': killzoneRadius}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    sheepId = 0
    wolfId = 1
    wolfOneId = 1
    wolfTwoId = 2
    xPosIndex = [0, 1]
    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
    getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)
    isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
    isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
    playIsTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

    xPosIndex = [0, 1]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)

    playAliveBonus = -1 / maxRunningSteps
    playDeathPenalty = 1
    playKillzoneRadius = killzoneRadius
    playReward = reward.RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)

    # compute statistics on the trajectories
    fuzzySearchParameterNames = ['sampleIndex','agentId']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)

    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)

    # plot the results
    fig = plt.figure()
    numRows = 1
    numColumns = 1
    plotCounter = 1
    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    axForDraw.set_ylim(-1, 1)

    statisticsDf.plot(ax=axForDraw, y='mean', yerr='std', marker='o', logx=False)
    plt.ylabel('Accumulated rewards')
    plt.xlim(0)
    plt.suptitle('Evaulate Center Control Wolves MCTS with trained sheep')
    plt.legend(loc='best')
    plt.show()

    def calculateSuriveRatio(trajectory):
        lenght = np.array(len(trajectory))
        count = np.array([lenght < 50, lenght >= 50 and lenght < 100, lenght >= 100])
        return count
    computeNumbers = ComputeStatistics(loadTrajectoriesFromDf, calculateSuriveRatio)
    df = toSplitFrame.groupby(levelNames).apply(computeNumbers)
    print(df)

    fig = plt.figure()
    numRows = 1
    numColumns = 1
    plotCounter = 1
    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
    xlabel = ['0-50', '50-100', '100-150']
    x = np.arange(len(xlabel))

    yMean = df['mean'].tolist()
    yRrr = np.array(df['std'].tolist()) / (np.sqrt(numTrials) - 1)

    totalWidth, n = 0.3, 5
    width = totalWidth / n

    x = x - (totalWidth - width) / 2
    xlabels = manipulatedVariables['numSimulations']
    for i in range(len(yMean)):
        plt.bar(x + width * i, yMean[i], width=width, label='simulation={}'.format(xlabels[i]))
        # plt.bar(x + width * i, yMean[i], yerr=yRrr[i], width=width, label='simulation={}'.format(xlabels[i]))

    plt.xticks(x, xlabel)
    plt.ylim(0, 1)
    plt.xlabel('living steps')
    plt.legend(loc='best')
    plt.suptitle('Evaulate Center Control Wolves MCTS with trained sheep')
    # plt.show()


if __name__ == "__main__":
    main()
