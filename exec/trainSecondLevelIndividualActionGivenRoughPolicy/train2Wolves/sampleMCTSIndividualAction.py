import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
import itertools as it

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, TransitGivenOtherPolicy, IsTerminal
import src.MDPChasing.reward as reward
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import PolicyOnChangableIntention, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy

from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import Render, SampleTrajectoryWithRender
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution


def main():
    DEBUG = 0
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 0
        endSampleIndex = 9
        agentId = 1
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    else:
        parametersForTrajectoryPath = json.loads(sys.argv[1])
        startSampleIndex = int(sys.argv[2])
        endSampleIndex = int(sys.argv[3])
        agentId = int(parametersForTrajectoryPath['agentId'])
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'trainLevel2IndividualActionPolicy', '2Wolves', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 300
    killzoneRadius = 80
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        numOfAgent = 3
        possiblePreyIds = [0]
        possiblePredatorIds = list(range(1, numOfAgent))
        posIndexInState = [0, 1]
        getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
        getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
        isTerminal = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)

        # space
        #actionSpace = [(10, 0), (7.7, 7.7), (0, 10), (-7.7, 7.7), (-10, 0), (-7.7, -7.7), (0, -10), (7.7, -7.7), (0, 0)]
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfActionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)]

        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

        predatorPowerRatio = 8
        wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

        wolfIndividualActionSpaceInCentral = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfCentralControlActionSpace = list(it.product(wolfIndividualActionSpaceInCentral, repeat=numOfAgent - 1))

        #actionSpaceList = [sheepActionSpace, wolvesActionSpace]
        numStateSpace = 2 * numOfAgent
        numSheepActionSpace = len(sheepActionSpace)
        numWolfCentralControlActionSpace = len(wolfCentralControlActionSpace)
        numWofActionSpace = len(wolfIndividualActionSpace)

        # sheep Policy
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        depth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'preTrainModel')
        sheepNNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 50, 'numSimulations': 100, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getSheepNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, sheepNNModelFixedParameters)
        sheepTrainedModelPath = getSheepNNModelSavePath({'trainSteps': 50000, 'depth': depth})

        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

        # wolves Rough Policy
        generateWolvesModel = GenerateModel(numStateSpace, numWolfCentralControlActionSpace, regularizationFactor)

        wolvesRoughNNModelFixedParameters = {'agentId': 55, 'maxRunningSteps': 50, 'numSimulations': 300, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getWolvesRoughNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, wolvesRoughNNModelFixedParameters)
        wolvesRoughTrainedModelPath = getWolvesRoughNNModelSavePath({'trainSteps': 50000, 'depth': depth})

        initWolvesRoughNNModel = generateWolvesModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        wolvesRoughTrainedModel = restoreVariables(initWolvesRoughNNModel, wolvesRoughTrainedModelPath)
        wolvesRoughPolicy = ApproximatePolicy(wolvesRoughTrainedModel, wolfCentralControlActionSpace)

        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        def getActionPrior(state): return {action: 1 / len(wolfIndividualActionSpace) for action in wolfIndividualActionSpace}

        # transitCentralControl
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

        # transitInTree
        wolvesImaginedWeId = list(range(1, numOfAgent))
        otherWolvesId = list(range(2, numOfAgent))
        actionIndexesInCentralControl = [wolvesImaginedWeId.index(wolfId) for wolfId in otherWolvesId]

        def transitInMCTS(state, individualAction): return transit(state, np.concatenate([[maxFromDistribution(sheepPolicy(state)), individualAction],
                                                                                          np.array(maxFromDistribution(wolvesRoughPolicy(state)))[actionIndexesInCentralControl]]))

        # reward function
        aliveBonus = -1 / maxRunningSteps
        deathPenalty = 1
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            wolfIndividualActionSpace, transitInMCTS, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # rollout
        rolloutHeuristicWeight = 1e-2
        sheepId = 0
        getSheepPos = GetAgentPosFromState(sheepId, posIndexInState)
        getWolvesPoses = [GetAgentPosFromState(wolfId, posIndexInState) for wolfId in range(1, numOfAgent)]

        minDistance = 400
        rolloutHeuristics = [reward.HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfPos, getSheepPos, minDistance)
                             for getWolfPos in getWolvesPoses]

        def rolloutHeuristic(state): return np.mean([rolloutHeuristic(state)
                                                     for rolloutHeuristic in rolloutHeuristics])

        # random rollout policy
        def rolloutPolicy(state): return wolfIndividualActionSpace[np.random.choice(range(numWofActionSpace))]

        def transitInRollout(state, individualAction): return transit(state, np.concatenate([[maxFromDistribution(sheepPolicy(state)), individualAction],
                                                                                             np.array(wolfCentralControlActionSpace[np.random.choice(range(numWolfCentralControlActionSpace))])[actionIndexesInCentralControl]]))

        maxRolloutSteps = 5
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInRollout, rewardFunction, isTerminal, rolloutHeuristic)

        wolfIndividualPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        def policyInPlay(state): return [sheepPolicy(state), wolfIndividualPolicy(state), wolvesRoughPolicy(state)]
        chooseActionList = [sampleFromDistribution, maxFromDistribution, sampleFromDistribution]

        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['yellow'], THECOLORS['red']]
            circleSize = 10

            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, posIndexInState, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        # Sample Trajectory
        reset = Reset(xBoundary, yBoundary, numOfAgent)
        individualWolfId = 1
        centralControlActionIndex = 2

        def transitInPlay(state, action): return transit(state,
                                                         np.concatenate([[np.array(action)[sheepId], np.array(action)[individualWolfId]],
                                                                         np.array(action[centralControlActionIndex])[actionIndexesInCentralControl]]))

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitInPlay, isTerminal, reset, chooseActionList, render, renderOn)
        trajectories = [sampleTrajectory(policyInPlay) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
