import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import random
import json
import numpy as np
import scipy.stats
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp
import math
from anytree import AnyNode as Node
import pygame as pg

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal, InterpolateState
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory, SampleTrajectoryWithRender
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables, ApproximateValue
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.inference import CalPolicyLikelihood, InferOneStep, InferOnTrajectory
from src.evaluation import ComputeStatistics
from src.valueFromNode import EstimateValueFromNode
from src.MDPChasing import reward


def sortSelfIdFirst(weId, selfId):
    weId.insert(0, weId.pop(selfId))
    return weId


class MCTS:
    def __init__(self, numSimulation, selectChild, expand, estimateValue, backup, outputDistribution, mctsRender, mctsRenderOn):
        self.numSimulation = numSimulation
        self.selectChild = selectChild
        self.expand = expand
        self.estimateValue = estimateValue
        self.backup = backup
        self.outputDistribution = outputDistribution
        self.mctsRender = mctsRender
        self.mctsRenderOn = mctsRenderOn

    def __call__(self, currentState):
        backgroundScreen = None
        root = Node(id={None: currentState}, numVisited=0, sumValue=0, isExpanded=False)
        root = self.expand(root)

        for exploreStep in range(self.numSimulation):
            currentNode = root
            nodePath = [currentNode]

            while currentNode.isExpanded:
                nextNode = self.selectChild(currentNode)
                if self.mctsRenderOn:
                    backgroundScreen = self.mctsRender(currentNode, nextNode, backgroundScreen)
                nodePath.append(nextNode)
                currentNode = nextNode

            leafNode = self.expand(currentNode)
            value = self.estimateValue(leafNode)
            self.backup(value, nodePath)
        actionDistribution = self.outputDistribution(root)
        return actionDistribution


class MCTSRender():
    def __init__(self, numAgent, screen, surfaceWidth, surfaceHeight, screenColor, circleColorList, mctsLineColor, circleSize, saveImage, saveImageDir, drawState, scalePos):
        self.numAgent = numAgent
        self.screen = screen
        self.surfaceWidth = surfaceWidth
        self.surfaceHeight = surfaceHeight
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.mctsLineColor = mctsLineColor
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir
        self.drawState = drawState
        self.scalePos = scalePos

    def __call__(self, currNode, nextNode, backgroundScreen):
        parentNumVisit = currNode.numVisited
        parentValueToTal = currNode.sumValue
        originalState = list(currNode.id.values())[0]
        poses = self.scalePos(originalState)

        childNumVisit = nextNode.numVisited
        childValueToTal = nextNode.sumValue
        originalNextState = list(nextNode.id.values())[0]
        nextPoses = self.scalePos(originalNextState)

        if not os.path.exists(self.saveImageDir):
            os.makedirs(self.saveImageDir)

        lineWidth = math.ceil(0.1 * (nextNode.numVisited + 1))
        surfaceToDraw = pg.Surface((self.surfaceWidth, self.surfaceHeight))
        surfaceToDraw.fill(self.screenColor)
        if backgroundScreen == None:
            backgroundScreen = self.drawState(poses, self.circleColorList)
            if self.saveImage == True:
                for numStaticImage in range(120):
                    filenameList = os.listdir(self.saveImageDir)
                    pg.image.save(self.screen, self.saveImageDir + '/' + str(len(filenameList)) + '.png')

        surfaceToDraw.set_alpha(180)
        surfaceToDraw.blit(backgroundScreen, (0, 0))
        self.screen.blit(surfaceToDraw, (0, 0))

        pg.display.flip()
        pg.time.wait(1)

        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit

            for i in range(self.numAgent):
                oneAgentPosition = np.array(poses[i])
                oneAgentNextPosition = np.array(nextPoses[i])
                if i != 0:  # draw mcts line for wolves
                    pg.draw.line(surfaceToDraw, self.mctsLineColor, [np.int(oneAgentPosition[0]), np.int(oneAgentPosition[1])], [np.int(oneAgentNextPosition[0]), np.int(oneAgentNextPosition[1])], lineWidth)
                if i == 1:
                    agentPos = [np.int(np.array(nextPoses[1])[0]), np.int(np.array(nextPoses[1])[1])]
                    pg.draw.circle(surfaceToDraw, [255, 255, 255], agentPos, 8)

                pg.draw.circle(surfaceToDraw, self.circleColorList[i], [np.int(oneAgentNextPosition[0]), np.int(oneAgentNextPosition[1])], self.circleSize)

            self.screen.blit(surfaceToDraw, (0, 0))
            pg.display.flip()
            pg.time.wait(1)

            if self.saveImage == True:
                filenameList = os.listdir(self.saveImageDir)
                pg.image.save(self.screen, self.saveImageDir + '/' + str(len(filenameList)) + '.png')
        return self.screen


class Render():
    def __init__(self, numOfAgent, posIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir):
        self.numOfAgent = numOfAgent
        self.posIndex = posIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self, state, timeStep):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
            self.screen.fill(self.screenColor)
            for i in range(self.numOfAgent):
                agentPos = state[i][self.posIndex]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(
                    agentPos[0]), np.int(agentPos[1])], self.circleSize)
            pg.display.flip()
            pg.time.wait(100)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                for numStaticImage in range(120):
                    filenameList = os.listdir(self.saveImageDir)
                    pg.image.save(self.screen, self.saveImageDir + '/' + str(len(filenameList)) + '.png')


class DrawBackground:
    def __init__(self, screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.lineColor = lineColor
        self.lineWidth = lineWidth

    def __call__(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    exit()
        self.screen.fill(self.screenColor)
        rectPos = [self.xBoundary[0], self.yBoundary[0], self.xBoundary[1], self.yBoundary[1]]
        return


class DrawState():
    def __init__(self, screen, circleSize, numOfAgent, positionIndex, drawBackGround):
        self.screen = screen
        self.circleSize = circleSize
        self.numOfAgent = numOfAgent
        self.xIndex, self.yIndex = positionIndex
        self.drawBackGround = drawBackGround

    def __call__(self, state, circleColorList):
        self.drawBackGround()
        for agentIndex in range(self.numOfAgent):
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = circleColorList[agentIndex]
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)

        pg.display.flip()
        pg.time.wait(10)

        return self.screen


class DrawCircleOutside:
    def __init__(self, screen, outsideCircleAgentIds, positionIndex, circleColors, circleSize):
        self.screen = screen
        self.outsideCircleAgentIds = outsideCircleAgentIds
        self.xIndex, self.yIndex = positionIndex
        self.circleColors = circleColors
        self.circleSize = circleSize

    def __call__(self, state):
        for agentIndex in self.outsideCircleAgentIds:
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = tuple(self.circleColors[list(self.outsideCircleAgentIds).index(agentIndex)])
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        return


class ScalePos:
    def __init__(self, positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange):
        self.xIndex, self.yIndex = positionIndex
        self.rawXMin, self.rawXMax = rawXRange
        self.rawYMin, self.rawYMax = rawYRange

        self.scaledXMin, self.scaledXMax = scaledXRange
        self.scaledYMin, self.scaledYMax = scaledYRange

    def __call__(self, state):
        xScale = (self.scaledXMax - self.scaledXMin) / (self.rawXMax - self.rawXMin)
        yScale = (self.scaledYMax - self.scaledYMin) / (self.rawYMax - self.rawYMin)

        adjustX = lambda rawX: (rawX - self.rawXMin) * xScale + self.scaledXMin
        adjustY = lambda rawY: (rawY - self.rawYMin) * yScale + self.scaledYMin

        adjustPosPair = lambda pair: [adjustX(pair[self.xIndex]), adjustY(pair[self.yIndex])]
        agentCount = len(state)

        adjustPos = lambda state: [adjustPosPair(state[agentIndex]) for agentIndex in range(agentCount)]
        adjustedPoses = adjustPos(state)

        return adjustedPoses


def main():
    DEBUG = 1
    renderOn = 1
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 0
        endSampleIndex = 2
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
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'generateGuidedMCTSWeWithRollout', 'OneLeveLPolicy', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    numOneWolfActionSpace = 5
    NNNumSimulations = 200  # 300 with distance Herustic; 200 without distanceHerustic
    numWolves = 2
    maxRunningSteps = 101
    softParameterInPlanning = 2.5
    sheepPolicyName = 'sampleNNPolicy'
    wolfPolicyName = 'maxNNPolicy'
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName, 'NNNumSimulations': NNNumSimulations,
                                 'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'numOneWolfActionSpace': numOneWolfActionSpace, 'numWolves': numWolves}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, trajectoryFixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        # MDP Env
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        numSheep = 2
        numOfAgent = numWolves + numSheep
        reset = Reset(xBoundary, yBoundary, numOfAgent)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

        possiblePreyIds = list(range(numSheep))
        possiblePredatorIds = list(range(numSheep, numSheep + numWolves))
        posIndexInState = [0, 1]
        getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
        getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
        killzoneRadius = 50
        isTerminalInPlay = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)

        # MDP Policy
        sheepImagindWeIntentionPrior = {tuple(range(numSheep, numSheep + numWolves)): 1}
        wolfImaginedWeIntentionPrior = {(sheepId, ): 1 / numSheep for sheepId in range(numSheep)}
        imaginedWeIntentionPriors = [sheepImagindWeIntentionPrior] * numSheep + [wolfImaginedWeIntentionPrior] * numWolves

        # Percept Action
        imaginedWeIdCopys = [list(range(numSheep, numSheep + numWolves)) for _ in range(numWolves)]
        imaginedWeIdsForInferenceSubject = [sortSelfIdFirst(weIdCopy, selfId)
                                            for weIdCopy, selfId in zip(imaginedWeIdCopys, list(range(numWolves)))]

        numStateSpace = 2 * (numSheep + numWolves - 1)
        # actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
        #              (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        actionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)]
        predatorPowerRatio = 8
        wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        mappingActionToAnotherSpace = MappingActionToAnotherSpace(wolfIndividualActionSpace)

        perceptSelfAction = lambda singleAgentAction: mappingActionToAnotherSpace(singleAgentAction)
        perceptOtherAction = lambda singleAgentAction: mappingActionToAnotherSpace(singleAgentAction)
        perceptImaginedWeAction = [PerceptImaginedWeAction(imaginedWeIds, perceptSelfAction, perceptOtherAction)
                                   for imaginedWeIds in imaginedWeIdsForInferenceSubject]
        perceptActionForAll = [lambda action: action] * numSheep + perceptImaginedWeAction

        # Inference of Imagined We
        noInferIntention = lambda intentionPrior, action, perceivedAction: intentionPrior
        sheepUpdateIntentionMethod = noInferIntention

        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        wolfCentralControlActionSpace = list(it.product(wolfIndividualActionSpace, repeat=numWolves))
        numWolvesActionSpace = len(wolfCentralControlActionSpace)
        regularizationFactor = 1e-4
        generateWolfCentralControlModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        wolfNNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initWolfCentralControlModel = generateWolfCentralControlModel(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths,
                                                                      resBlockSize, initializationMethod, dropoutRate)
        wolfModelPath = os.path.join('..', '..', '..', 'data', 'preTrainModel',
                                     'agentId=' + str(numOneWolfActionSpace * np.sum([10**_ for _ in range(numWolves)])) + '_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=' + str(NNNumSimulations) + '_trainSteps=50000')
        wolfCentralControlNNModel = restoreVariables(initWolfCentralControlModel, wolfModelPath)
        wolfCentralControlPolicyGivenIntention = ApproximatePolicy(wolfCentralControlNNModel, wolfCentralControlActionSpace)

        softParameterInInference = 1
        softPolicyInInference = SoftPolicy(softParameterInInference)
        softenWolfCentralControlPolicyGivenIntentionInInference = lambda state: softPolicyInInference(wolfCentralControlPolicyGivenIntention(state))

        getStateForPolicyGivenIntentionInInference = [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in
                                                      imaginedWeIdsForInferenceSubject]

        calPoliciesLikelihood = [CalPolicyLikelihood(getState, softenWolfCentralControlPolicyGivenIntentionInInference)
                                 for getState in getStateForPolicyGivenIntentionInInference]

        # ActionPerception Likelihood
        calActionPerceptionLikelihood = lambda action, perceivedAction: int(np.allclose(np.array(action), np.array(perceivedAction)))

        # Joint Likelihood
        composeCalJointLikelihood = lambda calPolicyLikelihood, calActionPerceptionLikelihood: lambda intention, state, action, perceivedAction: \
            calPolicyLikelihood(intention, state, action) * calActionPerceptionLikelihood(action, perceivedAction)
        calJointLikelihoods = [composeCalJointLikelihood(calPolicyLikelihood, calActionPerceptionLikelihood)
                               for calPolicyLikelihood in calPoliciesLikelihood]

        # Joint Hypothesis Space
        priorDecayRate = 1
        intentionSpace = [(id,) for id in range(numSheep)]
        actionSpaceInInference = wolfCentralControlActionSpace
        variables = [intentionSpace, actionSpaceInInference]
        jointHypothesisSpace = pd.MultiIndex.from_product(variables, names=['intention', 'action'])
        concernedHypothesisVariable = ['intention']
        inferImaginedWe = [InferOneStep(priorDecayRate, jointHypothesisSpace,
                                        concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in calJointLikelihoods]
        updateIntention = [sheepUpdateIntentionMethod] * numSheep + inferImaginedWe
        chooseIntention = sampleFromDistribution

        # Get State of We and Intention
        imaginedWeIdsForAllAgents = [[id] for id in range(numSheep)] + imaginedWeIdsForInferenceSubject
        getStateForPolicyGivenIntentions = [GetStateForPolicyGivenIntention(imaginedWeId)
                                            for imaginedWeId in imaginedWeIdsForAllAgents]

        # NN Policy Given Intention
        sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                            (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 12
        sheepIndividualActionSpace = list(map(tuple, np.array(sheepActionSpace) * preyPowerRatio))
        sheepCentralControlActionSpace = list(it.product(sheepIndividualActionSpace))
        numSheepActionSpace = len(sheepCentralControlActionSpace)
        regularizationFactor = 1e-4
        generateSheepCentralControlModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        sheepNNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepCentralControlModel = generateSheepCentralControlModel(sharedWidths * sheepNNDepth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        sheepModelPath = os.path.join('..', '..', '..', 'data', 'preTrainModel',
                                      'agentId=0.' + str(numWolves) + '_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=100_trainSteps=50000')
        sheepCentralControlNNModel = restoreVariables(initSheepCentralControlModel, sheepModelPath)
        sheepCentralControlPolicyGivenIntention = ApproximatePolicy(sheepCentralControlNNModel, sheepCentralControlActionSpace)

        wolfLevel2ActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                                 (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfLevel2IndividualActionSpace = list(map(tuple, np.array(wolfLevel2ActionSpace) * predatorPowerRatio))
        wolfLevel2CentralControlActionSpace = list(it.product(wolfLevel2IndividualActionSpace))
        numWolfLevel2ActionSpace = len(wolfLevel2CentralControlActionSpace)
        regularizationFactor = 1e-4
        generatewolfLevel2Model = GenerateModel(numStateSpace, numWolfLevel2ActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        wolfLevel2NNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initwolfLevel2Model = generatewolfLevel2Model(sharedWidths * wolfLevel2NNDepth, actionLayerWidths, valueLayerWidths,
                                                      resBlockSize, initializationMethod, dropoutRate)
        wolfLevel2ModelPath = os.path.join('..', '..', '..', 'data', 'preTrainModel',
                                           'agentId=1.' + str(numWolves) + '_depth=9_hierarchy=2_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=' + str(NNNumSimulations) + '_trainSteps=50000')
        wolfLevel2NNModel = restoreVariables(initwolfLevel2Model, wolfLevel2ModelPath)
        wolfLevel2PolicyGivenIntention = ApproximatePolicy(wolfLevel2NNModel, wolfLevel2CentralControlActionSpace)

        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = wolfLevel2PolicyGivenIntention

        # terminal and transit InMCTS
        possiblePreyIdsInMCTS = [0]
        possiblePredatorIdsInMCTS = list(range(1, numWolves + 1))
        getPreyPosInMCTS = GetAgentPosFromState(possiblePreyIdsInMCTS, posIndexInState)
        getPredatorPosInMCTS = GetAgentPosFromState(possiblePredatorIdsInMCTS, posIndexInState)
        isTerminalInMCTS = IsTerminal(killzoneRadius, getPreyPosInMCTS, getPredatorPosInMCTS)
        numFrameToInterpolate = 3
        interpolateStateInMCTS = InterpolateState(3, transit, isTerminalInMCTS)

        wolvesImaginedWeIdInMCTS = list(range(1, 1 + numWolves))
        otherWolvesIdInMCTS = list(range(2, 1 + numWolves))
        actionIndexesInCentralControl = [wolvesImaginedWeIdInMCTS.index(wolfId) for wolfId in otherWolvesIdInMCTS]
        transitInMCTS = lambda state, wolfLevel2Action: interpolateStateInMCTS(state, np.concatenate([sampleFromDistribution(sheepCentralControlPolicyGivenIntention(state)), wolfLevel2Action, np.array(sampleFromDistribution(wolfCentralControlPolicyGivenIntention(state)))[actionIndexesInCentralControl]]))

        # initialize children; expand
        initializeChildren = InitializeChildren(
            wolfLevel2CentralControlActionSpace, transitInMCTS, getActionPrior)
        expand = Expand(isTerminalInMCTS, initializeChildren)

        # Rollout Value
        rolloutHeuristicWeight = 1e-2
        sheepId = 0
        getSheepPos = GetAgentPosFromState(sheepId, posIndexInState)
        getWolvesPoses = [GetAgentPosFromState(wolfId, posIndexInState) for wolfId in range(1, numWolves + 1)]

        minDistance = 400
        rolloutHeuristics = [reward.HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfPos, getSheepPos, minDistance)
                             for getWolfPos in getWolvesPoses]

        rolloutHeuristic = lambda state: np.mean([rolloutHeuristic(state)
                                                  for rolloutHeuristic in rolloutHeuristics])

        rolloutPolicy = lambda state: wolfLevel2CentralControlActionSpace[np.random.choice(range(numWolfLevel2ActionSpace))]

        aliveBonus = -1 / maxRunningSteps
        deathPenalty = 1
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminalInMCTS)

        def transitInRollout(state, wolfLevel2Action):
            return interpolateStateInMCTS(state, np.concatenate([sampleFromDistribution(sheepCentralControlPolicyGivenIntention(state)), wolfLevel2Action,
                                                                 np.array(wolfCentralControlActionSpace[np.random.choice(range(numWolvesActionSpace))])[actionIndexesInCentralControl]]))

        maxRolloutSteps = 5
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transitInRollout, rewardFunction, isTerminalInMCTS, rolloutHeuristic)

        import pygame as pg
        from pygame.color import THECOLORS
        screenWidth = 600
        screenHeight = 600

        screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        leaveEdgeSpace = 195
        lineWidth = 4
        circleSize = 10
        xBoundary = [leaveEdgeSpace, screenWidth - leaveEdgeSpace * 2]
        yBoundary = [leaveEdgeSpace, screenHeight - leaveEdgeSpace * 2]
        screenColor = THECOLORS['black']
        lineColor = THECOLORS['white']

        posIndex = [0, 1]
        drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
        numOfAgentInMCTS = numOfAgent - 1

        outsideCircleSize = 15
        outsideCircleColor = np.array([[255, 0, 0]] * numWolves)
        drawState = DrawState(screen, circleSize, numOfAgentInMCTS, posIndex, drawBackground)

        screenColor = THECOLORS['black']
        circleColorListInMCTS = [THECOLORS['green'], THECOLORS['red'], THECOLORS['red']]
        mctsLineColor = np.array([240, 240, 240, 180])
        circleSizeForMCTS = int(0.6 * circleSize)

        circleSize = 10
        rawXRange = [0, 600]
        rawYRange = [0, 600]
        scaledXRange = [0, 600]
        scaledYRange = [0, 600]
        scalePos = ScalePos(posIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)

        saveImage = True
        saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
        if not os.path.exists(saveImageDir):
            os.makedirs(saveImageDir)

        mctsRenderOn = True
        mctsRender = MCTSRender(numOfAgentInMCTS, screen, screenWidth, screenHeight, screenColor, circleColorListInMCTS, mctsLineColor, circleSizeForMCTS, saveImage, saveImageDir, drawState, scalePos)

        numSimulations = 100
        wolfLevel2GuidedMCTSPolicyGivenIntention = MCTS(numSimulations, selectChild, expand, rollout, backup, establishPlainActionDist, mctsRender, mctsRenderOn)

    # final individual polices
        softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
        softSheepParameterInPlanning = softParameterInPlanning
        softSheepPolicyInPlanning = SoftPolicy(softSheepParameterInPlanning)
        softenSheepCentralControlPolicyGivenIntentionInPlanning = lambda state: softSheepPolicyInPlanning(sheepCentralControlPolicyGivenIntention(state))
        softenWolfLevel2GuidedMCTSPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(wolfLevel2GuidedMCTSPolicyGivenIntention(state))
        centralControlPoliciesGivenIntentions = [softenSheepCentralControlPolicyGivenIntentionInPlanning] * numSheep + [softenWolfLevel2GuidedMCTSPolicyGivenIntentionInPlanning] * numWolves
        planningIntervals = [1] * numSheep + [1] * numWolves
        intentionInferInterval = 1
        individualPolicies = [PolicyOnChangableIntention(perceptAction,
                                                         imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention, planningInterval, intentionInferInterval)
                              for perceptAction, imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention, planningInterval
                              in zip(perceptActionForAll, imaginedWeIntentionPriors, getStateForPolicyGivenIntentions,
                                     updateIntention, centralControlPoliciesGivenIntentions, planningIntervals)]

        individualIdsForAllAgents = list(range(numWolves + numSheep))
        actionChoiceMethods = {'sampleNNPolicy': sampleFromDistribution, 'maxNNPolicy': maxFromDistribution}
        chooseCentrolAction = [actionChoiceMethods[sheepPolicyName]] * numSheep + [actionChoiceMethods[wolfPolicyName]] * numWolves
        assignIndividualAction = [AssignCentralControlToIndividual(imaginedWeId, individualId) for imaginedWeId, individualId in
                                  zip(imaginedWeIdsForAllAgents, individualIdsForAllAgents)]
        individualActionMethods = [lambda centrolActionDist: assign(chooseAction(centrolActionDist)) for assign, chooseAction in
                                   zip(assignIndividualAction, chooseCentrolAction)]

        policiesResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
        policiesResetAttributeValues = [dict(zip(policiesResetAttributes, [0, None, None, intentionPrior, [intentionPrior]])) for intentionPrior in
                                        imaginedWeIntentionPriors]
        returnAttributes = ['formerIntentionPriors']
        resetPolicy = ResetPolicy(policiesResetAttributeValues, individualPolicies, returnAttributes)
        attributesToRecord = ['lastAction']
        recordActionForPolicy = RecordValuesForPolicyAttributes(attributesToRecord, individualPolicies)

        # Sample and Save Trajectory
        render = None
        if renderOn:
            circleColorList = [THECOLORS['green'], THECOLORS['green'], THECOLORS['red'], THECOLORS['red']]
            render = Render(numOfAgent, posIndexInState, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        interpolateStateInPlay = InterpolateState(3, transit, isTerminalInPlay)
        transitInPlay = lambda state, action: interpolateStateInPlay(state, action)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitInPlay, isTerminalInPlay, reset, individualActionMethods, resetPolicy,
                                                      recordActionForPolicy, render, renderOn)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]

        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        print([len(traj) for traj in trajectories])


if __name__ == '__main__':
    main()
# ffmpeg -r 120 -f image2 -s 1920x1080 -i  %0d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/ModellingJointInferenceOfPhysicsAndMind/data/demo/demo.mp4
