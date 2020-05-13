import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))
import time
import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp

from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from src.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from src.MDPChasing.reward import RewardFunctionCompete


class TrainModelForConditions:
    def __init__(self, trainIntervelIndexes, trainStepsIntervel, trainData, NNModel, getTrain, getModelSavePath):
        self.trainIntervelIndexes = trainIntervelIndexes
        self.trainStepsIntervel = trainStepsIntervel
        self.trainData = trainData
        self.NNModel = NNModel
        self.getTrain = getTrain
        self.getModelSavePath = getModelSavePath

    def __call__(self, parameters):
        print(parameters)
        miniBatchSize = parameters['miniBatchSize']
        learningRate = parameters['learningRate']

        model = self.NNModel
        train = self.getTrain(miniBatchSize, learningRate)
        parameters.update({'trainSteps': 0})
        modelSavePath = self.getModelSavePath(parameters)
        saveVariables(model, modelSavePath)

        for trainIntervelIndex in self.trainIntervelIndexes:
            parameters.update({'trainSteps': trainIntervelIndex * self.trainStepsIntervel})
            modelSavePath = self.getModelSavePath(parameters)
            if not os.path.isfile(modelSavePath + '.index'):
                trainedModel = train(model, self.trainData)
                saveVariables(trainedModel, modelSavePath)
            else:
                trainedModel = restoreVariables(model, modelSavePath)
            model = trainedModel


def trainOneCondition(manipulatedVariables):
    depth = int(manipulatedVariables['depth'])
    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'trainLevel2IndividualActionPolicy', '2Wolves', 'trajectories')

    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
    dataSetMaxRunningSteps = 50
    dataSetNumSimulations = 300
    killzoneRadius = 80
    agentId = 1
    wolvesId = 1
    dataSetFixedParameters = {'agentId': agentId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'killzoneRadius': killzoneRadius}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    # MDP Env
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    numOfAgent = 3
    reset = Reset(xBoundary, yBoundary, numOfAgent)

    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

    reset = Reset(xBoundary, yBoundary, numOfAgent)

    possiblePreyIds = [0]
    possiblePredatorIds = list(range(1, numOfAgent))
    posIndexInState = [0, 1]
    getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
    getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
    isTerminal = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)

    playAliveBonus = -1 / dataSetMaxRunningSteps
    playDeathPenalty = 1
    playKillzoneRadius = killzoneRadius

    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, isTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 12
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

    predatorPowerRatio = 8
    wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    numActionSpace = len(wolfActionSpace)

    actionIndex = 1
    actionToOneHot = ActionToOneHot(wolfActionSpace)

    def getTerminalActionFromTrajectory(trajectory): return trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, wolvesId)

    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
    loadedTrajectories = loadTrajectories(parameters={})
    # print(loadedTrajectories[0])

    def filterState(timeStep): return (timeStep[0][0:numOfAgent], timeStep[1], timeStep[2])  # !!? magic
    trajectories = [[filterState(timeStep) for timeStep in trajectory] for trajectory in loadedTrajectories]
    print(len(trajectories))

    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))

    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]
    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]

    # neural network init and save path
    numStateSpace = 2 * numOfAgent
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]

    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    sheepNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

    initTimeStep = 0
    valueIndex = 3
    trainDataMeanAccumulatedReward = np.mean([tra[initTimeStep][valueIndex] for tra in valuedTrajectories])
    print(trainDataMeanAccumulatedReward)

    # function to train NN model
    terminalThreshold = 1e-10
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    initCoeff = (initActionCoeff, initValueCoeff)
    afterActionCoeff = 1
    afterValueCoeff = 1
    afterCoeff = (afterActionCoeff, afterValueCoeff)

    def terminalController(evalDict, numSteps): return False
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)
    reportInterval = 10000
    trainStepsIntervel = 10000
    trainReporter = TrainReporter(trainStepsIntervel, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1

    def learningRateModifier(learningRate): return LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)

    def getTrainNN(batchSize, learningRate): return Train(trainStepsIntervel, batchSize, sampleData, learningRateModifier(learningRate), terminalController, coefficientController, trainReporter)

    # get path to save trained models
    NNModelFixedParameters = {'agentId': agentId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'hierarchy': 2}

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'trainLevel2IndividualActionPolicy', '2Wolves', 'trainedResNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

    # function to train models
    numOfTrainStepsIntervel = 6
    trainIntervelIndexes = list(range(numOfTrainStepsIntervel))
    trainModelForConditions = TrainModelForConditions(trainIntervelIndexes, trainStepsIntervel, trainData, sheepNNModel, getTrainNN, getNNModelSavePath)
    trainModelForConditions(manipulatedVariables)


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['depth'] = [9]
    manipulatedVariables['miniBatchSize'] = [256]
    manipulatedVariables['learningRate'] = [1e-4]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    trainPool = mp.Pool(numCpuToUse)

    startTime = time.time()

    trainPool.map(trainOneCondition, parametersAllCondtion)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
