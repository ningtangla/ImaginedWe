import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import scipy.stats 
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp
import math 

from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, \
IsTerminal, InterpolateOneFrame, TransitWithTerminalCheckOfInterpolation
from src.MDPChasing.reward import RewardFunctionByTerminal
from src.MDPChasing.trajectory import ForwardOneStep, SampleTrajectory
from src.MDPChasing.policy import RandomPolicy
from src.MDPChasing.state import getStateFirstPersonPerspective, getStateThirdPersonPerspective
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution
from src.mathTools.soft import SoftMax
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, ApproximateValue, restoreVariables
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.intention import CreateIntentionSpaceGivenSelfId, CalIntentionValueGivenState, AdjustIntentionPriorGivenValueOfState, UpdateIntention
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsPolicyLikelihood, InferOneStep
from src.generateAction.imaginedWeSampleAction import PolicyForUncommittedAgent, PolicyForCommittedAgent, SampleIndividualActionGivenIntention, \
        SampleActionOnChangableIntention, SampleActionOnFixedIntention, SampleActionMultiagent
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.sampleTrajectoryTools.evaluation import ComputeStatistics

class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)
        numWolves = parameters['numWolves']
        numSheep = 1
        
        ## MDP Env  
	# state is all multi agent state # action is all multi agent action
        xBoundary = [0,600]
        yBoundary = [0,600]
        numOfAgent = numWolves + numSheep
        reset = Reset(xBoundary, yBoundary, numOfAgent)

        possibleSheepIds = list(range(numSheep))
        possibleWolvesIds = list(range(numSheep, numSheep + numWolves))
        getSheepStatesFromAll = lambda state: np.array(state)[possibleSheepIds]
        getWolvesStatesFromAll = lambda state: np.array(state)[possibleWolvesIds]
        killzoneRadius = 50
        isTerminal = IsTerminal(killzoneRadius, getSheepStatesFromAll, getWolvesStatesFromAll)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        interpolateOneFrame = InterpolateOneFrame(stayInBoundaryByReflectVelocity)
        numFramesToInterpolate = 3
        transit = TransitWithTerminalCheckOfInterpolation(numFramesToInterpolate, interpolateOneFrame, isTerminal)

        maxRunningSteps = 52
        timeCost = 1/maxRunningSteps
        terminalBonus = 1
        rewardFunction = RewardFunctionByTerminal(timeCost, terminalBonus, isTerminal)

        forwardOneStep = ForwardOneStep(transit, rewardFunction)
        sampleTrajectory = SampleTrajectory(maxRunningSteps, isTerminal, reset, forwardOneStep)

        ## MDP Policy
	# Sheep Part

	# Sheep Policy Function
        numSheepPolicyStateSpace = 2 * (numWolves + 1)
        sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 12
        sheepIndividualActionSpace = list(map(tuple, np.array(sheepActionSpace) * preyPowerRatio))
        numSheepActionSpace = len(sheepIndividualActionSpace)
        regularizationFactor = 1e-4
        generateSheepModel = GenerateModel(numSheepPolicyStateSpace, numSheepActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        sheepNNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepModel = generateSheepModel(sharedWidths * sheepNNDepth, actionLayerWidths, valueLayerWidths, 
                resBlockSize, initializationMethod, dropoutRate)
        sheepModelPath = os.path.join('..', '..', 'data', 'preTrainModel',
                'agentId=0.'+str(numWolves)+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=110_trainSteps=50000')
        sheepNNModel = restoreVariables(initSheepModel, sheepModelPath)
        sheepPolicy = ApproximatePolicy(sheepNNModel, sheepIndividualActionSpace)

        # Sheep Generate Action
        softParameterInPlanningForSheep = 2.5
        softPolicyInPlanningForSheep = SoftDistribution(softParameterInPlanningForSheep)
        softenSheepPolicy = lambda relativeAgentsStatesForSheepPolicy: softPolicyInPlanningForSheep(sheepPolicy(relativeAgentsStatesForSheepPolicy))

        sheepChooseActionMethod = sampleFromDistribution
        sheepSampleActions = [SampleActionOnFixedIntention(selfId, possibleWolvesIds, sheepPolicy, sheepChooseActionMethod) for selfId in possibleSheepIds]

	# Wolves Part

        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        numWolvesStateSpaces = [2 * (numInWe + 1) 
                for numInWe in range(2, numWolves + 1)]
        actionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10)]
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
        wolvesCentralControlPolicies = [ApproximatePolicy(NNModel, actionSpace) 
                for NNModel, actionSpace in zip(wolvesCentralControlNNModels, wolvesCentralControlActionSpaces)] 

        centralControlPolicyListBasedOnNumAgentsInWe = wolvesCentralControlPolicies # 0 for two agents in We, 1 for three agents...
        softParameterInInference = 1
        softPolicyInInference = SoftDistribution(softParameterInInference)
        policyForCommittedAgentsInInference = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWe, softPolicyInInference,
                getStateThirdPersonPerspective)
        calCommittedAgentsPolicyLikelihood = CalCommittedAgentsPolicyLikelihood(policyForCommittedAgentsInInference)
        
        wolfLevel2ActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]
        wolfLevel2IndividualActionSpace = list(map(tuple, np.array(wolfLevel2ActionSpace) * predatorPowerRatio))
        wolfLevel2CentralControlActionSpace = list(it.product(wolfLevel2IndividualActionSpace))
        numWolfLevel2ActionSpace = len(wolfLevel2CentralControlActionSpace)
        regularizationFactor = 1e-4
        generatewolfLevel2Models = [GenerateModel(numStateSpace, numWolfLevel2ActionSpace, regularizationFactor) for numStateSpace in numWolvesStateSpaces]
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        wolfLevel2NNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initwolfLevel2Models = [generatewolfLevel2Model(sharedWidths * wolfLevel2NNDepth, actionLayerWidths, valueLayerWidths, 
                resBlockSize, initializationMethod, dropoutRate) for generatewolfLevel2Model in generatewolfLevel2Models]
        wolfLevel2ModelPaths = [os.path.join('..', '..', 'data', 'preTrainModel', 
                'agentId=1.'+str(numInWe)+'_depth=9_hierarchy=2_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations='+str(NNNumSimulations)+'_trainSteps=50000') 
                for numInWe in range(2, numWolves + 1)]
        wolfLevel2NNModels = [restoreVariables(initwolfLevel2Model, wolfLevel2ModelPath)
                for initwolfLevel2Model, wolfLevel2ModelPath in zip(initwolfLevel2Models, wolfLevel2ModelPaths)]
        wolfLevel2Policies = [ApproximatePolicy(wolfLevel2NNModel, wolfLevel2CentralControlActionSpace) 
                for wolfLevel2NNModel in wolfLevel2NNModels]
        level2PolicyListBasedOnNumAgentsInWe = wolfLevel2Policies # 0 for two agents in We, 1 for three agents...

        softPolicy = SoftDistribution(2.5)
        totalInSmallRangeFlags = []
        for trial in range(self.numTrajectories):
            state = reset()
            while isTerminal(state):
                state = reset()

            jointActions = sampleFromDistribution(softPolicy(wolvesCentralControlPolicies[numWolves - 2](state)))

            hierarchyActions = []
            weIds = [list(range(numSheep, numWolves + numSheep)) for _ in range(numWolves)]
            for index in range(numWolves):
                weId = weIds[index].copy()
                weId.insert(0, weId.pop(index))
                relativeId = [0] + weId
                action = sampleFromDistribution(softPolicy(wolfLevel2Policies[numWolves - 2](state[relativeId])))
                hierarchyActions.append(action)

            reasonableActionRange = [int(np.linalg.norm(np.array(jointAction) - np.array(hierarchyAction)) <= 8 * predatorPowerRatio)
                    for jointAction, hierarchyAction in zip(jointActions, hierarchyActions) if jointAction != (0, 0) and hierarchyAction != (0, 0)]
            totalInSmallRangeFlags = totalInSmallRangeFlags + reasonableActionRange
        inSmallRangeRateMean = np.mean(totalInSmallRangeFlags)
        return inSmallRangeRateMean

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [2, 3]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'diagnoseAnalysis',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = lambda trajectoryFixedParameters: GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, trajectoryFixedParameters, parameters: saveToPickle(trajectories, getTrajectorySavePath(trajectoryFixedParameters)(parameters))
   
    numTrajectories = 10000
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, saveTrajectoryByParameters)
    inSmallRangeRateMeans = [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]
    toSplitFrame['inSmallRangeRate'] = inSmallRangeRateMeans
    toSplitFrame.plot.bar(y = 'inSmallRangeRate', ylim = (0, 1))
    plt.show()
    plt.suptitle('Is hierarchical planned action in small range?')

if __name__ == '__main__':
    main()
