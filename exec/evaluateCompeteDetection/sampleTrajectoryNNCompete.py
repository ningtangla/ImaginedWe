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
from src.MDPChasing.policy import RandomPolicy, HeatSeekingDiscreteStochasticPolicy
from src.MDPChasing.state import getStateOrActionFirstPersonPerspective, getStateOrActionThirdPersonPerspective
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution
from src.mathTools.soft import SoftMax
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, ApproximateValue, restoreVariables
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.intention import CreateIntentionSpaceGivenSelfId, CalIntentionValueGivenState, AdjustIntentionPriorGivenValueOfState, UpdateIntention
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsPolicyLikelihood, InferOneStep
from src.generateAction.imaginedWeSampleAction import PolicyForUncommittedAgent, PolicyForCommittedAgent, GetActionFromJointActionDistribution, \
        HierarchyPolicyForCommittedAgent, SampleIndividualActionGivenIntention, \
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
        numSheep = parameters['numSheep']
        valuePriorEndTime = parameters['valuePriorEndTime']
        otherCompeteRate = parameters['otherCompeteRate']
        competeDetectionRate = parameters['competeDetectionRate'] 
        
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

        maxRunningSteps = 100
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
        softParameterInPlanningForSheep = 12.5
        softPolicyInPlanningForSheep = SoftDistribution(softParameterInPlanningForSheep)
        softenSheepPolicy = lambda relativeAgentsStatesForSheepPolicy: softPolicyInPlanningForSheep(sheepPolicy(relativeAgentsStatesForSheepPolicy))

        sheepChooseActionMethod = sampleFromDistribution
        sheepSampleActions = [SampleActionOnFixedIntention(selfId, possibleWolvesIds, softenSheepPolicy, sheepChooseActionMethod) for selfId in possibleSheepIds]

	# Wolves Part

        # Percept Action For Inference
        perceptAction = lambda action: action
        
        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        numWolvesStateSpaces = [2 * (numInWe + 1) 
                for numInWe in range(2, numWolves + 1)]
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]
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
                'agentId='+str(len(wolfIndividualActionSpace) * np.sum([10**_ for _ in
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
                getStateOrActionThirdPersonPerspective)
        concernedAgentsIds = [2]
        calCommittedAgentsPolicyLikelihood = CalCommittedAgentsPolicyLikelihood(concernedAgentsIds, policyForCommittedAgentsInInference)
        
        numCompeteActionSpace = len(wolfIndividualActionSpace)
        generateWolvesCompeteModels = [GenerateModel(numStateSpace, numCompeteActionSpace, regularizationFactor) 
            for numStateSpace in numWolvesStateSpaces]
        initWolvesCompeteModels =[generateWolvesCompeteModel(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths, 
                resBlockSize, initializationMethod, dropoutRate) for generateWolvesCompeteModel in generateWolvesCompeteModels] 
        wolvesCompeteModelPaths = [os.path.join('..', '..', 'data', 'preTrainModel', 
                'agentId=-1.'+str(numWolves)+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations='+str(NNNumSimulations)+'_trainSteps=50000') 
                for numInWe in range(2, numWolves + 1)]
        wolvesCompeteNNModels = [restoreVariables(initWolvesCompeteModel, wolvesModelPath) 
                for initWolvesCompeteModel, wolvesModelPath in zip(initWolvesCompeteModels, wolvesCompeteModelPaths)]
        wolvesCompetePolicies = [ApproximatePolicy(NNModel, wolfIndividualActionSpace) 
                for NNModel in wolvesCompeteNNModels] 
        wolvesCompetePolicy = wolvesCompetePolicies[numWolves - 2]

        policyForUncommittedAgentsInInference = PolicyForUncommittedAgent(possibleWolvesIds, wolvesCompetePolicy, 
                softPolicyInInference, getStateOrActionFirstPersonPerspective)
        calUncommittedAgentsPolicyLikelihood = CalUncommittedAgentsPolicyLikelihood(possibleWolvesIds, 
                concernedAgentsIds, policyForUncommittedAgentsInInference)

        # Joint Likelihood
        calJointLikelihood = lambda intention, state, perceivedAction: calCommittedAgentsPolicyLikelihood(intention, state, perceivedAction) * \
                calUncommittedAgentsPolicyLikelihood(intention, state, perceivedAction)

        wolvesValueListBasedOnNumAgentsInWe = [ApproximateValue(NNModel) 
                for NNModel in wolvesCentralControlNNModels]
        calIntentionValueGivenState = CalIntentionValueGivenState(wolvesValueListBasedOnNumAgentsInWe)
        softParamterForValue = 0.01
        softValueToBuildDistribution = SoftMax(softParamterForValue)
        adjustIntentionPriorGivenValueOfState = AdjustIntentionPriorGivenValueOfState(calIntentionValueGivenState, softValueToBuildDistribution)
        
        # Sample and Save Trajectory
        trajectoriesWithIntentionDists = []
        for trajectoryId in range(self.numTrajectories):
	
            # Intention Prior For inference
            otherWolfPossibleIntentionSpaces = {0: [(0, (1, 2))], 1: [(0, ())]} 
            otherIntentionType = np.random.choice([1, 0], p = [otherCompeteRate, 1 - otherCompeteRate])
            otherWolfIntentionSpace = otherWolfPossibleIntentionSpaces[otherIntentionType]
            selfPossibleIntentionSpaces = {0: [(0, (1, 2))], 0.5: [(0, (1, 2)), (0, ())], 1: [(0, ())]}
            selfWolfIntentionSpace = selfPossibleIntentionSpaces[competeDetectionRate]
            intentionSpacesForAllWolves = [selfWolfIntentionSpace, otherWolfIntentionSpace]
            wolvesIntentionPriors = [{tuple(intention): 1/len(allPossibleIntentionsOneWolf) for intention in allPossibleIntentionsOneWolf} 
                    for allPossibleIntentionsOneWolf in intentionSpacesForAllWolves]        
            # Infer and update Intention
            variablesForAllWolves = [[intentionSpace] for intentionSpace in intentionSpacesForAllWolves]
            jointHypothesisSpaces = [pd.MultiIndex.from_product(variables, names=['intention']) for variables in variablesForAllWolves]
            concernedHypothesisVariable = ['intention']
            priorDecayRate = 1
            softPrior = SoftDistribution(priorDecayRate)
            inferIntentionOneStepList = [InferOneStep(jointHypothesisSpace, concernedHypothesisVariable, 
                calJointLikelihood, softPrior) for jointHypothesisSpace in jointHypothesisSpaces]

            chooseIntention = sampleFromDistribution
            updateIntentions = [UpdateIntention(intentionPrior, valuePriorEndTime, adjustIntentionPriorGivenValueOfState, 
                perceptAction, inferIntentionOneStep, chooseIntention) 
                for intentionPrior, inferIntentionOneStep in zip(wolvesIntentionPriors, inferIntentionOneStepList)]

            # reset intention and adjuste intention prior attributes tools for multiple trajectory
            intentionResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
            intentionResetAttributeValues = [dict(zip(intentionResetAttributes, [0, None, None, intentionPrior, [intentionPrior]]))
                    for intentionPrior in wolvesIntentionPriors]
            resetIntentions = ResetObjects(intentionResetAttributeValues, updateIntentions)
            returnAttributes = ['formerIntentionPriors']
            getIntentionDistributions = GetObjectsValuesOfAttributes(returnAttributes, updateIntentions)
            attributesToRecord = ['lastAction']
            recordActionForUpdateIntention = RecordValuesForObjects(attributesToRecord, updateIntentions) 

            # Wovels Generate Action
            softParameterInPlanning = 2.5
            softPolicyInPlanning = SoftDistribution(softParameterInPlanning)
            policyForCommittedAgentInPlanning = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWe, softPolicyInPlanning,
                    getStateOrActionThirdPersonPerspective)
            
            policyForUncommittedAgentInPlanning = PolicyForUncommittedAgent(possibleWolvesIds, wolvesCompetePolicy, softPolicyInPlanning,
                    getStateOrActionFirstPersonPerspective)
            
            wolfChooseActionMethod = sampleFromDistribution
            getSelfActionThirdPersonPerspective = lambda weIds, selfId : list(weIds).index(selfId)
            chooseCommittedAction = GetActionFromJointActionDistribution(wolfChooseActionMethod, getSelfActionThirdPersonPerspective)
            chooseUncommittedAction = sampleFromDistribution
            wolvesSampleIndividualActionGivenIntentionList = [SampleIndividualActionGivenIntention(selfId, policyForCommittedAgentInPlanning, 
                policyForUncommittedAgentInPlanning, chooseCommittedAction, chooseUncommittedAction) 
                for selfId in possibleWolvesIds]

            wolvesSampleActions = [SampleActionOnChangableIntention(updateIntention, wolvesSampleIndividualActionGivenIntention) 
                    for updateIntention, wolvesSampleIndividualActionGivenIntention in zip(updateIntentions, wolvesSampleIndividualActionGivenIntentionList)]
            allIndividualSampleActions = sheepSampleActions + wolvesSampleActions
            sampleActionMultiAgent = SampleActionMultiagent(allIndividualSampleActions, recordActionForUpdateIntention)
            trajectory = sampleTrajectory(sampleActionMultiAgent)
            intentionDistributions = getIntentionDistributions()
            trajectoryWithIntentionDists = [tuple(list(SASRPair) + list(intentionDist)) 
                    for SASRPair, intentionDist in zip(trajectory, intentionDistributions)]
            trajectoriesWithIntentionDists.append(tuple(trajectoryWithIntentionDists)) 
            resetIntentions()
            #print(intentionDistributions[-1], otherCompeteRate)
        trajectoryFixedParameters = {'sheepPolicySoft': softParameterInPlanningForSheep, 'wolfPolicySoft': softParameterInPlanning,
                'maxRunningSteps': maxRunningSteps, 'competePolicy': 'NNCompete', 'NNNumSimulations':NNNumSimulations}
        self.saveTrajectoryByParameters(trajectoriesWithIntentionDists, trajectoryFixedParameters, parameters)
        print(np.mean([len(tra) for tra in trajectoriesWithIntentionDists]))

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [2] # temp just 2
    manipulatedVariables['numSheep'] = [1] # temp just 1
    manipulatedVariables['valuePriorEndTime'] = [-100]
    manipulatedVariables['otherCompeteRate'] = [0.0, 0.5, 1.0] # 0 never compete, 1 always compete
    manipulatedVariables['competeDetectionRate'] = [0.5, 1.0] # 0 never detect compete, 1 only detect compete
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateCompeteDetection',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = lambda trajectoryFixedParameters: GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, trajectoryFixedParameters, parameters: saveToPickle(trajectories, getTrajectorySavePath(trajectoryFixedParameters)(parameters))
   
    numTrajectories = 200
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

if __name__ == '__main__':
    main()
