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
        softParamterForValue = parameters['valuePriorSoftMaxBeta']
        valuePriorEndTime = parameters['valuePriorEndTime']
        
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

        maxRunningSteps = 51
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
        sheepSampleActions = [SampleActionOnFixedIntention(selfId, possibleWolvesIds, softenSheepPolicy, sheepChooseActionMethod) for selfId in possibleSheepIds]

	# Wolves Part

	# Intention Prior For inference
        #createIntentionSpaceGivenSelfId = CreateIntentionSpaceGivenSelfId(possibleSheepIds, possibleWolvesIds)
        #intentionSpacesForAllWolves = [createAllPossibleIntentionsGivenSelfId(wolfId) 
        #        for wolfId in possibleWolvesIds]
        intentionSpacesForAllWolves = [tuple(it.product(possibleSheepIds, [tuple(possibleWolvesIds)])) 
                for wolfId in possibleWolvesIds]
        print(intentionSpacesForAllWolves)
        wolvesIntentionPriors = [{tuple(intention): 1/len(allPossibleIntentionsOneWolf) for intention in allPossibleIntentionsOneWolf} 
                for allPossibleIntentionsOneWolf in intentionSpacesForAllWolves]        
        # Percept Action For Inference
        actionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10)]
        predatorPowerRatio = 8
        wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        mappingActionToAnotherSpace = MappingActionToAnotherSpace(wolfIndividualActionSpace)
        
        perceptSelfAction = lambda singleAgentAction: mappingActionToAnotherSpace(singleAgentAction)
        perceptOtherAction = lambda singleAgentAction: mappingActionToAnotherSpace(singleAgentAction)
        perceptActions = [PerceptImaginedWeAction(selfId, perceptSelfAction, perceptOtherAction) 
                for selfId in possibleWolvesIds]

        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        numWolvesStateSpaces = [2 * (numInWe + 1) 
                for numInWe in range(2, numWolves + 1)]
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
                getStateOrActionThirdPersonPerspective)
        concernedAgentsIds = possibleWolvesIds
        calCommittedAgentsPolicyLikelihood = CalCommittedAgentsPolicyLikelihood(concernedAgentsIds, policyForCommittedAgentsInInference)
        
        randomPolicy = RandomPolicy(wolfIndividualActionSpace)
        policyForUncommittedAgentsInInference = PolicyForUncommittedAgent(possibleWolvesIds, randomPolicy, 
                softPolicyInInference, getStateOrActionFirstPersonPerspective)
        calUncommittedAgentsPolicyLikelihood = CalUncommittedAgentsPolicyLikelihood(concernedAgentsIds, possibleWolvesIds, policyForUncommittedAgentsInInference)

        # Joint Likelihood
        calJointLikelihood = lambda intention, state, perceivedAction: calCommittedAgentsPolicyLikelihood(intention, state, perceivedAction) * \
                calUncommittedAgentsPolicyLikelihood(intention, state, perceivedAction)

        # Infer and update Intention
        variablesForAllWolves = [[intentionSpace] for intentionSpace in intentionSpacesForAllWolves]
        jointHypothesisSpaces = [pd.MultiIndex.from_product(variables, names=['intention']) for variables in variablesForAllWolves]
        concernedHypothesisVariable = ['intention']
        priorDecayRate = 1
        softPrior = SoftDistribution(priorDecayRate)
        inferIntentionOneStepList = [InferOneStep(jointHypothesisSpace, concernedHypothesisVariable, 
            calJointLikelihood, softPrior) for jointHypothesisSpace in jointHypothesisSpaces]

        wolvesValueListBasedOnNumAgentsInWe = [ApproximateValue(NNModel) 
                for NNModel in wolvesCentralControlNNModels]
        calIntentionValueGivenState = CalIntentionValueGivenState(wolvesValueListBasedOnNumAgentsInWe)
        softValueToBuildDistribution = SoftMax(softParamterForValue)
        adjustIntentionPriorGivenValueOfState = AdjustIntentionPriorGivenValueOfState(calIntentionValueGivenState, softValueToBuildDistribution)

        chooseIntention = sampleFromDistribution
        updateIntentions = [UpdateIntention(intentionPrior, valuePriorEndTime, adjustIntentionPriorGivenValueOfState, 
            perceptAction, inferIntentionOneStep, chooseIntention) 
            for perceptAction, intentionPrior, inferIntentionOneStep in zip(perceptActions, wolvesIntentionPriors, inferIntentionOneStepList)]

	# reset intention and adjuste intention prior attributes tools for multiple trajectory
        intentionResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
        intentionResetAttributeValues = [dict(zip(intentionResetAttributes, [0, None, None, intentionPrior, [intentionPrior]]))
                for intentionPrior in wolvesIntentionPriors]
        resetIntentions = ResetObjects(intentionResetAttributeValues, updateIntentions)
        returnAttributes = ['formerIntentionPriors']
        getIntentionDistributions = GetObjectsValuesOfAttributes(returnAttributes, updateIntentions)
        attributesToRecord = ['lastAction']
        recordActionForUpdateIntention = RecordValuesForObjects(attributesToRecord, updateIntentions) 

        numWolvesLevel2StateSpaces = [2 * (numInWe + 1) * 2 - 2 
                for numInWe in range(2, numWolves + 1)]
        wolfLevel2ActionSpace = [-1, 0, 1]
        numWolfLevel2ActionSpace = len(wolfLevel2ActionSpace)
        regularizationFactor = 1e-4
        generatewolfLevel2Models = [GenerateModel(numStateSpace, numWolfLevel2ActionSpace, regularizationFactor) 
                for numStateSpace in numWolvesLevel2StateSpaces]
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
                'agentId=2.'+str(numInWe)+'_depth=9_hierarchy=2_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations='+str(NNNumSimulations)+'_trainSteps=50000') 
                for numInWe in range(2, numWolves + 1)]
        wolfLevel2NNModels = [restoreVariables(initwolfLevel2Model, wolfLevel2ModelPath)
                for initwolfLevel2Model, wolfLevel2ModelPath in zip(initwolfLevel2Models, wolfLevel2ModelPaths)]
        wolfLevel2Policies = [ApproximatePolicy(wolfLevel2NNModel, wolfLevel2ActionSpace) 
                for wolfLevel2NNModel in wolfLevel2NNModels]
        level2PolicyListBasedOnNumAgentsInWe = wolfLevel2Policies # 0 for two agents in We, 1 for three agents...

	# Wovels Generate Action
        composeGetStateFirstPersonPerspectiveForCommittedAgent = \
            lambda selfId: lambda state, goalId, weIds: getStateFirstPersonPerspective(state, goalId, weIds, selfId)  
        getStateFirstPersonPerspectiveForLevel2CommittedPolicyMethods = [composeGetStateFirstPersonPerspectiveForCommittedAgent(selfId) 
                for selfId in possibleWolvesIds]

        softParameterInPlanning = 1.0
        softPolicyInPlanning = SoftDistribution(softParameterInPlanning)
        level1PolicyForCommittedAgentInPlanning = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWe, softPolicyInPlanning,
                getStateOrActionThirdPersonPerspective)

        getWholeJointActionIndex = lambda weIds, selfId: list(range(len(weIds)))
        getRoughJointAction = GetActionFromJointActionDistribution(sampleFromDistribution, getWholeJointActionIndex) 

        fineActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]
        wolfFineActionSpace = list(map(tuple, np.array(fineActionSpace) * predatorPowerRatio))
        
        hierarchyPolicyForCommittedAgentInPlanningList = [HierarchyPolicyForCommittedAgent(numSheep, wolfId, wolfFineActionSpace, 
            level1PolicyForCommittedAgentInPlanning, getRoughJointAction, getStateOrActionFirstPersonPerspective,
            level2PolicyListBasedOnNumAgentsInWe, softPolicyInPlanning) for wolfId in possibleWolvesIds] 

        policyForUncommittedAgentInPlanning = PolicyForUncommittedAgent(possibleWolvesIds, randomPolicy, softPolicyInPlanning,
                getStateOrActionFirstPersonPerspective)

        chooseCommittedAction = lambda actionDistribution, weIds, selfId: sampleFromDistribution(actionDistribution)
        chooseUncommittedAction = sampleFromDistribution
        wolvesSampleIndividualActionGivenIntentionList = [SampleIndividualActionGivenIntention(selfId, policyForCommittedAgentsInPlanning, 
            policyForUncommittedAgentInPlanning, chooseCommittedAction, chooseUncommittedAction) 
            for selfId, policyForCommittedAgentsInPlanning in zip(possibleWolvesIds, hierarchyPolicyForCommittedAgentInPlanningList)]

        # Sample and Save Trajectory
        trajectoriesWithIntentionDists = []
        for trajectoryId in range(self.numTrajectories):
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
            #print(intentionDistributions)
        trajectoryFixedParameters = {'sheepPolicySoft': softParameterInPlanningForSheep, 'wolfPolicySoft': softParameterInPlanning,
                'maxRunningSteps': maxRunningSteps, 'hierarchy': 2, 'NNNumSimulations':NNNumSimulations}
        self.saveTrajectoryByParameters(trajectoriesWithIntentionDists, trajectoryFixedParameters, parameters)
        print(np.mean([len(tra) for tra in trajectoriesWithIntentionDists]))

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [2, 3]
    manipulatedVariables['numSheep'] = [1, 2, 4, 8]
    manipulatedVariables['valuePriorSoftMaxBeta'] = [0.0]
    manipulatedVariables['valuePriorEndTime'] = [-100]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateHierarchyPlanning',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = lambda trajectoryFixedParameters: GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, trajectoryFixedParameters, parameters: saveToPickle(trajectories, getTrajectorySavePath(trajectoryFixedParameters)(parameters))
   
    numTrajectories = 500
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

if __name__ == '__main__':
    main()
