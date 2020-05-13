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

from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.inference import CalPolicyLikelihood, InferOneStep, InferOnTrajectory
from src.evaluation import ComputeStatistics

def sortSelfIdFirst(weId, selfId):
    weId.insert(0, weId.pop(selfId))
    return weId

class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, composeIndividualPoliciesByEvaParameters, composeSampleTrajectory, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.composeIndividualPoliciesByEvaParameters = composeIndividualPoliciesByEvaParameters
        self.composeSampleTrajectory = composeSampleTrajectory
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)
        numWolves = parameters['numWolves']
        
        # MDP Env
        xBoundary = [0,600]
        yBoundary = [0,600]
        numSheep = 2
        getNumOfAgent = lambda numWolves: numWolves + numSheep
        composeReset = lambda numWolves: Reset(xBoundary, yBoundary, getNumOfAgent(numWolves))
        
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

        possiblePreyIds = [0, 1]
        getPossiblePredatorIds = lambda numWolves: list(range(numSheep, numSheep + numWolves))
        posIndexInState = [0, 1]
        getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
        getPredatorPos = lambda numWolves: GetAgentPosFromState(getPossiblePredatorIds(numWolves), posIndexInState)
        killzoneRadius = 30
        getIsTerminal = lambda numWolves: IsTerminal(killzoneRadius, getPreyPos, getPredatorPos(numWolves))

        # MDP Policy
        getSheepImagindWeIntentionPrior = lambda numWolves: {tuple(range(numSheep, numSheep + numWolves)): 1}
        getWolfImaginedWeIntentionPrior = lambda numWolves: {(sheepId, ): 1/numWolves for sheepId in range(numSheep)}
        getImaginedWeIntentionPriors = lambda numWolves: [getSheepImagindWeIntentionPrior(numWolves)] * numSheep + [getWolfImaginedWeIntentionPrior(numWolves)] * numWolves
        # Percept Action
        getImaginedWeIdCopys = lambda numWolves: [list(range(numSheep, numSheep + numWolves)) for _ in range(numWolves)]
        getImaginedWeIdsForInferenceSubject = lambda numWolves : [sortSelfIdFirst(weIdCopy, selfId) 
            for weIdCopy, selfId in zip(getImaginedWeIdCopys(numWolves), list(range(numWolves)))]
        perceptSelfAction = lambda singleAgentAction: singleAgentAction
        perceptOtherAction = lambda singleAgentAction: singleAgentAction
        composePerceptImaginedWeAction = lambda numWolves: [PerceptImaginedWeAction(imaginedWeIds, perceptSelfAction, perceptOtherAction) 
                for imaginedWeIds in getImaginedWeIdsForInferenceSubject(numWolves)]
        getPerceptActionForAll = lambda numWolves: [lambda action: action] * numSheep + composePerceptImaginedWeAction(numWolves)
         
        # Inference of Imagined We
        noInferIntention = lambda intentionPrior, action, perceivedAction: intentionPrior
        sheepUpdateIntentionMethod = noInferIntention
        
        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        getNumStateSpace = lambda numWolves: 2 * (numSheep + numWolves - 1)
        actionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)]
        predatorPowerRatio = 2
        wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        getWolfCentralControlActionSpace = lambda numWolves: list(it.product(wolfIndividualActionSpace, repeat = numWolves))
        getNumWolvesActionSpace = lambda numWolves: len(getWolfCentralControlActionSpace(numWolves))
        regularizationFactor = 1e-4
        getGenerateWolfCentralControlModel = lambda numWolves: GenerateModel(getNumStateSpace(numWolves), getNumWolvesActionSpace(numWolves), regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        wolfNNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        getInitWolfCentralControlModel = lambda numWolves: getGenerateWolfCentralControlModel(numWolves)(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths, 
                resBlockSize, initializationMethod, dropoutRate)
        getWolfModelPath = lambda numWolves: os.path.join('..', '..', 'data', 'preTrainModel', 
                'agentId='+str(5 * np.sum([10**_ for _ in range(numWolves)]))+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=200_trainSteps=50000')
        getWolfCentralControlNNModel = lambda numWolves: restoreVariables(getInitWolfCentralControlModel(numWolves), getWolfModelPath(numWolves))
        getWolfCentralControlPolicyGivenIntention = lambda numWolves: ApproximatePolicy(getWolfCentralControlNNModel(numWolves), getWolfCentralControlActionSpace(numWolves))

        softParameterInInference = 1
        softPolicyInInference = SoftPolicy(softParameterInInference)
        getSoftenWolfCentralControlPolicyGivenIntentionInInference = lambda numWolves: lambda state: softPolicyInInference(getWolfCentralControlPolicyGivenIntention(numWolves)(state))
        
        composeGetStateForPolicyGivenIntentionInInference = lambda numWolves: [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in
                getImaginedWeIdsForInferenceSubject(numWolves)]

        composeCalPoliciesLikelihood = lambda numWolves: [CalPolicyLikelihood(getStateForPolicyGivenIntentionInInference,
                getSoftenWolfCentralControlPolicyGivenIntentionInInference(numWolves)) for getStateForPolicyGivenIntentionInInference 
                in composeGetStateForPolicyGivenIntentionInInference(numWolves)]

        # ActionPerception Likelihood 
        calActionPerceptionLikelihood = lambda action, perceivedAction: int(np.allclose(np.array(action), np.array(perceivedAction)))

        # Joint Likelihood
        composeCalJointLikelihood = lambda calPolicyLikelihood, calActionPerceptionLikelihood: lambda intention, state, action, perceivedAction: \
            calPolicyLikelihood(intention, state, action) * calActionPerceptionLikelihood(action, perceivedAction)
        getCalJointLikelihood = lambda numWolves: [composeCalJointLikelihood(calPolicyLikelihood, calActionPerceptionLikelihood) 
            for calPolicyLikelihood in composeCalPoliciesLikelihood(numWolves)]

        # Joint Hypothesis Space
        priorDecayRate = 1
        intentionSpace = [(id,) for id in range(numSheep)]
        getActionSpaceInInference = lambda numWolves: getWolfCentralControlActionSpace(numWolves)
        getVariables = lambda numWolves: [intentionSpace, getActionSpaceInInference(numWolves)]
        getJointHypothesisSpace = lambda numWolves: pd.MultiIndex.from_product(getVariables(numWolves), names=['intention', 'action'])
        concernedHypothesisVariable = ['intention']
        composeInferImaginedWe = lambda numWolves: [InferOneStep(priorDecayRate, getJointHypothesisSpace(numWolves),
                concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in getCalJointLikelihood(numWolves)]
        composeUpdateIntention = lambda numWolves: [sheepUpdateIntentionMethod] * numSheep + composeInferImaginedWe(numWolves)
        chooseIntention = sampleFromDistribution

        # Get State of We and Intention
        getImaginedWeIdsForAllAgents = lambda numWolves: [[id] for id in range(numSheep)] + getImaginedWeIdsForInferenceSubject(numWolves)
        composeGetStateForPolicyGivenIntentions = lambda numWolves: [GetStateForPolicyGivenIntention(imaginedWeId) 
                for imaginedWeId in getImaginedWeIdsForAllAgents(numWolves)]

        #NN Policy Given Intention
        sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 2.5
        sheepIndividualActionSpace = list(map(tuple, np.array(sheepActionSpace) * preyPowerRatio))
        sheepCentralControlActionSpace = list(it.product(sheepIndividualActionSpace))
        numSheepActionSpace = len(sheepCentralControlActionSpace)
        regularizationFactor = 1e-4
        getGenerateSheepCentralControlModel = lambda numWolves: GenerateModel(getNumStateSpace(numWolves), numSheepActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        sheepNNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        getInitSheepCentralControlModel = lambda numWolves: getGenerateSheepCentralControlModel(numWolves)(sharedWidths * sheepNNDepth, actionLayerWidths, valueLayerWidths, 
                resBlockSize, initializationMethod, dropoutRate)
        getSheepModelPath = lambda numWolves: os.path.join('..', '..', 'data', 'preTrainModel',
                'agentId=0.'+str(numWolves)+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=100_trainSteps=50000')
        getSheepCentralControlNNModel = lambda numWolves: restoreVariables(getInitSheepCentralControlModel(numWolves), getSheepModelPath(numWolves))
        getSheepCentralControlPolicyGivenIntention = lambda numWolves: ApproximatePolicy(getSheepCentralControlNNModel(numWolves), sheepCentralControlActionSpace)

        softParameterInPlanning = 2.5
        softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
        getSoftenSheepCentralControlPolicyGivenIntentionInPlanning = lambda numWolves: lambda state: softPolicyInPlanning(getSheepCentralControlPolicyGivenIntention(numWolves)(state))
        getSoftenWolfCentralControlPolicyGivenIntentionInPlanning = lambda numWolves: lambda state: softPolicyInPlanning(getWolfCentralControlPolicyGivenIntention(numWolves)(state))
        getCentralControlPoliciesGivenIntentions = lambda numWolves: [getSoftenSheepCentralControlPolicyGivenIntentionInPlanning(numWolves)] * numSheep + [getSoftenWolfCentralControlPolicyGivenIntentionInPlanning(numWolves)] * numWolves
        composeIndividualPoliciesByEvaParameters = lambda numWolves: [PolicyOnChangableIntention(perceptAction, 
            imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
                for perceptAction, imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention 
                in zip(getPerceptActionForAll(numWolves), getImaginedWeIntentionPriors(numWolves), composeGetStateForPolicyGivenIntentions(numWolves), 
                    composeUpdateIntention(numWolves), getCentralControlPoliciesGivenIntentions(numWolves))]

        getIndividualIdsForAllAgents = lambda numWolves : list(range(numWolves + numSheep))
        actionChoiceMethods = {'sampleNNPolicy': sampleFromDistribution, 'maxNNPolicy': maxFromDistribution}
        sheepPolicyName = 'maxNNPolicy'
        wolfPolicyName = 'sampleNNPolicy'
        composeChooseCentrolAction = lambda numWolves: [actionChoiceMethods[sheepPolicyName]]* numSheep + [actionChoiceMethods[wolfPolicyName]]* numWolves
        composeAssignIndividualAction = lambda numWolves: [AssignCentralControlToIndividual(imaginedWeId, individualId) for imaginedWeId, individualId in
                zip(getImaginedWeIdsForAllAgents(numWolves), getIndividualIdsForAllAgents(numWolves))]
        composeGetIndividualActionMethods = lambda numWolves: [lambda centrolActionDist: assign(chooseAction(centrolActionDist)) for assign, chooseAction in
                zip(composeAssignIndividualAction(numWolves), composeChooseCentrolAction(numWolves))]

        policiesResetAttributes = ['lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
        getPoliciesResetAttributeValues = lambda numWolves: [dict(zip(policiesResetAttributes, [None, None, intentionPrior, [intentionPrior]])) for intentionPrior in
                getImaginedWeIntentionPriors(numWolves)]
        returnAttributes = ['formerIntentionPriors']
        composeResetPolicy = lambda numWolves, individualPolicies: ResetPolicy(getPoliciesResetAttributeValues(numWolves), individualPolicies, returnAttributes)
        attributesToRecord = ['lastAction']
        composeRecordActionForPolicy = lambda individualPolicies: RecordValuesForPolicyAttributes(attributesToRecord, individualPolicies) 
        
        # Sample and Save Trajectory
        maxRunningSteps = 101
        composeSampleTrajectory = lambda numWolves, individualPolicies: SampleTrajectory(maxRunningSteps, transit, getIsTerminal(numWolves),
                composeReset(numWolves), composeGetIndividualActionMethods(numWolves), composeResetPolicy(numWolves, individualPolicies),
                composeRecordActionForPolicy(individualPolicies))
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(numWolves)
        sampleTrajectory = self.composeSampleTrajectory(numWolves, individualPolicies)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]       
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [2]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithHierarchy',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName,
            'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'hierarchy': 1}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, parameters: saveToPickle(trajectories, getTrajectorySavePath(parameters))
   
    numTrajectories = 2
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, composeIndividualPoliciesByEvaParameters,
            composeSampleTrajectory, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    measureIntentionArcheivement = lambda df: lambda trajectory: int(len(trajectory) < maxRunningSteps) - 1 / maxRunningSteps * len(trajectory)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    fig = plt.figure()
    statisticsDf.index.name = 'Set Size of Intentions'
    __import__('ipdb').set_trace()
    ax = statisticsDf.plot(y = 'mean', yerr = 'se', ylim = (0, 0.5), label = '',  xlim = (21.95, 88.05), rot = 0)
    ax.set_ylabel('Accumulated Reward')
    #plt.suptitle('Wolves Accumulated Rewards')
    #plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
