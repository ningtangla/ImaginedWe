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

class ActionPerceptionLikelihood:
    def __init__(self, noise):
        self.noise = noise
    def __call__(self, action, perceivedAction):
        likelihood = np.prod([scipy.stats.multivariate_normal.pdf(
            perceivedAction[index], action[index], np.diag([self.noise**2] * len(action[index]))) for index in range(len(action))])
        return likelihood

class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, composeIndividualPoliciesByEvaParameters, composeSampleTrajectory, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.composeIndividualPoliciesByEvaParameters = composeIndividualPoliciesByEvaParameters
        self.composeSampleTrajectory = composeSampleTrajectory
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)
        perceptNoise = parameters['perceptNoiseForAll']
        maxRunningSteps = parameters['maxRunningSteps']
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(perceptNoise)
        sampleTrajectory = self.composeSampleTrajectory(maxRunningSteps, individualPolicies)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['perceptNoiseForAll'] = [1e-1, 4e1, 8e1, 1e3]
    manipulatedVariables['maxRunningSteps'] = [100]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    numOfAgent = 4
    reset = Reset(xBoundary, yBoundary, numOfAgent)
    
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

    possiblePreyIds = [0, 1]
    possiblePredatorIds = [2, 3]
    posIndexInState = [0, 1]
    getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
    getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
    killzoneRadius = 30
    isTerminal = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)

    # MDP Policy
    sheepImagindWeIntentionPrior = {(2, 3): 1}
    wolfImaginedWeIntentionPrior = {(0, ):0.5, (1,): 0.5}
    imaginedWeIntentionPriors = [sheepImagindWeIntentionPrior, sheepImagindWeIntentionPrior, wolfImaginedWeIntentionPrior, wolfImaginedWeIntentionPrior]
    
    imaginedWeIdsForInferenceSubjects = [[2, 3], [3, 2]]
    composePerceptSelfAction = lambda noise: SampleNoisyAction(noise)
    composePerceptOtherAction = lambda noise: SampleNoisyAction(noise)
    composePerceptImaginedWeAction = lambda noise: [PerceptImaginedWeAction(imaginedWeIds, composePerceptSelfAction(noise), composePerceptOtherAction(noise))
        for imaginedWeIds in imaginedWeIdsForInferenceSubjects]
    getPerceptActionForAll = lambda noise: [lambda action: action] * 2 + composePerceptImaginedWeAction(noise)

    # Inference of Imagined We
    noInferIntention = lambda intentionPrior, lastState, perceivedAction: intentionPrior
    sheepUpdateIntentionMethod = noInferIntention

    # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
    numStateSpace = 6
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    predatorPowerRatio = 2
    wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolfCentralControlActionSpace = list(it.product(wolfIndividualActionSpace, wolfIndividualActionSpace))
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
    wolfModelPath = os.path.join('..', '..', 'data', 'preTrainModel', 
            'agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    wolfCentralControlNNModel = restoreVariables(initWolfCentralControlModel, wolfModelPath)
    wolfCentralControlPolicyGivenIntention = ApproximatePolicy(wolfCentralControlNNModel, wolfCentralControlActionSpace)

    softParameterInInference = 1
    softPolicyInInference = SoftPolicy(softParameterInInference)
    softenWolfCentralControlPolicyGivenIntentionInInference = lambda state: softPolicyInInference(wolfCentralControlPolicyGivenIntention(state))
    
    getStateForPolicyGivenIntentionInInferences = [GetStateForPolicyGivenIntention(imaginedWeIds) 
            for imaginedWeIds in imaginedWeIdsForInferenceSubjects] 

    calPoliciesLikelihood = [CalPolicyLikelihood(getState, softenWolfCentralControlPolicyGivenIntentionInInference) 
            for getState in getStateForPolicyGivenIntentionInInferences]

    # ActionPerception Likelihood 
    composeCalActionPerceptionLikelihood = lambda perceptNoise : lambda action, perceivedAction: np.prod([scipy.stats.multivariate_normal.pdf(
        perceivedAction[index], action[index], np.diag([perceptNoise**2] * len(action[index]))) for index in range(len(action))])
    #composeCalActionPerceptionLikelihood = lambda perceptNoise : ActionPerceptionLikelihood(perceptNoise) 

    # Joint Likelihood
    composeCalJointLikelihood = lambda calPolicyLikelihood, calActionPerceptionLikelihood: lambda intention, state, action, perceivedAction: \
        calPolicyLikelihood(intention, state, action) * calActionPerceptionLikelihood(action, perceivedAction)
    getCalJointLikelihood = lambda perceptNoise: [composeCalJointLikelihood(calPolicyLikelihood, composeCalActionPerceptionLikelihood(perceptNoise)) 
        for calPolicyLikelihood in calPoliciesLikelihood]

    # Joint Hypothesis Space
    priorDecayRate = 1
    intentionSpace = [(0,), (1,)]
    actionSpaceInInference = wolfCentralControlActionSpace
    variables = [intentionSpace, actionSpaceInInference]
    jointHypothesisSpace = pd.MultiIndex.from_product(variables, names=['intention', 'action'])
    concernedHypothesisVariable = ['intention']
    composeInferImaginedWe = lambda perceptNoise: [InferOneStep(priorDecayRate, jointHypothesisSpace,
            concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in getCalJointLikelihood(perceptNoise)]
    getUpdateIntention = lambda perceptNoise: [sheepUpdateIntentionMethod, sheepUpdateIntentionMethod] + composeInferImaginedWe(perceptNoise)
    chooseIntention = sampleFromDistribution
    
    # Get State of We and Intention
    imaginedWeIdsForAllAgents = [[0], [1], [2, 3], [3, 2]]
    getStateForPolicyGivenIntentions = [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in imaginedWeIdsForAllAgents]

    #NN Policy Given Intention
    numStateSpace = 6
    preyPowerRatio = 2.5
    sheepIndividualActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    sheepCentralControlActionSpace = list(it.product(sheepIndividualActionSpace))
    numSheepActionSpace = len(sheepCentralControlActionSpace)
    regularizationFactor = 1e-4
    generateSheepCentralControlModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    sheepNNDepth = 5
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    initSheepCentralControlModel = generateSheepCentralControlModel(sharedWidths * sheepNNDepth, actionLayerWidths, valueLayerWidths, 
            resBlockSize, initializationMethod, dropoutRate)
    sheepModelPath = os.path.join('..', '..', 'data', 'preTrainModel',
            'agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    sheepCentralControlNNModel = restoreVariables(initSheepCentralControlModel, sheepModelPath)
    sheepCentralControlPolicyGivenIntention = ApproximatePolicy(sheepCentralControlNNModel, sheepCentralControlActionSpace)

    softParameterInPlanning = 2.5
    softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
    softensheepCentralControlPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(sheepCentralControlPolicyGivenIntention(state))
    softenWolfCentralControlPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(wolfCentralControlPolicyGivenIntention(state))
    centralControlPoliciesGivenIntentions = [softensheepCentralControlPolicyGivenIntentionInPlanning, softensheepCentralControlPolicyGivenIntentionInPlanning,
            softenWolfCentralControlPolicyGivenIntentionInPlanning, softenWolfCentralControlPolicyGivenIntentionInPlanning]
    composeIndividualPoliciesByEvaParameters = lambda perceptNoise: [PolicyOnChangableIntention(perceptImaginedWeAction, 
        imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
            for perceptImaginedWeAction, imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention 
            in zip(getPerceptActionForAll(perceptNoise), imaginedWeIntentionPriors, getStateForPolicyGivenIntentions, getUpdateIntention(perceptNoise), centralControlPoliciesGivenIntentions)]

    individualIdsForAllAgents = [0, 1, 2, 3]
    actionChoiceMethods = {'sampleNNPolicy': sampleFromDistribution, 'maxNNPolicy': maxFromDistribution}
    sheepPolicyName = 'maxNNPolicy'
    wolfPolicyName = 'sampleNNPolicy'
    chooseCentrolAction = [actionChoiceMethods[sheepPolicyName]]* 2 + [actionChoiceMethods[wolfPolicyName]]* 2
    assignIndividualAction = [AssignCentralControlToIndividual(imaginedWeId, individualId) 
            for imaginedWeId, individualId in zip(imaginedWeIdsForAllAgents, individualIdsForAllAgents)]
    getIndividualActionMethods = [lambda centrolActionDist: assign(chooseAction(centrolActionDist)) for assign, chooseAction in zip(assignIndividualAction, chooseCentrolAction)]
    
    policiesResetAttributes = ['timeStep', 'lastAction', 'lastState', 'intentionPrior', 'formerIntentionPriors']
    policiesResetAttributeValues = [dict(zip(policiesResetAttributes, [0, None, None, intentionPrior, [intentionPrior]])) for intentionPrior in imaginedWeIntentionPriors]
    returnAttributes = ['formerIntentionPriors']
    composeResetPolicy = lambda individualPolicies: ResetPolicy(policiesResetAttributeValues, individualPolicies, returnAttributes)
    attributesToRecord = ['lastAction']
    composeRecordActionForPolicy = lambda individualPolicies: RecordValuesForPolicyAttributes(attributesToRecord, individualPolicies) 
    
    # Sample and Save Trajectory
    composeSampleTrajectory = lambda maxRunningSteps, individualPolicies: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset,
            getIndividualActionMethods, composeResetPolicy(individualPolicies), composeRecordActionForPolicy(individualPolicies))

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithNoisePerception',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName,
        'policySoftParameter': softParameterInPlanning}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, parameters: saveToPickle(trajectories, getTrajectorySavePath(parameters))
   
    numTrajectories = 200
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, composeIndividualPoliciesByEvaParameters,
            composeSampleTrajectory, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    possibleIntentionIds = [[0],[1]]
    wolfImaginedWeId = [2, 3]
    stateIndexInTimestep = 0
    judgeSuccessCatchOrEscape = lambda booleanSign: int(booleanSign)
    #measureIntentionArcheivement = lambda df: MeasureIntentionArcheivement(possibleIntentionIds, wolfImaginedWeId, stateIndexInTimestep, posIndexInState, killzoneRadius, judgeSuccessCatchOrEscape)
    measureIntentionArcheivement = lambda df: lambda trajectory: int(len(trajectory) < readParametersFromDf(df)['maxRunningSteps']) - 1 / readParametersFromDf(df)['maxRunningSteps'] * len(trajectory) 
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    fig = plt.figure()
    #numColumns = len(manipulatedVariables['perceptNoise'])
    numColumns = 1
    numRows = len(manipulatedVariables['maxRunningSteps'])
    plotCounter = 1
    
    for maxRunningSteps, group in statisticsDf.groupby('maxRunningSteps'):
        group.index = group.index.droplevel('maxRunningSteps')
        
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_ylabel('Accumulated Reward')
        group.index.name = 'Action Perception Noise'
        group.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', label = '', xlim = (-5, 1005), ylim = (-1, 0.5), marker = 'o', rot = 0 )
        plotCounter = plotCounter + 1

    #plt.suptitle('Wolves Accumulated Reward')
    plt.show()
if __name__ == '__main__':
    main()
