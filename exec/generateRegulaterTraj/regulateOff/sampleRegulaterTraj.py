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

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, HeatSeekingDiscreteStochasticPolicy, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal, InterpolateState
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory, SampleTrajectoryWithRender, Render
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables, ApproximateValue
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.inference import CalPolicyLikelihood, InferOneStep, InferOnTrajectory
from src.evaluation import ComputeStatistics
from src.valueFromNode import EstimateValueFromNode
from src.commitment import RegulateCommitmentBroken, BreakCommitmentBasedOnTime, ActiveReCommit
from src.inference.regulatorInference import GetActionLikelihoodOnIndividualPlanning, GetPartnerActionLikelihoodOnJointPlanning, InferPartnerCommitment

def sortSelfIdFirst(weId, selfId):
    weId.insert(0, weId.pop(selfId))
    return weId

class PolicyCommunicatable:
    def __init__(self, individualPolicies, senderId, receiverId, senderAttribute, receiverAttribute):
        self.individualPolicies = individualPolicies
        self.senderId = senderId
        self.receiverId = receiverId
        self.senderAttribute = senderAttribute
        self.receiverAttribute = receiverAttribute
    
    def __call__(self, state):
        attributeValue = getattr(self.individualPolicies[self.senderId], self.senderAttribute)
        setattr(self.individualPolicies[self.receiverId], self.receiverAttribute, attributeValue)
        actionDist = [individualPolicy(state) for individualPolicy in self.individualPolicies]
        #print(attributeValue, self.individualPolicies[self.receiverId].warned, self.individualPolicies[self.receiverId].committed)
        #print(actionDist[self.receiverId])
        return actionDist

def main():
    DEBUG = 1
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 1
        endSampleIndex = 200
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
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'regulateOff', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    reCommitProbability = 0
    precision = 1.83
    regulateOn = 0
    numOneWolfActionSpace = 8
    NNNumSimulations = 300 #300 with distance Herustic; 200 without distanceHerustic
    numWolves = 2
    maxRunningSteps = 101
    softParameterInPlanning = 2.5
    sheepPolicyName = 'sampleNNPolicy'
    wolfPolicyName = 'sampleNNPolicy'
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName, 'NNNumSimulations': NNNumSimulations,
            'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'numOneWolfActionSpace': numOneWolfActionSpace, 'numWolves': numWolves,
            'reCommitedProbability': reCommitProbability, 'regulateOn': regulateOn, 'precision': precision}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, trajectoryFixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)


    if not os.path.isfile(trajectorySavePath):

        # MDP Env
        xBoundary = [0,600]
        yBoundary = [0,600]
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
        #wolfImaginedWeIntentionPrior = {(sheepId, ): 1/numSheep for sheepId in range(numSheep)}
        committedFirstSheepId = 0
        wolfImaginedWeIntentionPrior = {(committedFirstSheepId, ): 0.999, (1, ): 0.001}
        imaginedWeIntentionPriors = [sheepImagindWeIntentionPrior] * numSheep + [wolfImaginedWeIntentionPrior] * numWolves

        # Percept Action
        imaginedWeIdCopys = [list(range(numSheep, numSheep + numWolves)) for _ in range(numWolves)]
        imaginedWeIdsForInferenceSubject = [sortSelfIdFirst(weIdCopy, selfId)
            for weIdCopy, selfId in zip(imaginedWeIdCopys, list(range(numWolves)))]
        perceptSelfAction = lambda singleAgentAction: singleAgentAction
        perceptOtherAction = lambda singleAgentAction: singleAgentAction
        perceptImaginedWeAction = [PerceptImaginedWeAction(imaginedWeIds, perceptSelfAction, perceptOtherAction)
                for imaginedWeIds in imaginedWeIdsForInferenceSubject]
        perceptActionForAll = [lambda action: action] * numSheep + perceptImaginedWeAction

        # Inference of Imagined We
        noInferIntention = lambda intentionPrior, action, perceivedAction: intentionPrior
        sheepUpdateIntentionMethod = noInferIntention

        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        numStateSpace = 2 * (numSheep + numWolves - 1)
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]
        predatorPowerRatio = 8
        wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        wolfCentralControlActionSpace = list(it.product(wolfIndividualActionSpace, repeat = numWolves))
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
                'agentId='+str(numOneWolfActionSpace * np.sum([10**_ for _ in
                range(numWolves)]))+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations='+str(NNNumSimulations)+'_trainSteps=50000')
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
        priorDecayRate = 0.7
        intentionSpace = [(id,) for id in range(numSheep)]
        actionSpaceInInference = wolfCentralControlActionSpace
        variables = [intentionSpace, actionSpaceInInference]
        jointHypothesisSpace = pd.MultiIndex.from_product(variables, names=['intention', 'action'])
        concernedHypothesisVariable = ['intention']
        inferImaginedWe = [InferOneStep(priorDecayRate, jointHypothesisSpace,
                concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in calJointLikelihoods]
        updateIntention = [sheepUpdateIntentionMethod] * (numSheep + 0) + inferImaginedWe
        chooseIntention = sampleFromDistribution

        # Get State of We and Intention
        imaginedWeIdsForAllAgents = [[id] for id in range(numSheep)] + imaginedWeIdsForInferenceSubject
        getStateForPolicyGivenIntentions = [GetStateForPolicyGivenIntention(imaginedWeId)
                for imaginedWeId in imaginedWeIdsForAllAgents]

        #NN Policy Given Intention
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
        initSheepCentralControlModel = generateSheepCentralControlModel(sharedWidths * sheepNNDepth, actionLayerWidths, valueLayerWidths,
                resBlockSize, initializationMethod, dropoutRate)
        sheepModelPath = os.path.join('..', '..', '..', 'data', 'preTrainModel',
                'agentId=0.'+str(numWolves)+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=100_trainSteps=50000')
        sheepCentralControlNNModel = restoreVariables(initSheepCentralControlModel, sheepModelPath)
        sheepCentralControlPolicyGivenIntention = ApproximatePolicy(sheepCentralControlNNModel, sheepCentralControlActionSpace)

	#break Commitment Policy
        preyIdInBreakPolicy = 1
        predatorIdInBreakPolicy = 3
        posIndexInState = [0, 1]
        getPreyPosInBreak = GetAgentPosFromState(preyIdInBreakPolicy, posIndexInState)
        getPredatorPosInBreak = GetAgentPosFromState(predatorIdInBreakPolicy, posIndexInState)
        breakCommitmentPolicy = HeatSeekingDiscreteStochasticPolicy(precision, wolfIndividualActionSpace, getPredatorPosInBreak, getPreyPosInBreak)
        
        softBreakPolicy = SoftPolicy(0.5)
        softenBreakCommitmentPolicy = lambda state: softBreakPolicy(breakCommitmentPolicy(state)) 
        breakCommitmentPolicys = [None, None, None, softenBreakCommitmentPolicy]

        #active break 
        breakTime = 3
        activeBreak = BreakCommitmentBasedOnTime(breakTime)
        activeBreaks = [None, None, None, activeBreak]

        #regulate detect partner commitment
        partnerIndexInPerceivedAction = 1
        getActionLikelihoodOnIndividualPlanning= GetActionLikelihoodOnIndividualPlanning(partnerIndexInPerceivedAction)
        weIndexInPercivedAction = [0, 1]
        regulaterIndexInPerceivedAction = 0
        getPartnerActionLikelihoodOnJointPlanning = GetPartnerActionLikelihoodOnJointPlanning(weIndexInPercivedAction, regulaterIndexInPerceivedAction)
        commitLikelihoodFunction = lambda state, action: getPartnerActionLikelihoodOnJointPlanning(commitPolicy(state), action) 
        regulaterId = 2
        partnerId = 3
        commitPolicy = lambda state: wolfCentralControlPolicyGivenIntention(state[[committedFirstSheepId, regulaterId, partnerId]])
        noCommitLikelihoodFunction = lambda state, action: getActionLikelihoodOnIndividualPlanning(breakCommitmentPolicy(state), action) 
        commitmentLikelihoodFunctions = {0: noCommitLikelihoodFunction, 1: commitLikelihoodFunction}
        calJointLikelihoodInCommitInfer = lambda commitment, state, action: commitmentLikelihoodFunctions[commitment](state, action)

        priorDecayRateInCommitInfer = 0.9
        commitmentSpace = [0, 1]
        variablesInCommitInfer = [commitmentSpace]
        jointHypothesisSpaceInCommitInfer = pd.MultiIndex.from_product(variablesInCommitInfer, names=['commitment'])
        concernedHypothesisVariableInCommitInfer = ['commitment']
        updatePartnerCommitDistribution = InferPartnerCommitment(priorDecayRateInCommitInfer, jointHypothesisSpaceInCommitInfer, 
                concernedHypothesisVariableInCommitInfer, calJointLikelihoodInCommitInfer)
        
        partnerCommitPrior = {0: 0.5, 1: 0.5}
        chooseCommitmentWarning = sampleFromDistribution
        regulateCommitmentBroken = RegulateCommitmentBroken(partnerCommitPrior, updatePartnerCommitDistribution, chooseCommitmentWarning)
        regulateCommitmentBrokens = [None, None, None, None]
        
        #activeRecommit
        activateReCommit = ActiveReCommit(reCommitProbability)
        activateReCommits = [None, None, None, activateReCommit]
	
        #final individual polices
        softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
        softSheepParameterInPlanning = softParameterInPlanning
        softSheepPolicyInPlanning = SoftPolicy(softSheepParameterInPlanning)
        softenSheepCentralControlPolicyGivenIntentionInPlanning = lambda state: softSheepPolicyInPlanning(sheepCentralControlPolicyGivenIntention(state))
        softenWolfCentralControlPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(wolfCentralControlPolicyGivenIntention(state))
        centralControlPoliciesGivenIntentions = [softenSheepCentralControlPolicyGivenIntentionInPlanning] * numSheep + [softenWolfCentralControlPolicyGivenIntentionInPlanning] * numWolves
        planningIntervals = [1] * numSheep +  [1] * numWolves
        intentionInferInterval = 1
        individualPolicies = [PolicyOnChangableIntention(perceptAction,
            imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, 
            getStateForPolicyGivenIntention, policyGivenIntention, planningInterval, intentionInferInterval,
            regulateCommitment, breakCommit, breakPolicy, reCommit)
                for perceptAction, imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution,
                policyGivenIntention, planningInterval, regulateCommitment, breakCommit, breakPolicy, reCommit
                in zip(perceptActionForAll, imaginedWeIntentionPriors, getStateForPolicyGivenIntentions,
                    updateIntention, centralControlPoliciesGivenIntentions, planningIntervals, regulateCommitmentBrokens,
                    activeBreaks, breakCommitmentPolicys, activateReCommits)]

        individualIdsForAllAgents = list(range(numWolves + numSheep))
        actionChoiceMethods = {'sampleNNPolicy': sampleFromDistribution, 'maxNNPolicy': maxFromDistribution}
        chooseCentrolAction = [actionChoiceMethods[sheepPolicyName]]* numSheep + [actionChoiceMethods[wolfPolicyName]]* numWolves
        assignIndividualAction = [AssignCentralControlToIndividual(imaginedWeId, individualId) for imaginedWeId, individualId in
                zip(imaginedWeIdsForAllAgents, individualIdsForAllAgents)]
        individualActionMethods = [lambda centrolActionDist: assign(chooseAction(centrolActionDist)) for assign, chooseAction in
                zip(assignIndividualAction, chooseCentrolAction)]

        policiesResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 
                'formerIntentionPriors', 'commitmentWarn', 'formerCommitmentWarn', 'committed', 'formerCommitted']
        policiesResetAttributeValues = [dict(zip(policiesResetAttributes, [0, None, None, intentionPrior, [intentionPrior], 0, [0], 1, [1]])) for intentionPrior in
                imaginedWeIntentionPriors]
        returnAttributes = ['formerIntentionPriors', 'formerCommitmentWarn', 'formerCommitted']
        resetPolicy = ResetPolicy(policiesResetAttributeValues, individualPolicies, returnAttributes)
        attributesToRecord = ['lastAction']
        recordActionForPolicy = RecordValuesForPolicyAttributes(attributesToRecord, individualPolicies)
       
        senderAttribute = 'commitmentWarn'
        receiverAttribute = 'warned'
        policyCommunicatable = PolicyCommunicatable(individualPolicies, regulaterId, partnerId, senderAttribute, receiverAttribute)
        policy = lambda state: policyCommunicatable(state)
        
        # Sample and Save Trajectory
        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['blue'], THECOLORS['yellow'], THECOLORS['red']]
            circleSize = 10

            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, posIndexInState, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        interpolateStateInPlay = InterpolateState(4, transit, isTerminalInPlay)
        transitInPlay = lambda state, action : interpolateStateInPlay(state, action)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitInPlay, isTerminalInPlay,
                reset, individualActionMethods, resetPolicy,
                recordActionForPolicy, render, renderOn)
        

        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        print([len(traj) for traj in trajectories])


if __name__ == '__main__':
    main()
