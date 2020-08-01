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
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.MDPChasing.envMADDPG import *
from src.algorithms.MADDPG.myMADDPG import *
from src.visualization.visualizeEnvMADDPG import *
from src.MDPChasing.trajectory import ForwardOneStep, SampleTrajectory
from src.MDPChasing.policy import RandomPolicy
from src.MDPChasing.state import getStateOrActionFirstPersonPerspective, getStateOrActionThirdPersonPerspective
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution, BuildGaussianFixCov, sampleFromContinuousSpace
from src.mathTools.soft import SoftMax
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, ApproximateValue, restoreVariables
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.intention import CreateIntentionSpaceGivenSelfId, CalIntentionValueGivenState, AdjustIntentionPriorGivenValueOfState, UpdateIntention
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsContinuousPolicyLikelihood, InferOneStep
from src.generateAction.imaginedWeSampleAction import PolicyForUncommittedAgent, PolicyForCommittedAgent, GetActionFromJointActionDistribution, \
        HierarchyPolicyForCommittedAgent, SampleIndividualActionGivenIntention, \
        SampleActionOnChangableIntention, SampleActionOnFixedIntention, SampleActionMultiagent
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.sampleTrajectoryTools.evaluation import ComputeStatistics

class ComposeCentralControlPolicyByGaussianOnDeterministicAction:
    def __init__(self, reshapeAction, observe, actOneStepOneModel, buildGaussian):
        self.reshapeAction = reshapeAction
        self.observe = observe
        self.actOneStepOneModel = actOneStepOneModel
        self.buildGaussian = buildGaussian

    def __call__(self, individualModels, numAgentsInWe):
        centralControlPolicy = lambda state: [self.buildGaussian(tuple(self.reshapeAction(
            self.actOneStepOneModel(individualModels[agentId], self.observe(state))))) for agentId in range(numAgentsInWe)] 
        return centralControlPolicy

class ComposeCentralControlPolicyDeterministicAction:
    def __init__(self, reshapeAction, observe, actOneStepOneModel):
        self.reshapeAction = reshapeAction
        self.observe = observe
        self.actOneStepOneModel = actOneStepOneModel

    def __call__(self, individualModels, numAgentsInWe):
        centralControlPolicy = lambda state: [self.buildGaussian(tuple(self.reshapeAction(
            self.actOneStepOneModel(individualModels[agentId], self.observe(state))))) for agentId in range(numAgentsInWe)] 
        return centralControlPolicy

class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)
        visualizeTraj = False

        numWolves = parameters['numWolves']
        numSheep = parameters['numSheep']
        softParamterForValue = parameters['valuePriorSoftMaxBeta']
        valuePriorEndTime = parameters['valuePriorEndTime']
        deviationFor2DAction = parameters['deviationFor2DAction']
        rationalityBetaInInference = parameters['rationalityBetaInInference']
        wolfType = parameters['wolfType']
        sheepConcern = parameters['sheepConcern']
        print(rationalityBetaInInference)
        
        ## MDP Env  
	# state is all multi agent state # action is all multi agent action
        wolvesID = list(range(numWolves))
        sheepsID = list(range(numWolves, numWolves + numSheep))
        possibleWolvesIds = wolvesID
        possibleSheepIds = sheepsID

        numAgents = numWolves + numSheep
        numBlocks = 5 - numWolves
        blocksID = list(range(numAgents, numAgents + numBlocks))
        numEntities = numAgents + numBlocks
        
        sheepSize = 0.05
        wolfSize = 0.075
        blockSize = 0.2
        
        sheepMaxSpeed = 1.3 * 1
        wolfMaxSpeed = 1.0 * 1
        blockMaxSpeed = None

        entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheep + [blockSize] * numBlocks
        entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheep + [blockMaxSpeed] * numBlocks
        entitiesMovableList = [True]* numAgents + [False] * numBlocks
        massList = [1.0] * numEntities
        
        reshapeActionInTransit = lambda action: action
        getCollisionForce = GetCollisionForce()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                              getCollisionForce, getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                        entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
        transit = TransitMultiAgentChasing(numEntities, reshapeActionInTransit, applyActionForce, applyEnvironForce, integrateState)
        
        isCollision = IsCollision(getPosFromAgentState)
        collisonRewardWolf = 1
        punishForOutOfBoundForWolf = lambda stata: 0
        rewardWolf = RewardCentralControlPunishBond(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBoundForWolf, collisonRewardWolf)
        collisonRewardSheep = -1
        punishForOutOfBoundForSheep = PunishForOutOfBound()
        rewardSheep = RewardCentralControlPunishBond(sheepsID, wolvesID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBoundForSheep, collisonRewardSheep)

        forwardOneStep = ForwardOneStep(transit, rewardWolf)
        
        reset = ResetMultiAgentChasing(numAgents, numBlocks)
        isTerminal = lambda state: False
        maxRunningSteps = 101
        sampleTrajectory = SampleTrajectory(maxRunningSteps, isTerminal, reset, forwardOneStep)
        
        ## MDP Policy
        worldDim = 2
        actionDim = worldDim * 2 + 1

        layerWidth = [64 * (numWolves - 1), 64 * (numWolves - 1)]

	# Sheep Part
        # ------------ model ------------------------
        if sheepConcern == 'selfSheep':
            sheepConcernSelfOnly = 1
        if sheepConcern == 'allSheep':
            sheepConcernSelfOnly = 0
        numSheepToObserveWhenSheepSameOrDiff = [numSheep, 1]
        numSheepToObserve = numSheepToObserveWhenSheepSameOrDiff[sheepConcernSelfOnly]

        print(numSheepToObserve)
        sheepModelListOfDiffWolfReward = []
        sheepType = 'mixed'
        if sheepType == 'mixed':
            sheepPrefixList = ['maddpgIndividWolf', 'maddpg']
        else:
            sheepPrefixList = [sheepType]
        for sheepPrefix in sheepPrefixList:
            wolvesIDForSheepObserve = list(range(numWolves))
            sheepsIDForSheepObserve = list(range(numWolves, numSheepToObserve + numWolves))
            blocksIDForSheepObserve = list(range(numSheepToObserve + numWolves, numSheepToObserve + numWolves + numBlocks))
            observeOneAgentForSheep = lambda agentID: Observe(agentID, wolvesIDForSheepObserve, sheepsIDForSheepObserve, 
                    blocksIDForSheepObserve, getPosFromAgentState, getVelFromAgentState)
            observeSheep = lambda state: [observeOneAgentForSheep(agentID)(state) for agentID in range(numWolves + numSheepToObserve)]
           
            obsIDsForSheep = wolvesIDForSheepObserve + sheepsIDForSheepObserve + blocksIDForSheepObserve
            initObsForSheepParams = observeSheep(reset()[obsIDsForSheep])
            obsShapeSheep = [initObsForSheepParams[obsID].shape[0] for obsID in range(len(initObsForSheepParams))]
            
            buildSheepModels = BuildMADDPGModels(actionDim, numWolves + numSheepToObserve, obsShapeSheep)
            sheepModelsList = [buildSheepModels(layerWidth, agentID) for agentID in range(numWolves, numWolves + numSheepToObserve)]

            dirName = os.path.dirname(__file__)
            maxEpisode = 60000
            print(sheepPrefix)
            sheepFileName = "{}wolves{}sheep{}blocks{}eps_agent".format(numWolves, numSheepToObserve, numBlocks, maxEpisode)
            sheepModelPaths = [os.path.join(dirName, '..', '..', 'data', 'preTrainModel', sheepPrefix + sheepFileName + str(i) + '60000eps') 
                    for i in range(numWolves, numWolves + numSheepToObserve)]

            [restoreVariables(model, path) for model, path in zip(sheepModelsList, sheepModelPaths)]
            sheepModelListOfDiffWolfReward = sheepModelListOfDiffWolfReward + sheepModelsList 
        
        # Sheep Policy Function
        reshapeAction = ReshapeAction()
        actOneStepOneModelSheep = ActOneStep(actByPolicyTrainNoisy)
        
        # Sheep Generate Action
        numAllSheepModels = len(sheepModelListOfDiffWolfReward)

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
        #perceptAction = lambda action: action
        perceptSelfAction = SampleNoisyAction(deviationFor2DAction)
        perceptOtherAction = SampleNoisyAction(deviationFor2DAction)
        perceptAction = PerceptImaginedWeAction(possibleWolvesIds, perceptSelfAction, perceptOtherAction)
        #perceptAction = lambda action: action
        

        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        # ------------ model ------------------------
        weModelsListBaseOnNumInWe = []
        observeListBaseOnNumInWe = []
        for numAgentInWe in range(2, numWolves + 1):
            numBlocksForWe = 5 - numAgentInWe
            wolvesIDForWolfObserve = list(range(numAgentInWe))
            sheepsIDForWolfObserve = list(range(numAgentInWe, 1 + numAgentInWe))
            blocksIDForWolfObserve = list(range(1 + numAgentInWe, 1 + numAgentInWe + numBlocksForWe))
            observeOneAgentForWolf = lambda agentID: Observe(agentID, wolvesIDForWolfObserve, sheepsIDForWolfObserve, 
                    blocksIDForWolfObserve, getPosFromAgentState, getVelFromAgentState)
            observeWolf = lambda state: [observeOneAgentForWolf(agentID)(state) for agentID in range(numAgentInWe + 1)]
            observeListBaseOnNumInWe.append(observeWolf)

            obsIDsForWolf = wolvesIDForWolfObserve + sheepsIDForWolfObserve + blocksIDForWolfObserve
            initObsForWolfParams = observeWolf(reset()[obsIDsForWolf])
            obsShapeWolf = [initObsForWolfParams[obsID].shape[0] for obsID in range(len(initObsForWolfParams))]
            buildWolfModels = BuildMADDPGModels(actionDim, numAgentInWe + 1, obsShapeWolf)
            layerWidthForWolf = [64 * (numAgentInWe - 1), 64 * (numAgentInWe - 1)]
            wolfModelsList = [buildWolfModels(layerWidthForWolf, agentID) for agentID in range(numAgentInWe)]
            
            if wolfType == 'sharedAgencyByIndividualRewardWolf':
                wolfPrefix = 'maddpgIndividWolf'
            if wolfType == 'sharedAgencyBySharedRewardWolf':
                wolfPrefix = 'maddpg'
            wolfFileName = "{}wolves{}sheep{}blocks{}eps_agent".format(numAgentInWe, 1, numBlocksForWe, maxEpisode)
            wolfModelPaths = [os.path.join(dirName, '..', '..', 'data', 'preTrainModel', wolfPrefix + wolfFileName + str(i) + '60000eps') for i in range(numAgentInWe)]
            print(numAgentInWe, obsShapeWolf, wolfModelPaths) 

            [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]
            weModelsListBaseOnNumInWe.append(wolfModelsList)

        actionDimReshaped = 2
        cov = [deviationFor2DAction ** 2 for _ in range(actionDimReshaped)]
        buildGaussian = BuildGaussianFixCov(cov)
        actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoNoisy)
        #actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoisy)
        composeCentralControlPolicy = lambda observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction(reshapeAction, 
                observe, actOneStepOneModelWolf, buildGaussian) 
        wolvesCentralControlPolicies = [composeCentralControlPolicy(observeListBaseOnNumInWe[numAgentsInWe - 2])(weModelsListBaseOnNumInWe[numAgentsInWe - 2], numAgentsInWe) 
                for numAgentsInWe in range(2, numWolves + 1)]

        centralControlPolicyListBasedOnNumAgentsInWe = wolvesCentralControlPolicies # 0 for two agents in We, 1 for three agents...
        softPolicyInInference = lambda distribution : distribution
        getStateThirdPersonPerspective = lambda state, goalId, weIds: getStateOrActionThirdPersonPerspective(state, goalId, weIds, blocksID)
        policyForCommittedAgentsInInference = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWe, softPolicyInInference,
                getStateThirdPersonPerspective)
        concernedAgentsIds = possibleWolvesIds
        calCommittedAgentsPolicyLikelihood = CalCommittedAgentsContinuousPolicyLikelihood(concernedAgentsIds, 
                policyForCommittedAgentsInInference, rationalityBetaInInference)
        
        randomActionSpace = [(5, 0), (3.5, 3.5), (0, 5), (-3.5, 3.5), (-5, 0), (-3.5, -3.5), (0, -5), (3.5, -3.5), (0, 0)]
        randomPolicy = RandomPolicy(randomActionSpace)
        getStateFirstPersonPerspective = lambda state, goalId, weIds, selfId: getStateOrActionFirstPersonPerspective(state, goalId, weIds, selfId, blocksID)
        policyForUncommittedAgentsInInference = PolicyForUncommittedAgent(possibleWolvesIds, randomPolicy, 
                softPolicyInInference, getStateFirstPersonPerspective)
        calUncommittedAgentsPolicyLikelihood = CalUncommittedAgentsPolicyLikelihood(possibleWolvesIds, 
                concernedAgentsIds, policyForUncommittedAgentsInInference)

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

        if numSheep == 1:
            inferIntentionOneStepList = [lambda prior, state, action: prior] * 3

        adjustIntentionPriorGivenValueOfState = lambda state: 1
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
        covForPlanning = [0.03 ** 2 for _ in range(actionDimReshaped)]
        buildGaussianForPlanning = BuildGaussianFixCov(covForPlanning)
        composeCentralControlPolicyForPlanning = lambda observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction(reshapeAction, 
                observe, actOneStepOneModelWolf, buildGaussianForPlanning) 
        wolvesCentralControlPoliciesForPlanning = [composeCentralControlPolicyForPlanning(
            observeListBaseOnNumInWe[numAgentsInWe - 2])(weModelsListBaseOnNumInWe[numAgentsInWe - 2], numAgentsInWe) 
                for numAgentsInWe in range(2, numWolves + 1)]

        centralControlPolicyListBasedOnNumAgentsInWeForPlanning = wolvesCentralControlPoliciesForPlanning # 0 for two agents in We, 1 for three agents...
        softPolicyInPlanning = lambda distribution: distribution
        policyForCommittedAgentInPlanning = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWeForPlanning, softPolicyInPlanning,
                getStateThirdPersonPerspective)
        
        policyForUncommittedAgentInPlanning = PolicyForUncommittedAgent(possibleWolvesIds, randomPolicy, softPolicyInPlanning,
                getStateFirstPersonPerspective)
        
        def wolfChooseActionMethod(individualContinuousDistributions):
            centralControlAction = tuple([tuple(sampleFromContinuousSpace(distribution)) 
                for distribution in individualContinuousDistributions])
            return centralControlAction
        
        getSelfActionThirdPersonPerspective = lambda weIds, selfId : list(weIds).index(selfId)
        chooseCommittedAction = GetActionFromJointActionDistribution(wolfChooseActionMethod, getSelfActionThirdPersonPerspective)
        chooseUncommittedAction = sampleFromDistribution
        wolvesSampleIndividualActionGivenIntentionList = [SampleIndividualActionGivenIntention(selfId, policyForCommittedAgentInPlanning, 
            policyForUncommittedAgentInPlanning, chooseCommittedAction, chooseUncommittedAction) 
            for selfId in possibleWolvesIds]

        # Sample and Save Trajectory
        trajectoriesWithIntentionDists = []
        for trajectoryId in range(self.numTrajectories):
            sheepModelsForPolicy = [sheepModelListOfDiffWolfReward[np.random.choice(numAllSheepModels)] for sheepId in possibleSheepIds]
            if sheepConcernSelfOnly:
                composeSheepPolicy = lambda sheepModel : lambda state: {tuple(reshapeAction(actOneStepOneModelSheep(sheepModel, observeSheep(state)))): 1}
                sheepChooseActionMethod = sampleFromDistribution
                sheepSampleActions = [SampleActionOnFixedIntention(selfId, possibleWolvesIds, composeSheepPolicy(sheepModel), sheepChooseActionMethod, blocksID)
                        for selfId, sheepModel in zip(possibleSheepIds, sheepModelsForPolicy)]
            else:
                composeSheepPolicy = lambda sheepModel: lambda state: tuple(reshapeAction(actOneStepOneModelSheep(sheepModel, observeSheep(state))))
                sheepSampleActions = [composeSheepPolicy(sheepModel) for sheepModel in sheepModelsForPolicy]
            
            wolvesSampleActions = [SampleActionOnChangableIntention(updateIntention, wolvesSampleIndividualActionGivenIntention) 
                    for updateIntention, wolvesSampleIndividualActionGivenIntention in zip(updateIntentions, wolvesSampleIndividualActionGivenIntentionList)]
            allIndividualSampleActions = wolvesSampleActions + sheepSampleActions
            sampleActionMultiAgent = SampleActionMultiagent(allIndividualSampleActions, recordActionForUpdateIntention)
            trajectory = sampleTrajectory(sampleActionMultiAgent)
            intentionDistributions = getIntentionDistributions()
            trajectoryWithIntentionDists = [tuple(list(SASRPair) + list(intentionDist)) 
                    for SASRPair, intentionDist in zip(trajectory, intentionDistributions)]
            trajectoriesWithIntentionDists.append(tuple(trajectoryWithIntentionDists)) 
            resetIntentions()
            #print(intentionDistributions)
        trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps}
        self.saveTrajectoryByParameters(trajectoriesWithIntentionDists, trajectoryFixedParameters, parameters)
        print(np.mean([len(tra) for tra in trajectoriesWithIntentionDists]))
    
        # visualize
        if visualizeTraj:
            wolfColor = np.array([0.85, 0.35, 0.35])
            sheepColor = np.array([0.35, 0.85, 0.35])
            blockColor = np.array([0.25, 0.25, 0.25])
            entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheep + [blockColor] * numBlocks
            render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
            trajToRender = np.concatenate(trajectoriesWithIntentionDists)
            render(trajToRender)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [1, 2, 4]
    manipulatedVariables['valuePriorSoftMaxBeta'] = [0.0]
    manipulatedVariables['valuePriorEndTime'] = [-100]
    manipulatedVariables['deviationFor2DAction'] = [1.0]#, 3.0, 9.0]
    manipulatedVariables['rationalityBetaInInference'] = [1.0]#[0.0, 0.1, 0.2, 0.5, 1.0]
    manipulatedVariables['sheepConcern'] = ['selfSheep']
    manipulatedVariables['wolfType'] = ['sharedAgencyByIndividualRewardWolf']
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateHierarchyPlanningEnvMADDPG',
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
