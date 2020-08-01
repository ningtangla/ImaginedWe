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
        wolfType = parameters['wolfType']
        sheepConcern = parameters['sheepConcern']
        
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

        # ------------ model ------------------------
        wolvesIDForWolfObserve = list(range(numWolves))
        sheepsIDForWolfObserve = list(range(numWolves, numSheep + numWolves))
        blocksIDForWolfObserve = list(range(numSheep + numWolves, numSheep + numWolves + numBlocks))
        observeOneAgentForWolf = lambda agentID: Observe(agentID, wolvesIDForWolfObserve, sheepsIDForWolfObserve, 
                blocksIDForWolfObserve, getPosFromAgentState, getVelFromAgentState)
        observeWolf = lambda state: [observeOneAgentForWolf(agentID)(state) for agentID in range(numWolves + numSheep)]

        obsIDsForWolf = wolvesIDForWolfObserve + sheepsIDForWolfObserve + blocksIDForWolfObserve
        initObsForWolfParams = observeWolf(reset()[obsIDsForWolf])
        obsShapeWolf = [initObsForWolfParams[obsID].shape[0] for obsID in range(len(initObsForWolfParams))]
        buildWolfModels = BuildMADDPGModels(actionDim, numWolves + numSheep, obsShapeWolf)
        layerWidthForWolf = [64 * (numWolves - 1), 64 * (numWolves - 1)]
        wolfModelsList = [buildWolfModels(layerWidthForWolf, agentID) for agentID in range(numWolves)]

        if wolfType == 'sharedReward':
            prefix = 'maddpg'
        if wolfType == 'individualReward':
            prefix = 'maddpgIndividWolf'
        wolfFileName = "{}wolves{}sheep{}blocks{}eps_agent".format(numWolves, numSheep, numBlocks, maxEpisode)
        wolfModelPaths = [os.path.join(dirName, '..', '..', 'data', 'preTrainModel', prefix + wolfFileName + str(i) + '60000eps') for i in range(numWolves)]
        print(numWolves, obsShapeWolf, wolfModelPaths) 

        [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]

        actionDimReshaped = 2
        cov = [0.03 ** 2 for _ in range(actionDimReshaped)]
        buildGaussian = BuildGaussianFixCov(cov)
        actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoNoisy)
        composeWolfPolicy = lambda wolfModel: lambda state: sampleFromContinuousSpace(buildGaussian(
            tuple(reshapeAction(actOneStepOneModelWolf(wolfModel, observeWolf(state))))))
        
        #actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoisy)
        #composeWolfPolicy = lambda wolfModel: lambda state: tuple(reshapeAction(actOneStepOneModelSheep(wolfModel, observeWolf(state))))
        wolvesSampleActions = [composeWolfPolicy(wolfModel) for wolfModel in wolfModelsList]
       
        trajectories = []
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
            allIndividualSampleActions = wolvesSampleActions + sheepSampleActions
            sampleAction = lambda state: [sampleIndividualAction(state) for sampleIndividualAction in allIndividualSampleActions]
            trajectory = sampleTrajectory(sampleAction)
            trajectories.append(trajectory) 
        trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps}
        self.saveTrajectoryByParameters(trajectories, trajectoryFixedParameters, parameters)
        print(np.mean([len(tra) for tra in trajectories]))
    
        # visualize
        if visualizeTraj:
            wolfColor = np.array([0.85, 0.35, 0.35])
            sheepColor = np.array([0.35, 0.85, 0.35])
            blockColor = np.array([0.25, 0.25, 0.25])
            entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheep + [blockColor] * numBlocks
            render = Render(entitiesSizeList, entitiesColorList, numAgents, getPosFromAgentState)
            trajToRender = np.concatenate(trajectories)
            render(trajToRender)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [4]
    manipulatedVariables['wolfType'] = ['individualReward', 'sharedReward']
    manipulatedVariables['sheepConcern'] = ['selfSheep']
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
