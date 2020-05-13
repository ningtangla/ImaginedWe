import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt

from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal, InterpolateState
from src.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.MDPChasing.reward import RewardFunctionCompete

if __name__ == '__main__':
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'generateGuidedMCTSWeWithRollout', 'OneLeveLPolicy', 'trajectories')
    #trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'generateGuidedMCTSWeWithRollout', 'HierarchyPolicy', 'trajectories')
    trajectorySaveExtension = '.pickle'
    saveTrajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'evaluateIntentionInPlanningWithHierarchyGuidedMCTS', 'trajectories')

    trajLenList = []
    accumulateRewardList = []

    NNNumSimulations = 200  # 300 with distance Herustic; 200 without distanceHerustic
    diff = []
    for maxStep in range(40, 51):
        result = []
        for numOneWolfActionSpace in [5]:
            numWolves = 2
            maxRunningSteps = 101
            softParameterInPlanning = 2.5
            sheepPolicyName = 'sampleNNPolicy'
            wolfPolicyName = 'sampleNNPolicy'
            trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName, 'NNNumSimulations': NNNumSimulations,
                                         'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'numOneWolfActionSpace': numOneWolfActionSpace, 'numWolves': numWolves}

            generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, trajectoryFixedParameters)

            fuzzySearchParameterNames = ['sampleIndex']
            loadTrajectories = LoadTrajectories(generateTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
            loadedTrajectories = loadTrajectories({'agentId': 1})
             
            hierarchy = 1
            trajectorySaveFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName, 'NNNumSimulations': NNNumSimulations,
                                         'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'hierarchy': hierarchy, 'numWolves': numWolves}
            generateTrajectorySavePathForResave = GetSavePath(saveTrajectoriesSaveDirectory, trajectorySaveExtension, trajectorySaveFixedParameters)
            trajectoriesResavePath = generateTrajectorySavePathForResave({})
            saveToPickle(loadedTrajectories, trajectoriesResavePath)
            #print(len(loadedTrajectories))
    
    # traj length
            trajLen = np.mean([len(traj) for traj in loadedTrajectories])
            #print([len(tra) for tra in loadedTrajectories], trajLen)
            filtedTrajLen = [len(traj) for traj in loadedTrajectories if len(traj)]
            trajLenList.append(trajLen)
            reward = np.mean([(int(lenTra<maxStep) - min(lenTra, maxStep)/maxStep) for lenTra in filtedTrajLen])
            result.append(reward)
            print(reward)
            #print(len(filtedTrajLen), filtedTrajLen)

    # accumulate reward
            xBoundary = [0, 600]
            yBoundary = [0, 600]
            numSheep = 2
            numWolves = 2
            numOfAgent = numWolves + numSheep
            reset = Reset(xBoundary, yBoundary, numOfAgent)

            possiblePreyIds = list(range(numSheep))
            possiblePredatorIds = list(range(numSheep, numSheep + numWolves))
            posIndexInState = [0, 1]
            getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
            getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
            killzoneRadius = 50
            isTerminalInPlay = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)

            playAliveBonus = -1 / maxStep
            playDeathPenalty = 1
            playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, isTerminalInPlay)

            decay = 1
            accumulateRewards = AccumulateRewards(decay, playReward)
            addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

            def filterState(timeStep): return (timeStep[0][0:numOfAgent], timeStep[1], timeStep[2])
            trajectories = [[filterState(trajectory[timeStepIndex]) for timeStepIndex in range(len(trajectory)) if timeStepIndex < maxStep] 
                for trajectory in loadedTrajectories if len(trajectory) <= maxStep]
            valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]
            dataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
            #result.append(dataMeanAccumulatedReward)
            #print(dataMeanAccumulatedReward)
            accumulateRewardList.append(dataMeanAccumulatedReward)
        #diff.append(result[1] - result[0])
        #print(diff)
    #print(diff.index(max(diff)) + 20)
# plot
#    fig = plt.figure()
#    axForDraw = fig.add_subplot(1, 1, 1)
#    axForDraw.set_ylim(0, 1)
#
#    xlabel = ['5*5Wolves', '9*9Wolves', ]
#
#    x = np.arange(2)
#    a = accumulateRewardList
#
#    totalWidth, n = 0.6, 2
#    width = totalWidth / n
#
#    x = x - (totalWidth - width) / 2
#    plt.bar(x, a, width=width)
#
#    plt.xticks(x, xlabel)
#
#    xlocs, xlabs = plt.xticks()
#    for i, v in enumerate(a):
#        plt.text(xlocs[i] - 0.05, v + 0.1, str(v))
#
#    plt.legend()
#    plt.title('mean trajectory length with simulation={} totalSteps=100'.format(NNNumSimulations))
#    plt.savefig('compareWolfWithDiffActionSpace.png')
#    plt.show()
