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


def main():
    # manipulated variables
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSheep'] = [1, 2, 4]
    manipulatedVariables['wolfType'] = ['individualReward', 'sharedReward', 'sharedAgencyByIndividualRewardWolf']#['no', 'reward', 'agencyBySharedWolf', 'agencyByIndividualWolf']
    manipulatedVariables['sheepConcern'] = ['selfSheep']#['sheepWithSharedRewardWolf', 'copySheepWithSharedRewardWolf', 'sheepWithIndividualRewardWolf', 'copySheepWithIndividualRewardWolf']
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
   
    maxRunningSteps = 101
    numWolves = 3
    trajectoryFixedParameters = {'numWolves': numWolves, 'maxRunningSteps': maxRunningSteps}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
     
    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    #loadTrajectoriesFromDf = lambda df: [trajectory for trajectory in loadTrajectories(readParametersFromDf(df)) if len(trajectory) < readParametersFromDf(df)['maxRunningSteps']]
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df)) 
    
    measureIntentionArcheivement = lambda df: lambda trajectory: np.sum(np.array(trajectory)[:, 3])
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf) 
    #__import__('ipdb').set_trace() 
    fig = plt.figure()
    fig.set_dpi(120)
    numColumns = len(manipulatedVariables['sheepConcern'])
    numRows = 1#len(manipulatedVariables['sheepType'])
    plotCounter = 1
    
    sheepTypeTable = {'sheepWithSharedRewardWolf': 'No Copy Shared',
            'copySheepWithSharedRewardWolf': 'Copy Shared', 
            'sheepWithIndividualRewardWolf': 'No Copy Individual', 
            'copySheepWithIndividualRewardWolf': 'Copy Individual'}
    sheepConcernTable = {
            'selfSheep': 'sheep concern self only', 
            'allSheep': 'sheep concern all sheep'}
    wolfTypeTable = {'individualReward': 'Individual Reward', 'sharedReward': 'Shared Reward', 
            'sharedAgencyBySharedRewardWolf': 'Shared Agency By Shared Reward Wolf',
            'sharedAgencyByIndividualRewardWolf': 'Shared Agency'}# By Individual Reward Wolf'}

    for key, group in statisticsDf.groupby(['sheepConcern']):
        group.index = group.index.droplevel(['sheepConcern'])
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        if (plotCounter) % max(numColumns, 2) == 1:
            axForDraw.set_ylabel('Number of Biting Sheep')
        
        #if plotCounter <= numColumns:
        #    axForDraw.set_title(str(sheepConcernTable[key]))
        
        numWolvesLabels = ['2', '3']
        for wolfType, grp in group.groupby('wolfType'):
            grp.index = grp.index.droplevel('wolfType')
            grp.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', label = wolfTypeTable[wolfType], ylim = (0, 18), marker = 'o', rot = 0 )
            if int((plotCounter - 1) / numColumns) == numRows - 1:
                axForDraw.xaxis.set_label_text('Number of Sheep') 
            else:
                xAxis = axForDraw.get_xaxis()
                xLabel = xAxis.get_label()
                xLabel.set_visible(False)

       
        plotCounter = plotCounter + 1

    #plt.suptitle('Sheep Type')
    #fig.text(x = 0.5, y = 0.92, s = 'Wolf Type', ha = 'center', va = 'center')
    #fig.text(x = 0.05, y = 0.5, s = 'Sheep Type', ha = 'center', va = 'center', rotation=90)
    plt.show()

if __name__ == '__main__':
    main()
