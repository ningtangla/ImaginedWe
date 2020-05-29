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
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [2]
    manipulatedVariables['numSheep'] = [2, 4, 8]#, 4, 8]
    manipulatedVariables['hierarchy'] = [0]
    manipulatedVariables['valuePriorEndTime'] = [-100]#[-100, 0, 100]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateObstacle',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
   
    NNNumSimulations = 250
    maxRunningSteps = 51
    wolfPolicySoft = 2.5
    sheepPolicySoft = 2.5
    trajectoryFixedParameters = {'wolfPolicySoft': wolfPolicySoft, 'sheepPolicySoft': sheepPolicySoft, 
            'maxRunningSteps': maxRunningSteps, 'NNNumSimulations': NNNumSimulations}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    
    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    maxSteps = maxRunningSteps-26
    measureIntentionArcheivement = lambda df: lambda trajectory: int(len(trajectory) < maxSteps) - 1 / maxSteps * min(len(trajectory), maxSteps)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf) 
    
    fig = plt.figure()
    numColumns = len(manipulatedVariables['hierarchy'])
    numRows = len(manipulatedVariables['valuePriorEndTime'])
    plotCounter = 1

    for key, group in statisticsDf.groupby(['valuePriorEndTime', 'hierarchy']):
        group.index = group.index.droplevel(['valuePriorEndTime', 'hierarchy'])
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_ylabel('Accumulated Reward')
        #axForDraw.set_ylabel(str(numWolves))
        
        #if plotCounter <= numColumns:
        #    axForDraw.set_title(str(key[1]) + 'hierarchy')
        numWolvesLabels = ['2', '3']
        for numWolves, grp in group.groupby('numWolves'):
            grp.index = grp.index.droplevel('numWolves')
            grp.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', label = str(numWolves)+' Wolves', ylim = (0.2, 0.8), marker = 'o', rot = 0 )
        axForDraw.xaxis.set_label_text('Number of Sheep') 
       
        plotCounter = plotCounter + 1

    #plt.suptitle('3 Wolves')
    #fig.text(x = 0.5, y = 0.04, s = 'numWolves', ha = 'center', va = 'center')
    #fig.text(x = 0.05, y = 0.5, s = 'valuePriorEndTime', ha = 'center', va = 'center', rotation=90)
    plt.show()

if __name__ == '__main__':
    main()
