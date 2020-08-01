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
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [1, 2, 4]
    manipulatedVariables['deviationFor2DAction'] = [1.0, 3.0, 9.0]
    #manipulatedVariables['rationalityBetaInInference'] = [0.0, 0.125, 0.25, 0.5, 1.0]
    manipulatedVariables['rationalityBetaInInference'] = [0.0, 0.1, 0.2, 1.0]
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
   
    valuePriorSoftMaxBeta = 0.0
    maxRunningSteps = 102
    valuePriorEndTime = -100
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps,'valuePriorSoftMaxBeta': valuePriorSoftMaxBeta,
            'valuePriorEndTime': valuePriorEndTime}
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
    __import__('ipdb').set_trace() 
    fig = plt.figure()
    fig.set_dpi(120)
    numColumns = len(manipulatedVariables['rationalityBetaInInference'])
    numRows = len(manipulatedVariables['deviationFor2DAction'])
    plotCounter = 1
    
    for key, group in statisticsDf.groupby(['deviationFor2DAction', 'rationalityBetaInInference']):
        group.index = group.index.droplevel(['deviationFor2DAction', 'rationalityBetaInInference'])
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        if (plotCounter) % max(numColumns, 2) == 1:
            axForDraw.set_ylabel('Observe Noise = '+str(key[0]))
        
        if plotCounter <= numColumns:
            axForDraw.set_title('Rationality Beta = '+str(key[1]))
        
        numWolvesLabels = ['2', '3']
        for numWolves, grp in group.groupby('numWolves'):
            grp.index = grp.index.droplevel('numWolves')
            grp.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', label = str(numWolves)+' Wolves', ylim = (0, 16), marker = 'o', rot = 0 )
            if int((plotCounter - 1) / numColumns) == numRows - 1:
                axForDraw.xaxis.set_label_text('Number of Sheep') 
            else:
                xAxis = axForDraw.get_xaxis()
                xLabel = xAxis.get_label()
                xLabel.set_visible(False)

       
        plotCounter = plotCounter + 1

    plt.suptitle('Accumulated Reward')
    #fig.text(x = 0.5, y = 0.04, s = 'numWolves', ha = 'center', va = 'center')
    #fig.text(x = 0.05, y = 0.5, s = 'valuePriorEndTime', ha = 'center', va = 'center', rotation=90)
    plt.show()

if __name__ == '__main__':
    main()
