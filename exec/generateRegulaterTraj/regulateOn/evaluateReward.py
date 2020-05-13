import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))

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


def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [2]
    manipulatedVariables['reCommitedProbability'] = [0, 1, 2]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', '..', 'data', 'regulateOn',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
  
    regulateOn = 1
    numOneWolfActionSpace = 8
    precision = 1.83
    NNNumSimulations = 300
    maxRunningSteps = 101
    softParameterInPlanning = 2.5
    sheepPolicyName = 'sampleNNPolicy'
    wolfPolicyName = 'sampleNNPolicy'
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName,
            'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'NNNumSimulations': NNNumSimulations, 
            'numOneWolfActionSpace': numOneWolfActionSpace, 'regulateOn': regulateOn, 'precision': precision}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    
    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    maxSteps = 30
    measureIntentionArcheivement = lambda df: lambda trajectory: int(len(trajectory) < maxSteps) - 1 / maxSteps * min(len(trajectory), maxSteps)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf) 
    
    fig = plt.figure()
    numColumns = 1#len(manipulatedVariables['numActionSpaceForOthers'])
    numRows = len(manipulatedVariables['numWolves'])
    plotCounter = 1

    for numWolves, group in statisticsDf.groupby('numWolves'):
        
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_ylabel('Accumulated Reward')
        #axForDraw.set_ylabel(str(numWolves))
        
        group.index = group.index.droplevel('numWolves')
        group.index.name = 'Regulater and Partner type'
        group.index = ['regulate, uncooperative partner', 'regulate, cooperative partner', 'no regulate, uncooperative partner']
        group.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', label = '', xlim = (-0.1, 2.1), ylim = (-.2, 0.4), marker = 'o', rot = 0 )
        plotCounter = plotCounter + 1

    #plt.suptitle('Wolves Accumulated Reward')
    plt.show()

if __name__ == '__main__':
    main()
