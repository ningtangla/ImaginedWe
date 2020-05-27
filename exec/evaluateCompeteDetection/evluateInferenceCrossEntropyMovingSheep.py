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
import scipy.stats as stats
from scipy.interpolate import interp1d

from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, \
IsTerminal, InterpolateOneFrame, TransitWithTerminalCheckOfInterpolation
from src.MDPChasing.reward import RewardFunctionByTerminal
from src.MDPChasing.trajectory import ForwardOneStep, SampleTrajectory
from src.MDPChasing.policy import RandomPolicy
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.intention import CreateIntentionSpaceGivenSelfId, UpdateIntention
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsPolicyLikelihood, InferOneStep
from src.generateAction.imaginedWeSampleAction import WePolicyForCommittedAgents, SampleIndividualActionGivenIntention, \
        SampleActionOnChangableIntention, SampleActionOnFixedIntention, SampleActionMultiagent
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.sampleTrajectoryTools.evaluation import ComputeStatistics

class MeasureCrossEntropy:
    def __init__(self, imaginedWeIds, priorIndex):
        self.baseId, self.nonBaseIds = imaginedWeIds
        self.priorIndex = priorIndex

    def __call__(self, trajectory):
        priors = [timeStep[self.priorIndex] for timeStep in trajectory]
        baseDistributions = [list(prior[self.baseId].values())
                for prior in priors] 
        nonBaseDistributionsAllNonBaseAgents = [[list(prior[nonBaseId].values()) 
            for nonBaseId in self.nonBaseIds] for prior in priors]
        crossEntropies = [stats.entropy(baseDistribution) + np.mean([stats.entropy(baseDistribution, nonBaseDistribution) 
            for nonBaseDistribution in nonBaseDistributions]) 
            for baseDistribution, nonBaseDistributions in zip(baseDistributions, nonBaseDistributionsAllNonBaseAgents)]
        return crossEntropies

class MeasureConvergeRate:
    def __init__(self, imaginedWeIds, priorIndex, chooseIntention):
        self.imaginedWeIds = imaginedWeIds
        self.priorIndex = priorIndex
        self.chooseIntention = chooseIntention

    def __call__(self, trajectory):
        priors = [timeStep[self.priorIndex] for timeStep in trajectory]
        intentionSamplesTrajectory = [[[self.chooseIntention(prior[agentId])[0] for agentId in self.imaginedWeIds] 
            for _ in range(50)] 
            for prior in priors]
        convergeRates = [np.mean([np.all([intentionEachAgent == intentionSample[0] for intentionEachAgent in intentionSample])
            for intentionSample in intentionSamples]) 
            for intentionSamples in intentionSamplesTrajectory]
        return convergeRates

class MeasureConvergeOnGoal:
    def __init__(self, imaginedWeIds, priorIndex, chooseIntention):
        self.imaginedWeIds = imaginedWeIds
        self.priorIndex = priorIndex
        self.chooseIntention = chooseIntention

    def __call__(self, trajectory):
        priors = [timeStep[self.priorIndex] for timeStep in trajectory]
        goalSamplesTrajectory = [[[self.chooseIntention(prior[agentId])[0] for agentId in range(len(self.imaginedWeIds))] 
            for _ in range(50)] 
            for prior in priors]
        convergeRates = [np.mean([np.all([goalEachAgent == goalSample[0] for goalEachAgent in goalSample])
            for goalSample in goalSamples]) 
            for goalSamples in goalSamplesTrajectory]
        return convergeRates

class Interpolate1dData:
    def __init__(self, numToInterpolate):
        self.numToInterpolate = numToInterpolate
    def __call__(self, data):
        #__import__('ipdb').set_trace()
        x = np.divide(np.arange(len(data)), len(data) - 1)
        y = np.array(data)
        f = interp1d(x, y, kind='linear')
        xnew = np.linspace(0., 1., self.numToInterpolate)
        interpolatedData = f(xnew)
        return interpolatedData

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numSheep'] = [2, 4, 8]
    manipulatedVariables['hierarchy'] = [0, 1, 2]
    manipulatedVariables['valuePriorEndTime'] = [-100, 0, 100]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateHierarchyPlanning',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
   
    NNNumSimulations = 200
    maxRunningSteps = 50
    wolfPolicySoft = 2.5
    sheepPolicySoft = 2.5
    numWolves = 3
    trajectoryFixedParameters = {'numWolves': numWolves, 'wolfPolicySoft': wolfPolicySoft, 'sheepPolicySoft': sheepPolicySoft, 
            'maxRunningSteps': maxRunningSteps, 'NNNumSimulations': NNNumSimulations}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
     
    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    #loadTrajectoriesFromDf = lambda df: [trajectory for trajectory in loadTrajectories(readParametersFromDf(df)) if len(trajectory) < readParametersFromDf(df)['maxRunningSteps']]
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df)) 

    priorIndexinTimestep = 4
    composeInterpolateFunction = lambda df: Interpolate1dData(maxRunningSteps)
    #getImaginedWeIdsForCrossEntropy = lambda df: [readParametersFromDf(df)['numSheep'], list(range(readParametersFromDf(df)['numSheep'] + 1, readParametersFromDf(df)['numSheep'] +
    #    readParametersFromDf(df)['numWolves']))] 
    #composeMeasureCrossEntropy = lambda df: MeasureCrossEntropy(getImaginedWeIdsForCrossEntropy(df), priorIndexinTimestep)
    #measureFunction = lambda df: lambda trajectory: composeInterpolateFunction(df)(composeMeasureCrossEntropy(df)(trajectory))
    getImaginedWeIdsForConvergeRate = lambda df: list(range(readParametersFromDf(df)['numSheep'], readParametersFromDf(df)['numSheep'] + numWolves )) 
    measureConvergeRate = lambda df: MeasureConvergeOnGoal(getImaginedWeIdsForConvergeRate(df), priorIndexinTimestep, sampleFromDistribution)
    measureFunction = lambda df: lambda trajectory: composeInterpolateFunction(df)(measureConvergeRate(df)(trajectory))
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    
    fig = plt.figure()
    numColumns = len(manipulatedVariables['hierarchy'])
    numRows = len(manipulatedVariables['valuePriorEndTime'])
    plotCounter = 1
    
    hierarchyLabels = ['noHierarchy9', 'noHierarchy5', 'Hierarchy9']

    for key, group in statisticsDf.groupby(['valuePriorEndTime', 'hierarchy']):
        group.index = group.index.droplevel(['valuePriorEndTime', 'hierarchy'])
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        #if plotCounter % numColumns == 1:
        #axForDraw.set_ylabel('Cross Entropy')
        axForDraw.set_ylabel('Converge Rate')
        if plotCounter <= numColumns:
            axForDraw.set_title(hierarchyLabels[key[1]])
        for numSheep, grp in group.groupby('numSheep'):
            df = pd.DataFrame(grp.values[0].tolist(), columns = list(range(maxRunningSteps)), index = ['mean','se']).T
            #df.plot.line(ax = axForDraw, label = 'Set Size of Intentions = {}'.format(numSheep), y = 'mean', yerr = 'se', ylim = (0, 3), rot = 0)
            df.plot.line(ax = axForDraw, label = 'numSheep = {}'.format(numSheep), y = 'mean', yerr = 'se', ylim = (0, 1), rot = 0)

        plotCounter = plotCounter + 1

    plt.suptitle(str(numWolves) + 'Wolves')
    #plt.legend(loc='best')
    fig.text(x = 0.5, y = 0.04, s = 'hierarchy', ha = 'center', va = 'center')
    fig.text(x = 0.05, y = 0.5, s = 'valuePriorEndTime', ha = 'center', va = 'center', rotation=90)
    plt.show()

if __name__ == '__main__':
    main()
