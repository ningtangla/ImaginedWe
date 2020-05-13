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

from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.inference.inference import CalPolicyLikelihood, InferOneStep, InferOnTrajectory
from src.evaluation import ComputeStatistics
from scipy.interpolate import interp1d

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

class Interpolate1dData:
    def __init__(self, numToInterpolate):
        self.numToInterpolate = numToInterpolate
    def __call__(self, data):
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
    manipulatedVariables['numWolves'] = [3]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    #trajectory dir
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithNumIntentions',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    maxRunningSteps = 50
    softParameterInPlanning = 2.5
    NNNumSimulations = 200
    sheepPolicyName = 'sampleNNPolicy'
    wolfPolicyName = 'sampleNNPolicy'
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName, 'NNNumSimulations': NNNumSimulations,
            'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'hierarchy': 2}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    
    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    #loadTrajectoriesFromDf = lambda df: [trajectory for trajectory in loadTrajectories(readParametersFromDf(df)) if len(trajectory) < readParametersFromDf(df)['maxRunningSteps']]
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df)) 

    priorIndexinTimestep = 3
    getImaginedWeIdsForCrossEntropy = lambda df: [readParametersFromDf(df)['numSheep'], list(range(readParametersFromDf(df)['numSheep'] + 1, readParametersFromDf(df)['numSheep'] +
        readParametersFromDf(df)['numWolves']))] 
    composeMeasureCrossEntropy = lambda df: MeasureCrossEntropy(getImaginedWeIdsForCrossEntropy(df), priorIndexinTimestep)
    composeInterpolateFunction = lambda df: Interpolate1dData(maxRunningSteps)
    measureFunction = lambda df: lambda trajectory: composeInterpolateFunction(df)(composeMeasureCrossEntropy(df)(trajectory))
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    
    fig = plt.figure()
    numColumns = len(manipulatedVariables['numWolves'])
    #numColumns = 1
    numRows = 1
    plotCounter = 1
    for numWolves, group in statisticsDf.groupby('numWolves'):
        group.index = group.index.droplevel('numWolves')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        #if plotCounter % numColumns == 1:
        axForDraw.set_ylabel('Cross Entropy')
        for numSheep, grp in group.groupby('numSheep'):
            df = pd.DataFrame(grp.values[0].tolist(), columns = list(range(maxRunningSteps)), index = ['mean','se']).T
            df.plot.line(ax = axForDraw, label = 'Set Size of Intentions = {}'.format(numSheep), y = 'mean', yerr = 'se', ylim = (0, 3), rot = 0)

        plotCounter = plotCounter + 1

    #plt.suptitle('Wolves Cross Entropy')
    #plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
