import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp
import pygame as pg
from pygame.color import THECOLORS

from src.visualization.drawDemo import DrawBackground, DrawCircleOutside, DrawState, ChaseTrialWithTraj, InterpolateState
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, \
IsTerminal, InterpolateOneFrame, TransitWithTerminalCheckOfInterpolation

def updateColorSpace(colorSpace, posterior, goalSpace, imaginedWeIds):
    goalProbabilities = np.mean([[max(0, 1 * (posterior[individualIndex][(goalId, tuple(imaginedWeIds))] - 0.5/len(goalSpace))) 
        for goalId in goalSpace] for individualIndex in range(len(imaginedWeIds))], axis = 0) #goalId, weIds = intention(in posterior)
    colorRepresentProbability = np.array([np.array([0, 170, 0]) * probability for probability in goalProbabilities]) + np.array(
            [colorSpace[goalId] for goalId in goalSpace])
    updatedColorSpace = np.array(colorSpace).copy()
    updatedColorSpace[[goalId for goalId in goalSpace]] = colorRepresentProbability
    return updatedColorSpace

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateHierarchyPlanning',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    NNNumSimulations = 250
    maxRunningSteps = 52
    softParameterInPlanningForSheep = 2.5
    softParameterInPlanning = 2.5
    hierarchy = 0
    trajectoryFixedParameters = {'sheepPolicySoft': softParameterInPlanningForSheep, 'wolfPolicySoft': softParameterInPlanning,
            'maxRunningSteps': maxRunningSteps, 'hierarchy': hierarchy, 'NNNumSimulations':NNNumSimulations}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    numWolves = 2
    numSheep = 2
    valuePriorEndTime = -100
    valuePriorSoftMaxBeta = 0.0
    trajectoryParameters = {'numWolves': numWolves, 'numSheep': numSheep, 'valuePriorEndTime': valuePriorEndTime, 
            'valuePriorSoftMaxBeta': valuePriorSoftMaxBeta}
    wolfType = 'sharedReward'
    #trajectoryParameters.update({'wolfType': wolfType})
    
    trajectories = loadTrajectories(trajectoryParameters) 
    # generate demo image
    screenWidth = 600
    screenHeight = 600
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    lineColor = THECOLORS['white']
    lineWidth = 4
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    
    FPS = 32
    circleColorSpace = [[100, 100, 100]]*numSheep + [[255, 255, 255]] * numWolves
    circleSize = 10
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numSheep + numWolves))
    saveImage = True
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str('forDemo')
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    goalSpace = list(range(numSheep))
    imaginedWeIdsForInferenceSubject = list(range(numSheep, numWolves + numSheep))
    softParameter = 1
    softFunction = SoftDistribution(softParameter)
    updateColorSpaceByPosterior = lambda colorSpace, posterior : updateColorSpace(
            colorSpace, [softFunction(individualPosterior) for individualPosterior in posterior], goalSpace, imaginedWeIdsForInferenceSubject)
    
    #updateColorSpaceByPosterior = lambda originalColorSpace, posterior : originalColorSpace
    outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
    outsideCircleColor = np.array([[255, 0, 0]] * numWolves)
    outsideCircleSize = 15 
    drawCircleOutside = DrawCircleOutside(screen, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize)
    drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
    
   # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = InterpolateOneFrame(stayInBoundaryByReflectVelocity)
    numFramesToInterpolate = 7
    interpolateState = InterpolateState(numFramesToInterpolate, transit)
    
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = 4
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep, posteriorIndexInTimeStep)
    
    print(len(trajectories))
    lens = [len(trajectory) for trajectory in trajectories]
    index = np.argsort(-np.array(lens))
    print(index)
    print(trajectories[0][1])
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[0:10]]]
    print([len(trajectory) for trajectory in np.array(trajectories)[index[:]]])
    #[chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[13:14]]]

if __name__ == '__main__':
    main()
