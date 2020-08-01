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

def updateColorSpace(colorSpace, posterior, selfPosteriorIndex, concernedAgentId, competeIntention, cooperateIntention):
    competeProbability = posterior[selfPosteriorIndex][competeIntention]
    cooperateProbability = posterior[selfPosteriorIndex][cooperateIntention]
    colorRepresentProbability = np.array([0, 0, 255]) * competeProbability + np.array([255, 0, 0]) * cooperateProbability
    updatedColorSpace = np.array(colorSpace).copy()
    updatedColorSpace[concernedAgentId] = colorRepresentProbability
    return updatedColorSpace

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateCompeteDetection',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    NNNumSimulations = 250
    maxRunningSteps = 61
    softParameterInPlanningForSheep = 2.0
    softParameterInPlanning = 2.0
    trajectoryFixedParameters = {'sheepPolicySoft': softParameterInPlanningForSheep, 'wolfPolicySoft': softParameterInPlanning,
            'maxRunningSteps': maxRunningSteps, 'NNNumSimulations':NNNumSimulations}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    numWolves = 2
    numSheep = 1
    competePolicy = 'heatseeking'
    heatseekingPrecesion = 1.83
    otherCompeteRate = 1.0
    competeDetectionRate = 0.5
    inferenceSoft = 0.05
    trajectoryParameters = {'heatseekingPrecesion': heatseekingPrecesion, 'inferenceSoft': inferenceSoft, 'numWolves': numWolves, 'numSheep': numSheep,
            'competePolicy': competePolicy, 'otherCompeteRate': otherCompeteRate, 'competeDetectionRate': competeDetectionRate}
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
    
    FPS = 24
    circleColorSpace = [[0, 255, 0]]*numSheep + [[255, 0, 0]] * numWolves
    circleSize = 10
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numSheep + numWolves))
    #saveImage = False
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
    softParameter = 1.1
    softFunction = SoftDistribution(softParameter)
    selfPosteriorIndex = 0
    concernedAgentId = 2
    competeIntention = (0, ())
    cooperateIntention = (0, tuple(range(numSheep, numSheep + numWolves)))
    updateColorSpaceByPosterior = lambda colorSpace, posterior : updateColorSpace(
            colorSpace, [softFunction(individualPosterior) for individualPosterior in posterior], 
            selfPosteriorIndex, concernedAgentId, competeIntention, cooperateIntention)
    
    #updateColorSpaceByPosterior = lambda originalColorSpace, posterior : originalColorSpace
    outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
    outsideCircleColor = np.array([[0, 0, 0]] * numWolves)
    outsideCircleSize = 15 
    drawCircleOutside = DrawCircleOutside(screen, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize)
    drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
    
   # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = InterpolateOneFrame(stayInBoundaryByReflectVelocity)
    numFramesToInterpolate = 5
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
    #[chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[0:10]]]
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[9:10]]]

if __name__ == '__main__':
    main()
