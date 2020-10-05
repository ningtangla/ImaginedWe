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

from src.visualization.drawDemo import DrawBackground, DrawCircleOutside, DrawCircleOutsideEnvMADDPG, DrawState, DrawStateEnvMADDPG, ChaseTrialWithTraj, InterpolateState
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution
from src.sampleTrajectoryTools.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
        GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, \
IsTerminal, InterpolateOneFrame, TransitWithTerminalCheckOfInterpolation

def updateColorSpace(colorSpace, posterior, goalSpace, imaginedWeIds):
    #print(posterior)
    goalProbabilities = np.mean([[max(0, (posterior[individualIndex][(goalId, tuple(imaginedWeIds))] - 1/len(goalSpace)) / (1.0 - 1/len(goalSpace))) 
        for goalId in goalSpace] for individualIndex in range(len(imaginedWeIds))], axis = 0) #goalId, weIds = intention(in posterior)
    #print(goalProbabilities)
    colorRepresentProbability = np.array([np.array([0, 170, 0]) * probability for probability in goalProbabilities]) + np.array(
            [colorSpace[goalId] for goalId in goalSpace])
    updatedColorSpace = np.array(colorSpace).copy()
    updatedColorSpace[[goalId for goalId in goalSpace]] = colorRepresentProbability
    return updatedColorSpace

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateHierarchyPlanningEnvMADDPG',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    valuePriorSoftMaxBeta = 0.0
    maxRunningSteps = 101
    deviationFor2DAction = 9.0
    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps,'valuePriorSoftMaxBeta': valuePriorSoftMaxBeta, 
            'deviationFor2DAction': deviationFor2DAction}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    numWolves = 3
    numSheep = 4
    rationalityBetaInInference = 0.5
    valuePriorEndTime = -100
    valuePriorSoftMaxBeta = 0.0
    sheepConcern = 'selfSheep'
    wolfType = 'sharedAgencyByIndividualRewardWolf'
    trajectoryParameters = {'numWolves': numWolves, 'numSheep': numSheep, 'valuePriorEndTime': valuePriorEndTime, 
            'valuePriorSoftMaxBeta': valuePriorSoftMaxBeta, 'rationalityBetaInInference': rationalityBetaInInference,
            'sheepConcern': sheepConcern, 'wolfType': wolfType}
    trajectories = loadTrajectories(trajectoryParameters) 
    
    # generate demo image
    screenWidth = 700
    screenHeight = 700
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 700]
    yBoundary = [0, 700]
    lineColor = THECOLORS['white']
    lineWidth = 4
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    
    FPS = 10
    numBlocks = 2
    wolfColor = [255, 255, 255]
    sheepColor = [80, 80, 80]
    blockColor = [200, 200, 200]
    circleColorSpace = [wolfColor] * numWolves + [sheepColor]*numSheep + [blockColor] * numBlocks
    viewRatio = 1.5
    sheepSize = int(0.05 * screenWidth / (2 * viewRatio))
    wolfSize = int(0.075 * screenWidth / (3 * viewRatio))
    blockSize = int(0.2 * screenWidth / (3 * viewRatio))
    circleSizeSpace = [wolfSize] * numWolves + [sheepSize] * numSheep + [blockSize] * numBlocks
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numWolves + numSheep + numBlocks))
    saveImage = True
    #saveImage = False
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str('forDemo')
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    goalSpace = list(range(numWolves, numWolves + numSheep))
    imaginedWeIdsForInferenceSubject = list(range(numWolves))
    softParameter = 0.9
    softFunction = SoftDistribution(softParameter)
    updateColorSpaceByPosterior = lambda colorSpace, posterior : updateColorSpace(
            colorSpace, [softFunction(individualPosterior) for individualPosterior in posterior], goalSpace, imaginedWeIdsForInferenceSubject)
    
    #updateColorSpaceByPosterior = lambda originalColorSpace, posterior : originalColorSpace
    outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
    outsideCircleColor = np.array([[255, 0, 0]] * numWolves)
    outsideCircleSize = int(wolfSize * 1.5)
    drawCircleOutside = DrawCircleOutsideEnvMADDPG(screen, viewRatio, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize)
    drawState = DrawStateEnvMADDPG(FPS, screen, viewRatio, circleColorSpace, circleSizeSpace, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
    
   # MDP Env
    interpolateState = None
    
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = 4
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep, posteriorIndexInTimeStep)
    
    print(len(trajectories))
    lens = [len(trajectory) for trajectory in trajectories]
    index = np.argsort(-np.array(lens))
    print(index)
    print(trajectories[0][1])
    #[chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[16:20]]]#3v4 rational0.5, no 3, 16, 18 
    print([len(trajectory) for trajectory in np.array(trajectories)[index[:]]])
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[[3, 16, 18]]]]

if __name__ == '__main__':
    main()
