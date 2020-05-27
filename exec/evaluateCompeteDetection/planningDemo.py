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
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.MDPChasing.envNoPhysics import TransitForNoPhysics, StayInBoundaryByReflectVelocity
from src.MDPChasing.policies import SoftPolicy

def updateColorSpace(colorSpace, posterior, intentionSpace, imaginedWeIds):
    intentionProbabilities = np.mean([[max(0, 1 * (posterior[individualId][intention] - 1/len(intentionSpace))) 
        for intention in intentionSpace] for individualId in imaginedWeIds], axis = 0)
    colorRepresentProbability = np.array([np.array([0, 170, 0]) * probability for probability in intentionProbabilities]) + np.array(
            [colorSpace[intention[0]] for intention in intentionSpace])
    updatedColorSpace = np.array(colorSpace).copy()
    updatedColorSpace[[intention[0] for intention in intentionSpace]] = colorRepresentProbability
    return updatedColorSpace

def main():
    DIRNAME = os.path.dirname(__file__)
    #trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithHierarchyGuidedMCTSBothWolfSheep',
    #                                'trajectories')
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithHierarchy',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    NNNumSimulations = 400
    maxRunningSteps = 50
    softParameterInPlanning = 12.5
    sheepPolicyName = 'sampleNNPolicy'
    wolfPolicyName = 'sampleNNPolicy'
    hierarchy = 0
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName,
            'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'hierarchy': hierarchy, 'NNNumSimulations': NNNumSimulations}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    numWolves = 3
    numSheep = 1
    trajectoryParameters = {'numWolves': numWolves, 'numSheep': numSheep}
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
    
    FPS = 30
    circleColorSpace = [[100, 100, 100]]*numSheep + [[255, 255, 255]] * numWolves
    circleSize = 10
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numSheep + numWolves))
    saveImage = True
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str(trajectoryParameters)
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    intentionSpace = list(it.product(range(numSheep)))
    imaginedWeIdsForInferenceSubject = list(range(numSheep, numWolves + numSheep))
    softParameter = 1
    softFunction = SoftPolicy(softParameter)
    updateColorSpaceByPosterior = lambda colorSpace, posterior : updateColorSpace(
            colorSpace, [softFunction(individualPosterior) for individualPosterior in posterior], intentionSpace, imaginedWeIdsForInferenceSubject)
    
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
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)
    numFramesToInterpolate = 5
    interpolateState = InterpolateState(numFramesToInterpolate, transit)
    
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = 3
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep, posteriorIndexInTimeStep)
    
    print(len(trajectories))
    lens = [len(trajectory) for trajectory in trajectories]
    index = np.argsort(-np.array(lens))
    print(index)
    print(trajectories[0][1])
    #[chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[0:10]]]
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[5:]]]

if __name__ == '__main__':
    main()
