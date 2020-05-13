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
    colorRepresentProbability = np.array([np.array(colorSpace[individualId]) * 2 * (1 - max(list(posterior[individualId].values()))) +
        np.sum([colorSpace[intention[0]] * max(0, 2 * (posterior[individualId][intention] - 1/len(intentionSpace))) 
        for intention in intentionSpace], axis = 0) for individualId in imaginedWeIds])
    updatedColorSpace = colorSpace.copy()
    updatedColorSpace[imaginedWeIds] = colorRepresentProbability
    return updatedColorSpace

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithNoisePerception',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    maxRunningSteps = 100
    softParameterInPlanning = 2.5
    sheepPolicyName = 'sampleNNPolicy'
    wolfPolicyName = 'sampleNNPolicy'
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName,
        'policySoftParameter': softParameterInPlanning}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectoryParameters = {'perceptNoiseForAll': 1e-1, 'maxRunningSteps': maxRunningSteps}
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
    
    FPS = 40
    circleColorSpace = np.array([[0, 0, 255], [0, 255, 0], [255, 255, 255], [255, 255, 255]])
    #circleColorSpace = [THECOLORS['green'], THECOLORS['green'], THECOLORS['red'], THECOLORS['red']]
    circleSize = 10
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(4))
    saveImage = True
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str(trajectoryParameters)
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    intentionSpace = [(0,), (1,)]
    imaginedWeIdsForInferenceSubject = [2, 3]
    softParameter = 0.1
    softFunction = SoftPolicy(softParameter)
    updateColorSpaceByPosterior = lambda colorSpace, posterior : updateColorSpace(
            colorSpace, [softFunction(individualPosterior) for individualPosterior in posterior], intentionSpace, imaginedWeIdsForInferenceSubject)
    
    outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
    outsideCircleColor = np.array([[255, 0, 0]] * 2) 
    outsideCircleSize = 15 
    drawCircleOutside = DrawCircleOutside(screen, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize)
    drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
    
    # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)
    numFramesToInterpolate = 3
    interpolateState = InterpolateState(numFramesToInterpolate, transit)
    
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    posteriorIndexInTimeStep = 3
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep, posteriorIndexInTimeStep)
   
    print(len(trajectories))
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[0:]]
    #[52, 63 for imaginedWe in 1.01e-1, 2, 5 for commitment in 1e-1]
if __name__ == '__main__':
    main()
