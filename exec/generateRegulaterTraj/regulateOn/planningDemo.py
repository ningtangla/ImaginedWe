import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))

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

def updateColor(color, individualPosterior, individualCommitted):
    if individualCommitted and individualPosterior:
        adjustedColorFloat = individualPosterior[(0,)] * np.array([0, 255, 0]) + individualPosterior[(1,)] * np.array([0, 0, 255]) 
        adjustedColor = [int(channel) for channel in adjustedColorFloat]
    else:
        adjustedColor = color
    return adjustedColor 

class DrawCircleOutside:
    def __init__(self, screen, outsideCircleAgentIds, positionIndex, circleColors, circleSize, updateColorByPosterior):
        self.screen = screen
        self.outsideCircleAgentIds = outsideCircleAgentIds
        self.xIndex, self.yIndex = positionIndex
        self.circleColors = circleColors
        self.circleSize = circleSize
        self.updateColorByPosterior = updateColorByPosterior

    def __call__(self, state, warn, committed, posterior = None):
        for agentIndex in self.outsideCircleAgentIds:
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            color = self.circleColors[list(self.outsideCircleAgentIds).index(agentIndex)][committed[agentIndex]]
            agentColor = tuple(self.updateColorByPosterior(color, posterior[agentIndex], committed[agentIndex]))
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize[warn[agentIndex]])
        return

class DrawState:
    def __init__(self, fps, screen, colorSpace, circleSize, agentIdsToDraw, positionIndex, saveImage, imagePath, 
            drawBackGround, drawCircleOutside = None, updateSizeByWarn = None):
        self.fps = fps
        self.screen = screen
        self.colorSpace = colorSpace
        self.circleSize = circleSize
        self.agentIdsToDraw = agentIdsToDraw
        self.xIndex, self.yIndex = positionIndex
        self.saveImage = saveImage
        self.imagePath = imagePath
        self.drawBackGround = drawBackGround
        self.drawCircleOutside = drawCircleOutside
        self.updateSizeByWarn = updateSizeByWarn

    def __call__(self, state, warn, committed, posterior = None):
        fpsClock = pg.time.Clock()
        
        self.drawBackGround()
        circleColors = self.colorSpace
        if self.drawCircleOutside:
            self.drawCircleOutside(state, warn, committed, posterior)
        for agentIndex in self.agentIdsToDraw:
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = tuple(circleColors[agentIndex])
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)

        pg.display.flip()
        
        if self.saveImage == True:
            filenameList = os.listdir(self.imagePath)
            pg.image.save(self.screen, self.imagePath + '/' + str(len(filenameList))+'.png')
        
        fpsClock.tick(self.fps)
        return self.screen

class ChaseTrialWithTraj:
    def __init__(self, stateIndex, drawState, interpolateState = None, actionIndex = None, posteriorIndex=None, commitmentWarnIndex = None, committedIndex = None):
        self.stateIndex = stateIndex
        self.drawState = drawState
        self.interpolateState = interpolateState
        self.actionIndex = actionIndex
        self.posteriorIndex = posteriorIndex
        self.commitmentWarnIndex = commitmentWarnIndex
        self.committedIndex = committedIndex

    def __call__(self, trajectory):
        for timeStepIndex in range(len(trajectory)):
            timeStep = trajectory[timeStepIndex]
            state = timeStep[self.stateIndex]
            action = timeStep[self.actionIndex]
            commitmentWarn = timeStep[self.commitmentWarnIndex]
            committed = timeStep[self.committedIndex]
            if self.posteriorIndex:
                posterior = timeStep[self.posteriorIndex] 
            else:
                posterior = None
            if self.interpolateState and timeStepIndex!= len(trajectory) - 1:
                statesToDraw = self.interpolateState(state, action)
            else:
                statesToDraw  = [state]
            for state in statesToDraw:
                screen = self.drawState(state, commitmentWarn, committed, posterior)
        return

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', '..', 'data', 'regulateOn',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    reCommitProbability = 2
    precision = 1.83
    regulateOn = 1
    numOneWolfActionSpace = 8
    NNNumSimulations = 300 #300 with distance Herustic; 200 without distanceHerustic
    numWolves = 2
    maxRunningSteps = 101
    softParameterInPlanning = 2.5
    sheepPolicyName = 'sampleNNPolicy'
    wolfPolicyName = 'sampleNNPolicy'
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName, 'NNNumSimulations': NNNumSimulations,
            'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'numOneWolfActionSpace': numOneWolfActionSpace, 'numWolves': numWolves,
            'reCommitedProbability': reCommitProbability, 'regulateOn': regulateOn, 'precision': precision}

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    numWolves = 2
    trajectoryParameters = {'numWolves': numWolves}
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
    
    FPS = 25
    numSheep = 2
    circleColorSpace = [[0, 255, 0], [0, 0, 255]] + [[255, 255, 255]] * numWolves
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
    
    #updateColorSpaceByPosterior = lambda originalColorSpace, posterior : originalColorSpace
    outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
    outsideCircleColor = [{0: [0, 0, 255], 1:[0, 255, 0]}] * len(imaginedWeIdsForInferenceSubject)
    outsideCircleSize = {0: 15, 1: 25}
    drawCircleOutside = DrawCircleOutside(screen, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize, updateColor)
    circleSize = 10
    drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, drawCircleOutside)
    
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
    commitmentWarnIndexInTimeStep = 4
    committedIndexInTimeStep = 5
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep, posteriorIndexInTimeStep,
            commitmentWarnIndexInTimeStep, committedIndexInTimeStep)
   
    print(len(trajectories))
    print(trajectories[0][0])
    
    lens = [len(trajectory) for trajectory in trajectories]
    index = np.argsort(-np.array(lens))
    print(index)
    #print(trajectories[0][1])
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[[20, 23, 27]]]]
    #[chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[22:]]]
    #[24 for 8intentions]
if __name__ == '__main__':
    main()
