import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import collections as co
import itertools as it
import numpy as np
import pickle
import pygame as pg
from pygame.color import THECOLORS
from src.visualization import DrawBackground, DrawNewState, DrawImage, GiveExperimentFeedback, InitializeScreen, DrawAttributionTrail
from src.controller import HumanController, ModelController
from src.updateWorld import InitialWorld, UpdateWorld, StayInBoundary
from src.writer import WriteDataFrameToCSV
from src.trial import Trial, AttributionTrail, isAnyKilled, CheckEaten, CheckTerminationOfTrial
from src.experiment import Experiment
from src.sheepPolicy import GenerateModel, restoreVariables, ApproximatePolicy, chooseGreedyAction, sampleAction, SoftmaxAction


def main():
    gridSize = 60
    bounds = [0, 0, gridSize - 1, gridSize - 1]
    minDistanceForReborn = 5
    condition = [-5, -3, -1, 0, 1, 3, 5]
    counter = [0] * len(condition)
    numPlayers = 2
    initialWorld = InitialWorld(bounds, numPlayers, minDistanceForReborn)
    updateWorld = UpdateWorld(bounds, condition, counter, minDistanceForReborn)

    screenWidth = 800
    screenHeight = 800
    screenCenter = [screenWidth / 2, screenHeight / 2]
    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()

    leaveEdgeSpace = 6
    lineWidth = 1
    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [THECOLORS['blue'], (0, 0, 128), (0, 168, 107), (0, 168, 107)]  # [255, 50, 50]
    playerColors = [THECOLORS['orange'], THECOLORS['red']]
    targetRadius = 10
    playerRadius = 10
    totalBarLength = 100
    barHeight = 20
    stopwatchUnit = 100
    finishTime = 1000 * 60 * 2
    block = 1
    softmaxBeita = -1
    textColorTuple = THECOLORS['green']
    stopwatchEvent = pg.USEREVENT + 1

    saveImage = True
    killzone = 2
    wolfSpeedRatio = 1

    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
    pg.key.set_repeat(120, 120)
    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))
    experimentValues = co.OrderedDict()
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    experimentValues["name"] = 'kill' + str(killzone)
    experimentValues["condition"] = 'all'
    writerPath = os.path.join(resultsPath, experimentValues["name"]) + '.csv'
    writer = WriteDataFrameToCSV(writerPath)

    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple, playerColors)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    drawAttributionTrail = DrawAttributionTrail(screen, playerColors, totalBarLength, barHeight, screenCenter)
    saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'), experimentValues["name"])

    xBoundary = [bounds[0], bounds[2]]
    yBoundary = [bounds[1], bounds[3]]
    stayInBoundary = StayInBoundary(xBoundary, yBoundary)
############
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    preyPowerRatio = 3
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    sheepActionSpace.append((0, 0))
    numActionSpace = len(sheepActionSpace)

    actionSpaceStill = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    sheepActionSpaceStill = list(map(tuple, np.array(actionSpaceStill) * preyPowerRatio))
    sheepActionSpaceStill.append((0, 0))
    numActionSpaceStill = len(sheepActionSpaceStill)

    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    depth = 5
    generateSheepModelSingle = GenerateModel(4, numActionSpace, regularizationFactor)
    generateSheepModelMulti = GenerateModel(6, numActionSpaceStill, regularizationFactor)
    initSheepNNModelSingle = generateSheepModelSingle(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
    initSheepNNModelMulti = generateSheepModelMulti(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
    sheepPreTrainModelPathSingle = os.path.join('..', 'trainedModelsSingle', 'agentId=0_dataSize=5000_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=100_sampleOneStepPerTraj=0_trainSteps=50000')
    sheepPreTrainModelPathMulti = os.path.join('..', 'trainedModelsMulti', 'agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    sheepPreTrainModelSingle = restoreVariables(initSheepNNModelSingle, sheepPreTrainModelPathSingle)
    sheepPreTrainModelMulti = restoreVariables(initSheepNNModelMulti, sheepPreTrainModelPathMulti)

    sheepPolicySingleModel = ApproximatePolicy(sheepPreTrainModelSingle, sheepActionSpace)
    sheepPolicyMulti = ApproximatePolicy(sheepPreTrainModelMulti, sheepActionSpaceStill)

    from src.sheepPolicy import SingleChasingPolicy, inferNearestWolf, ComputeLikelihoodByHeatSeeking, InferCurrentWolf, BeliefPolicy
    sheepPolicySingle = SingleChasingPolicy(sheepPolicySingleModel, inferNearestWolf)
    baseProb = 0.5
    assumePrecision = 50
    computeLikelihoodByHeatSeeking = ComputeLikelihoodByHeatSeeking(baseProb, assumePrecision)
    inferCurrentWolf = InferCurrentWolf(computeLikelihoodByHeatSeeking)
    beliefPolicy = BeliefPolicy(sheepPolicySingle, sheepPolicySingleModel, sheepPolicyMulti, inferCurrentWolf)
    # sheepPolicySingle = sheepPolicySingleModel

    # ## mcts sheep
    # from mcts import sheepMCTS
    # sheepPolicy = sheepMCTS()
    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    checkEaten = CheckEaten(killzone, isAnyKilled)
    totalScore = 10
    attributionTrail = AttributionTrail(totalScore, saveImageDir, saveImage, drawAttributionTrail)

    # sheepPolicy = [sheepPolicySingle,sheepPolicyMulti]

    softMaxBeta = 30
    softmaxAction = SoftmaxAction(softMaxBeta)
    humanController = HumanController(writer, gridSize, stopwatchEvent, stopwatchUnit, wolfSpeedRatio, drawNewState, finishTime, stayInBoundary, saveImage, saveImageDir, beliefPolicy, chooseGreedyAction)
    # policy = pickle.load(open("SingleWolfTwoSheepsGrid15.pkl","rb"))
    # modelController = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, softmaxBeita)

    actionSpace = list(it.product([0, 1, -1], repeat=2))

    trial = Trial(actionSpace, killzone, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten, attributionTrail, humanController)
    experiment = Experiment(trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath)
    giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)

    # drawImage(introductionImage)
    score = [0] * block
    for i in range(block):
        score[i] = experiment(finishTime)
        giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
        # else:
            # drawImage(restImage)

    participantsScore = np.sum(np.array(score))
    print(participantsScore)


if __name__ == "__main__":
    main()
