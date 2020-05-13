import numpy as np
import pygame as pg
from pygame.color import THECOLORS
from pygame import time
import collections as co
import pickle
from src.visualization import DrawBackground, DrawNewState, DrawImage, drawText
from src.controller import HumanController, ModelController
from src.updateWorld import InitialWorld
import random
import os


class AttributionTrail:
    def __init__(self, totalScore, saveImageDir, saveImage, drawAttributionTrail):
        self.totalScore = totalScore
        self.actionDict = [{pg.K_LEFT: -1, pg.K_RIGHT: 1}, {pg.K_a: -1, pg.K_d: 1}]
        self.comfirmDict = [pg.K_RETURN, pg.K_SPACE]
        self.distributeUnit = 0.1
        self.drawAttributionTrail = drawAttributionTrail
        self.saveImageDir = saveImageDir
        self.saveImage = saveImage

    def __call__(self, eatenFlag, hunterFlag, timeStepforDraw):
        hunterid = hunterFlag.index(True)
        attributionScore = [0, 0]
        attributorPercent = 0.5
        pause = True
        screen = self.drawAttributionTrail(hunterid, attributorPercent)
        pg.event.set_allowed([pg.KEYDOWN])

        attributionDelta = 0
        stayAttributionBoudray = lambda attributorPercent: max(min(attributorPercent, 1), 0)
        while pause:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                if event.type == pg.KEYDOWN:
                    print(event.key)
                    if event.key in self.actionDict[hunterid].keys():
                        attributionDelta = self.actionDict[hunterid][event.key] * self.distributeUnit
                        print(attributionDelta)

                        attributorPercent = stayAttributionBoudray(attributorPercent + attributionDelta)

                        screen = self.drawAttributionTrail(hunterid, attributorPercent)
                    elif event.key == self.comfirmDict[hunterid]:
                        pause = False
            pg.time.wait(10)
            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(screen, self.saveImageDir + '/' + format(timeStepforDraw, '04') + ".png")
                timeStepforDraw += 1

        recipentPercent = 1 - attributorPercent
        if hunterid == 0:
            attributionScore = [int(self.totalScore * attributorPercent), int(self.totalScore * recipentPercent)]
        else:  # hunterid=1
            attributionScore = [int(self.totalScore * recipentPercent), int(self.totalScore * attributorPercent)]

        return attributionScore, timeStepforDraw


def calculateGridDistance(gridA, gridB):
    return np.linalg.norm(np.array(gridA) - np.array(gridB), ord=2)


def isAnyKilled(humanGrids, targetGrid, killzone):
    return np.any(np.array([calculateGridDistance(humanGrid, targetGrid) for humanGrid in humanGrids]) < killzone)


class CheckEaten:
    def __init__(self, killzone, isAnyKilled):
        self.killzone = killzone
        self.isAnyKilled = isAnyKilled

    def __call__(self, targetPositions, playerPositions):
        eatenFlag = [False] * len(targetPositions)
        hunterFlag = [False] * len(targetPositions)
        for (i, targetPosition) in enumerate(targetPositions):
            if self.isAnyKilled(playerPositions, targetPosition, self.killzone):
                eatenFlag[i] = True
                break
        for (i, playerPosition) in enumerate(playerPositions):
            if self.isAnyKilled(targetPositions, playerPosition, self.killzone):
                hunterFlag[i] = True
                break
        return eatenFlag, hunterFlag


class CheckTerminationOfTrial:
    def __init__(self, finishTime):
        self.finishTime = finishTime

    def __call__(self, actionList, eatenFlag, currentStopwatch):
        for action in actionList:
            if np.any(eatenFlag) == True or action == pg.QUIT or currentStopwatch >= self.finishTime:
                pause = False
            else:
                pause = True
        return pause


class Trial():
    def __init__(self, actionSpace, killzone, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten, attributionTrail, humanController):
        self.humanController = humanController
        self.actionSpace = actionSpace
        self.killzone = killzone
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.beanReward = 1
        self.attributionTrail = attributionTrail
        self.checkEaten = checkEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        self.memorySize = 25

    def __call__(self, targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw):
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])

        from collections import deque
        dequeState = deque(maxlen=self.memorySize)
        pause = True
        while pause:
            dequeState.append([np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState)
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)
            pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        addSocre = [0, 0]
        if True in eatenFlag[:2]:
            addSocre, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["beanEaten"] = eatenFlag.index(True) + 1
        elif True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward

        else:
            results["beanEaten"] = 0
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score = np.add(score, addSocre)
        return results, targetPositions, playerPositions, score, currentStopwatch, eatenFlag, timeStepforDraw


def main():
    dimension = 21
    bounds = [0, 0, dimension - 1, dimension - 1]
    minDistanceBetweenGrids = 5
    condition = [-5, -3, -1, 0, 1, 3, 5]
    initialWorld = InitialWorld(bounds)
    pg.init()
    screenWidth = 720
    screenHeight = 720
    screen = pg.display.set_mode((screenWidth, screenHeight))
    gridSize = 21
    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    stopwatchUnit = 10
    textColorTuple = (255, 50, 50)
    stopwatchEvent = pg.USEREVENT + 1
    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
    finishTime = 90000
    currentStopwatch = 32888
    score = 0
    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    humanController = HumanController(gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)
    policy = pickle.load(open("SingleWolfTwoSheepsGrid15.pkl", "rb"))
    modelController = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)
    trial = Trial(modelController, drawNewState, stopwatchEvent, finishTime)
    bean1Grid, bean2Grid, playerGrid = initialWorld(minDistanceBetweenGrids)
    bean1Grid = (3, 13)
    bean2Grid = (5, 0)
    playerGrid = (0, 8)
    results = trial(bean1Grid, bean2Grid, playerGrid, score, currentStopwatch)


if __name__ == "__main__":
    main()
