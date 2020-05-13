import pygame as pg
import numpy as np
import os
from pygame.color import THECOLORS


def calculateIncludedAngle(vector1, vector2):
    includedAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
    return includedAngle


def findQuadrant(vector):
    quadrant = 0
    if vector[0] > 0 and vector[1] > 0:
        quadrant = 0


class InitializeScreen:
    def __init__(self, screenWidth, screenHeight, fullScreen):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.fullScreen = fullScreen

    def __call__(self):
        pg.init()
        if self.fullScreen:
            screen = pg.display.set_mode((self.screenWidth, self.screenHeight), pg.FULLSCREEN)
        else:
            screen = pg.display.set_mode((self.screenWidth, self.screenHeight))
        pg.display.init()
        pg.fastevent.init()
        return screen


def drawText(screen, text, textColorTuple, textPositionTuple):
    font = pg.font.Font(None, 50)
    textObj = font.render(text, 1, textColorTuple)
    screen.blit(textObj, textPositionTuple)
    return


class GiveExperimentFeedback():
    def __init__(self, screen, textColorTuple, screenWidth, screenHeight):
        self.screen = screen
        self.textColorTuple = textColorTuple
        self.screenHeight = screenHeight
        self.screenWidth = screenWidth

    def __call__(self, trialIndex, score):
        self.screen.fill((0, 0, 0))
        for j in range(trialIndex + 1):
            drawText(self.screen, "No. " + str(j + 1) + " experiment" + "  score: " + str(score[j]), self.textColorTuple,(self.screenWidth / 5, self.screenHeight * (j + 3) / 12))
        pg.display.flip()
        pg.time.wait(3000)


class DrawBackground():
    def __init__(self, screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple,playerColors):
        self.screen = screen
        self.gridSize = gridSize
        self.leaveEdgeSpace = leaveEdgeSpace
        self.widthLineStepSpace = np.int(screen.get_width() / (gridSize + 2 * self.leaveEdgeSpace))
        self.heightLineStepSpace = np.int(screen.get_height() / (gridSize + 2 * self.leaveEdgeSpace))
        self.backgroundColor = backgroundColor
        self.lineColor = lineColor
        self.lineWidth = lineWidth
        self.textColorTuple = textColorTuple
        self.playerColors=playerColors
    def __call__(self, currentTime, currentScore):
        self.screen.fill((0, 0, 0))
        pg.draw.rect(self.screen, self.backgroundColor, pg.Rect(np.int(self.leaveEdgeSpace * self.widthLineStepSpace), np.int(self.leaveEdgeSpace * self.heightLineStepSpace), np.int(self.gridSize * self.widthLineStepSpace), np.int(self.gridSize * self.heightLineStepSpace)))

        seconds = currentTime / 1000
        drawText(self.screen, 'Time: ' + str("%4.1f" % seconds) + 's', THECOLORS['white'], (self.widthLineStepSpace * 5, self.leaveEdgeSpace * 3))
        drawText(self.screen, '1P: ' + str(currentScore[0]), self.playerColors[0], (self.widthLineStepSpace * 35  , self.leaveEdgeSpace * 3))
        drawText(self.screen, '2P: ' + str(currentScore[1]), self.playerColors[1], (self.widthLineStepSpace * 50, self.leaveEdgeSpace * 3))
        return


class DrawNewState():
    def __init__(self, screen, drawBackground, targetColors, playerColors, targetRadius, playerRadius):
        self.screen = screen
        self.drawBackground = drawBackground
        self.targetColors = targetColors
        self.playerColors = playerColors
        self.targetRadius = targetRadius
        self.playerRadius = playerRadius
        self.leaveEdgeSpace = drawBackground.leaveEdgeSpace
        self.widthLineStepSpace = drawBackground.widthLineStepSpace
        self.heightLineStepSpace = drawBackground.heightLineStepSpace

    def __call__(self, targetPositions, playerPositions, currentTime, currentScore):
        self.drawBackground(currentTime, currentScore)

        for targetPosition, targetColor in zip(targetPositions[:2], self.targetColors[:2]):
            pg.draw.circle(self.screen, targetColor, [np.int((targetPosition[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((targetPosition[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.targetRadius+2)

        for targetPosition, targetColor in zip(targetPositions[2:], self.targetColors[2:]):
            pg.draw.circle(self.screen, targetColor, [np.int((targetPosition[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((targetPosition[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.targetRadius)

        for playerPosition, playerColor in zip(playerPositions, self.playerColors):
            pg.draw.circle(self.screen, playerColor, [np.int((playerPosition[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((playerPosition[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.playerRadius)
        return self.screen


class DrawImage():
    def __init__(self, screen):
        self.screen = screen
        self.screenCenter = (self.screen.get_width() / 2, self.screen.get_height() / 2)

    def __call__(self, image):
        imageRect = image.get_rect()
        imageRect.center = self.screenCenter
        pause = True
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
        self.screen.fill((0, 0, 0))
        self.screen.blit(image, imageRect)
        pg.display.flip()
        while pause:
            pg.time.wait(10)
            for event in pg.event.get():
                if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    pause = False
                elif event.type == pg.QUIT:
                    pg.quit()
            pg.time.wait(10)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

class DrawAttributionTrail:
    def __init__(self, screen,playerColors,totalBarLength,barHeight,screenCenter):
        self.screen = screen
        self.playerColors=playerColors
        self.screenCenter =screenCenter
        self.totalBarLength=totalBarLength
        self.barHeight=barHeight

    def __call__(self, attributorId,attributorPercent):
        print(attributorId)
        recipentId=1-attributorId
        attributorLen=int(self.totalBarLength*attributorPercent)

        attributorRect=((self.screenCenter[0]-self.totalBarLength/2,self.screenCenter[1]-self.barHeight/2),(attributorLen,self.barHeight))
        recipentRect=((self.screenCenter[0]-self.totalBarLength/2+attributorLen,self.screenCenter[1]-self.barHeight/2),(self.totalBarLength-attributorLen,self.barHeight))

        pg.draw.rect(self.screen, self.playerColors[attributorId], attributorRect)
        pg.draw.rect(self.screen, self.playerColors[recipentId], recipentRect)

        pg.display.flip()
        return self.screen
if __name__ == "__main__":
    pg.init()
    screenWidth = 720
    screenHeight = 720
    screen = pg.display.set_mode((screenWidth, screenHeight))
    gridSize = 20
    leaveEdgeSpace = 2
    lineWidth = 2
    backgroundColor = [188, 188, 0]
    lineColor = [255, 255, 255]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    targetPositionA = [5, 5]
    targetPositionB = [15, 5]
    playerPosition = [10, 15]
    picturePath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Pictures/'
    restImage = pg.image.load(picturePath + 'rest.png')
    currentTime = 138456
    currentScore = 5
    textColorTuple = (255, 50, 50)
    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    drawBackground(currentTime, currentScore)
    pg.time.wait(5000)
    pg.quit()
