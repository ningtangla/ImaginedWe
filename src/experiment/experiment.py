import numpy as np
class Experiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath

    def __call__(self, finishTime):
        targetPositions, playerGrid = self.initialWorld()
        trialIndex = 0
        score =np.array ([0,0])
        currentStopwatch = 0
        timeStepforDraw=0
        while True:
            print('trialIndex', trialIndex)
            response = self.experimentValues.copy()
            results, targetPositions, playerGrid, score, currentStopwatch, eatenFlag,timeStepforDraw = self.trial(targetPositions, playerGrid, score, currentStopwatch, trialIndex,timeStepforDraw)
            response.update(results)
            self.writer(response, trialIndex)
            if currentStopwatch >= finishTime:
                break
            targetPositions = self.updateWorld(targetPositions, playerGrid, eatenFlag)
            trialIndex += 1
        return score
