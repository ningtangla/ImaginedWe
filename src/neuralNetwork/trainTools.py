import numpy as np
import tensorflow as tf


class CoefficientCotroller():
    def __init__(self, initCoeffs, afterCoeffs, threshold=0):
        self.actionCoeff, self.valueCoeff = initCoeffs
        self.afterActionCoeff, self.afterValueCoeff = afterCoeffs
        self.threshold = threshold
        self.update = False

    def __call__(self, evalDict):
        if evalDict is not None:
            if evalDict["actionLoss"] < self.threshold and not self.update:
                self.actionCoeff = self.afterActionCoeff
                self.valueCoeff = self.afterValueCoeff
                self.update = True
                print("Coefficients of losses Updated to {:.2f} {:.2f}".format(self.actionCoeff, self.valueCoeff))
        return self.actionCoeff, self.valueCoeff


class TrainTerminalController():
    def __init__(self, lossHistorySize, terminalThreshold):
        self.lossHistorySize = lossHistorySize
        self.lossHistory = np.ones(self.lossHistorySize)
        self.actionAccuracyHistory = np.zeros(self.lossHistorySize)
        self.valueAccuracyHistory = np.zeros(self.lossHistorySize)
        self.terminalThresHold = terminalThreshold

    def __call__(self, evalDict, stepNum):
        self.lossHistory[stepNum % self.lossHistorySize] = evalDict["loss"]
        lossChange = np.mean(np.abs(self.lossHistory - np.min(self.lossHistory)))
        self.actionAccuracyHistory[stepNum % self.lossHistorySize] = evalDict["actionAcc"]
        self.valueAccuracyHistory[stepNum % self.lossHistorySize] = evalDict["valueAcc"]

        if lossChange < self.terminalThresHold:
            return True
        return False


class TrainReporter():
    def __init__(self, maxStepNum, reportInterval, tensorBoardSummaryOn=False):
        self.maxStepNum = maxStepNum
        self.reportInterval = reportInterval
        self.tensorBoardSummaryOn = tensorBoardSummaryOn

    def __call__(self, evalDict, stepNum, writer, summary):
        if stepNum % self.reportInterval == 0 or stepNum == self.maxStepNum - 1:
            print("#{} {}".format(stepNum, evalDict))
            if self.tensorBoardSummaryOn:
                writer.add_summary(summary, stepNum)


class LearningRateModifier():
    def __init__(self, initLearningRate, decayRate, decayStep):
        self.initLearningRate = initLearningRate
        self.decayRate = decayRate
        self.decayStep = decayStep

    def __call__(self, globalStep):
        return self.initLearningRate * np.power(self.decayRate, globalStep / self.decayStep)
