import numpy as np
import random
import copy
from scipy import stats

def stationaryAgentPolicy(state):
    return {(0, 0): 1}


class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        actionDist = {action: 1 / len(self.actionSpace) for action in self.actionSpace}
        return actionDist


class HeatSeekingDiscreteDeterministicPolicy:
    def __init__(self, actionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors):
        self.actionSpace = actionSpace
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.computeAngleBetweenVectors = computeAngleBetweenVectors

    def __call__(self, state):
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        heatSeekingVector = np.array(preyPosition) - np.array(predatorPosition)
        anglesBetweenHeatSeekingAndActions = np.array([self.computeAngleBetweenVectors(heatSeekingVector, np.array(action)) for action in self.actionSpace]).flatten()
        minIndex = np.argwhere(anglesBetweenHeatSeekingAndActions == np.min(anglesBetweenHeatSeekingAndActions)).flatten()
        actionsShareProbability = [tuple(self.actionSpace[index]) for index in minIndex]
        actionDist = {action: 1 / len(actionsShareProbability) if action in actionsShareProbability else 0 for action in self.actionSpace}
        return actionDist

class HeatSeekingContinuesDeterministicPolicy:
    def __init__(self, getPredatorPos, getPreyPos, actionMagnitude):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):
        action = np.array(self.getPreyPos(state)) - np.array(self.getPredatorPos(state))
        actionL2Norm = np.linalg.norm(action, ord=2)
        assert actionL2Norm != 0
        action = action / actionL2Norm
        action *= self.actionMagnitude

        actionTuple = tuple(action)
        actionDist = {actionTuple: 1}
        return actionDist

class HeatSeekingDiscreteStochasticPolicy:
    def __init__(self, assumePrecision, actionSpace, getPredatorPos, getPreyPos):
        self.assumePrecision = assumePrecision
        self.actionSpace = actionSpace
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.vecToAngle = lambda vector: np.angle(complex(vector[0], vector[1]))
        self.degreeList = [self.vecToAngle(vector) for vector in self.actionSpace]

    def __call__(self, state):
        heatseekingVector = self.getPreyPos(state) - self.getPredatorPos(state)
        heatseekingDirection = self.vecToAngle(heatseekingVector)
        pdf = np.array([stats.vonmises.pdf(heatseekingDirection - degree, self.assumePrecision) * 2 for degree in self.degreeList])
        normProb = pdf / pdf.sum()
        actionDict = {action: prob for action, prob in zip(self.actionSpace, normProb)}
        return actionDict



