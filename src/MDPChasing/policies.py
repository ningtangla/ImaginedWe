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

class PolicyOnChangableIntention:
    def __init__(self, perceptAction, intentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention):
        self.lastState = None
        self.lastAction = None
        self.formerIntentionPriors = [intentionPrior]
        self.perceptAction = perceptAction
        self.intentionPrior = intentionPrior
        self.updateIntentionDistribution = updateIntentionDistribution
        self.chooseIntention = chooseIntention
        self.getStateForPolicyGivenIntention = getStateForPolicyGivenIntention
        self.policyGivenIntention = policyGivenIntention

    def __call__(self, state):
        if not isinstance(self.lastAction, type(None)):
            perceivedAction = self.perceptAction(self.lastAction)

        intentionPosterior = self.updateIntentionDistribution(self.intentionPrior, self.lastState, perceivedAction)
        intentionId = self.chooseIntention(intentionPosterior)
        stateRelativeToIntention = self.getStateForPolicyGivenIntention(state, intentionId)

        actionDist = self.policyGivenIntention(stateRelativeToIntention)

        self.lastState = state.copy()
        self.formerIntentionPriors.append(intentionPosterior.copy())
        self.intentionPrior = intentionPosterior.copy()

        return actionDist


class SoftPolicy:
    def __init__(self, softParameter):
        self.softParameter = softParameter

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        softenUnnormalizedProbabilities = np.array([np.power(probability, self.softParameter) for probability in list(actionDist.values())])
        softenNormalizedProbabilities = list(softenUnnormalizedProbabilities / np.sum(softenUnnormalizedProbabilities))
        softenActionDist = dict(zip(actions, softenNormalizedProbabilities))
        return softenActionDist


class RecordValuesForPolicyAttributes:
    def __init__(self, attributes):
        self.attributes = attributes

    def __call__(self, values, policyObjects):
        [[setattr(policy, attribute, value) for attribute, value in zip(self.attributes, copy.deepcopy(values))]
         for policy in policyObjects]
        return None


class ResetPolicy:
    def __init__(self, attributeValues, returnAttributes=None):
        self.attributeValues = attributeValues
        self.returnAttributes = returnAttributes

    def __call__(self, policyObjects):
        returnAttributeValues = None
        if self.returnAttributes:
            returnAttributeValues = list(zip(*[list(zip(*[getattr(individualPolicy, attribute).copy() for individualPolicy in policyObjects]))
                                               for attribute in self.returnAttributes]))
        [[setattr(policy, attribute, value) for attribute, value in zip(list(attributeValue.keys()), copy.deepcopy(list(attributeValue.values())))]
         for policy, attributeValue in zip(policyObjects, self.attributeValues)]
        return returnAttributeValues
