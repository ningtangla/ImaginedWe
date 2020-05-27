import numpy as np
import pandas as pd
import itertools as it

class CreateIntentionSpaceGivenSelfId:
    def __init__(self, goalIds, allAgentsIdsWholeGroup):
        self.goalIds = goalIds
        self.allAgentsIdsWholeGroup = allAgentsIdsWholeGroup
        self.possibleNumAgentsInWe = list(range(2, len(self.allAgentsIdsWholeGroup) + 1))

    def __call__(self, selfId):
        possibleIdsWithDiffNumAgentsInWe = []
        for numAgentInWe in self.possibleNumAgentsInWe:
            possibleIdsInWe = list(it.combinations(self.allAgentsIdsWholeGroup, numAgentInWe))
            possibleIdsWithDiffNumAgentsInWe = possibleIdsWithDiffNumAgentsInWe + possibleIdsInWe

        intentionSpace = list(it.product(self.goalIds, possibleIdsWithDiffNumAgentsInWe))
        return intentionSpace

class CalIntentionValueGivenState:
    def __init__(self, valueFunctionListBaseOnNumAgentsInWe):
        self.valueFunctionListBaseOnNumAgentsInWe = valueFunctionListBaseOnNumAgentsInWe

    def __call__(self, intention, state):
        goalId, weIds = intention
        numAgentsInWe = len(weIds)
        valueFunctionIndexInList = (numAgentsInWe - 2)
        valueFunction = self.valueFunctionListBaseOnNumAgentsInWe[valueFunctionIndexInList]

        relativeAgentsIdsForValue = np.sort([goalId] + list(weIds))
        relativeAgentsStatesForValue = np.array(state)[relativeAgentsIdsForValue]
        value = valueFunction(relativeAgentsStatesForValue)
        return value 

class AdjustIntentionPriorGivenValueOfState:
    def __init__(self, calIntentionValue, softFunction):
        self.calIntentionValue = calIntentionValue
        self.softFunction = softFunction

    def __call__(self, intentionPrior, state):
        intentions = list(intentionPrior.keys())
        intentionValues = [self.calIntentionValue(intention, state) for intention in intentions]
        intentionValueMapping = {intention: value for intention, value in zip(intentions, intentionValues)}
        intentionsLikelihoodsGivenValues = self.softFunction(intentionValueMapping)
        unnormalizedProbabilities = [prior * likelihood for prior, likelihood 
                in zip(list(intentionPrior.values()), list(intentionsLikelihoodsGivenValues.values()))]
        normalizedProbabilities = np.array(unnormalizedProbabilities) / np.sum(unnormalizedProbabilities)
        adjustedIntentionPrior = {intention: probability 
                for intention, probability in zip(intentions, normalizedProbabilities)}
        return adjustedIntentionPrior

class UpdateIntention:
    def __init__(self, intentionPrior, endAdjustedPriorTimeStep, adjustIntentionPrior, perceptAction, inferIntentionOneStep, chooseIntention):
        self.timeStep = 0
        self.lastState = None
        self.lastAction = None
        self.intentionPrior = intentionPrior
        self.formerIntentionPriors = [intentionPrior]
        self.endAdjustedTimeStep = endAdjustedPriorTimeStep
        self.adjustIntentionPrior = adjustIntentionPrior
        self.perceptAction = perceptAction
        self.inferIntentionOneStep = inferIntentionOneStep
        self.chooseIntention = chooseIntention

    def __call__(self, state):
        if self.timeStep <= self.endAdjustedTimeStep:
            adjustedIntentionPrior = self.adjustIntentionPrior(self.intentionPrior, state)
        else:
            adjustedIntentionPrior = self.intentionPrior.copy()

        if self.timeStep == 0:
            intentionPosterior = adjustedIntentionPrior.copy()
        else:
            perceivedAction = self.perceptAction(self.lastAction)
            intentionPosterior = self.inferIntentionOneStep(adjustedIntentionPrior, self.lastState, perceivedAction)
        intention = self.chooseIntention(intentionPosterior)

        self.lastState = state.copy()
        self.intentionPrior = intentionPosterior.copy()
        self.formerIntentionPriors.append(self.intentionPrior.copy())
        self.timeStep = self.timeStep + 1
        return intention

