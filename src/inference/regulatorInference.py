import numpy as np
import pandas as pd

class GetActionLikelihoodOnIndividualPlanning:
    def __init__(self, indivualIndex):
        self.indivualIndex = indivualIndex
    
    def __call__(self, actionDist, perceivedAction):
        indivualAction = tuple(perceivedAction[self.indivualIndex])
        actionLikelihood = actionDist[indivualAction]
        normalizedActionLikelihood = actionLikelihood / np.sum(list(actionDist.values()))
        return normalizedActionLikelihood

class GetPartnerActionLikelihoodOnJointPlanning:
    def __init__(self, weIndex, selfIndex):
        self.weIndex = weIndex
        self.selfIndex = selfIndex

    def __call__(self, actionDist, percivedAction):
        selfAction = tuple(percivedAction[self.selfIndex])
        weAction = [tuple(percivedAction[Index]) for Index in self.weIndex]
        actionSpace = list(actionDist.keys())
        actionProbabilities = list(actionDist.values())
        selfActionMatchFlags = [possibleAction[self.selfIndex] == selfAction for possibleAction in actionSpace]
        weActionMatchFlags = [list(possibleAction) == weAction for possibleAction in actionSpace]
        actionLikelihood = float(np.array(actionProbabilities)[np.nonzero(weActionMatchFlags)])
        normalizedActionLikelihood = actionLikelihood / np.sum(np.array(actionProbabilities)[np.nonzero(selfActionMatchFlags)])
        return normalizedActionLikelihood

class InferPartnerCommitment:
    def __init__(self, priorDecayRate, jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood, perceptNextState = None):	
        self.priorDecayRate = priorDecayRate
        self.jointHypothesisSpace = jointHypothesisSpace
        self.concernedHypothesisVariable = concernedHypothesisVariable	
        self.calJointLikelihood = calJointLikelihood

    def __call__(self, commitmentPrior, state, perceivedAction):
        jointHypothesisDf = pd.DataFrame(index = self.jointHypothesisSpace)
        commitments = jointHypothesisDf.index.get_level_values('commitment')
        jointHypothesisDf['likelihood'] = [self.calJointLikelihood(commitment, state, perceivedAction) for commitment in commitments]
        #__import__('ipdb').set_trace()
        #jointHypothesisDf['jointLikelihood'] = jointHypothesisDf.apply(lambda row: self.calJointLikelihood(row.name[0], state, row.name[1], nextState))
        marginalLikelihood = jointHypothesisDf.groupby(self.concernedHypothesisVariable).sum()
        oneStepLikelihood = marginalLikelihood['likelihood'].to_dict()
        decayedLogPrior = {key: np.log(value) * self.priorDecayRate for key, value in commitmentPrior.items()}
        unnomalizedPosterior = {key: np.exp(decayedLogPrior[key] + np.log(oneStepLikelihood[key])) for key in list(commitmentPrior.keys())}
        normalizedProbabilities = np.array(list(unnomalizedPosterior.values())) / np.sum(list(unnomalizedPosterior.values()))
        normalizedPosterior = dict(zip(list(unnomalizedPosterior.keys()),normalizedProbabilities))
        return normalizedPosterior

