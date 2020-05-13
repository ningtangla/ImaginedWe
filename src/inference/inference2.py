import numpy as np
import pandas as pd

class CalPolicyLikelihood:
    def __init__(self, getStateForPolicyGivenIntention, policyGivenIntention):
        self.getStateForPolicyGivenIntention = getStateForPolicyGivenIntention
        self.policyGivenIntention = policyGivenIntention

    def __call__(self, intention, state, weActionSelfFirst):
        stateRelativeToIntention = self.getStateForPolicyGivenIntention(state, intention)
        centralControlActionDist = self.policyGivenIntention(stateRelativeToIntention)
        centralControlAction = tuple([tuple(action) for action in weActionSelfFirst])
        policyLikelihood = centralControlActionDist[centralControlAction]
        return policyLikelihood

class InferOneStep:
    def __init__(self, priorDecayRate, jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood):	
        self.priorDecayRate = priorDecayRate
        self.jointHypothesisSpace = jointHypothesisSpace
        self.concernedHypothesisVariable = concernedHypothesisVariable	
        self.calJointLikelihood = calJointLikelihood

    def __call__(self, intentionPrior, state, perceivedAction):
        jointHypothesisDf = pd.DataFrame(index = self.jointHypothesisSpace)
        intentions = jointHypothesisDf.index.get_level_values('intention')
        jointHypothesisDf['likelihood'] = [self.calJointLikelihood(intention, state, perceivedAction) for intention in intentions]
        #__import__('ipdb').set_trace()
        #jointHypothesisDf['jointLikelihood'] = jointHypothesisDf.apply(lambda row: self.calJointLikelihood(row.name[0], state, row.name[1], nextState))
        marginalLikelihood = jointHypothesisDf.groupby(self.concernedHypothesisVariable).sum()
        oneStepLikelihood = marginalLikelihood['likelihood'].to_dict()
        decayedLogPrior = {key: np.log(value) * self.priorDecayRate for key, value in intentionPrior.items()}
        unnomalizedPosterior = {key: np.exp(decayedLogPrior[key] + np.log(oneStepLikelihood[key] + 0.001)) for key in list(intentionPrior.keys())}
        normalizedProbabilities = np.array(list(unnomalizedPosterior.values())) / np.sum(list(unnomalizedPosterior.values()))
        normalizedPosterior = dict(zip(list(unnomalizedPosterior.keys()),normalizedProbabilities))
        return normalizedPosterior

class InferOnTrajectory:
    def __init__(self, prior, observe, inferOneStepLik, visualize):
        self.prior = prior
        self.observe = observe
        self.updateOneStepLik = updateOneStepLik        
        self.visualize = visualize

    def __call__(self, trajectory):
        initState = self.observe(trajectory[0])
        if self.visualize: 
            self.visualize(initState, self.prior)
        
        lastState = initState.copy() 
        prior = self.prior.copy()
        posteriorsWholeTrajectory = [prior]
        for timeStepIndex in range(1, len(trajectory)):
            state = self.observe(trajectory[timeStepIndex]) 
            posterior = self.inferOneStepLik(prior, lastState, state)
            if self.visualize:              
                self.visualize(state, posterior)
            lastState = state
            posteriorsWholeTrajectory.append(posterior)
            prior = posterior

        return posteriorsWholeTrajectory
