import numpy as np
import pandas as pd

class CalUncommittedAgentsPolicyLikelihood:
    def __init__(self, allAgentsIdsWholeGroup, concernedAgentsIds, policyForUncommittedAgent):
        self.allAgentsIdsWholeGroup = allAgentsIdsWholeGroup
        self.concernedAgentsIds = concernedAgentsIds
        self.policyForUncommittedAgent = policyForUncommittedAgent

    def __call__(self, intention, state, percivedAction):
        goalId, weIds = intention
        uncommittedAgentIds = [Id for Id in self.allAgentsIdsWholeGroup 
                if (Id not in list(weIds)) and (Id in self.concernedAgentsIds)]
        if len(uncommittedAgentIds) == 0:
            uncommittedAgentsPolicyLikelihood = 1
        else:
            uncommittedActionDistributions = [self.policyForUncommittedAgent(state, goalId, uncommittedAgentIds[index]) 
                    for index in range(len(uncommittedAgentIds))]
            uncommittedAgentsPolicyLikelihood = np.product([actionDistribution[tuple(percivedAction[Id])]
                    for actionDistribution, Id in zip(uncommittedActionDistributions, uncommittedAgentIds)])
        return uncommittedAgentsPolicyLikelihood

class CalCommittedAgentsPolicyLikelihood:
    def __init__(self, concernedAgentsIds, policyForCommittedAgent):
        self.concernedAgentsIds = concernedAgentsIds
        self.policyForCommittedAgent = policyForCommittedAgent

    def __call__(self, intention, state, percivedAction):
        goalId, weIds = intention
        committedAgentIds = [Id for Id in list(weIds) if Id in self.concernedAgentsIds]
        if len(committedAgentIds) == 0:
            committedAgentsPolicyLikelihood = 1
        else:
            jointActionDistribution = self.policyForCommittedAgent(state, goalId, weIds)
            jointDfIndex = pd.MultiIndex.from_tuples(list(jointActionDistribution.keys()), names = [str(Id) for Id in list(weIds)])
            jointDf = pd.DataFrame(list(jointActionDistribution.values()), jointDfIndex, columns = ['likelihood'])
            marginalDf = jointDf.groupby([str(Id) for Id in committedAgentIds]).sum()
            marginalLikelihood = marginalDf['likelihood'].to_dict()
            if len(committedAgentIds) == 1:
                jointAction = tuple(percivedAction[committedAgentIds[0]])
            else:
                jointAction = tuple([tuple(individualAction) 
                    for individualAction in np.array(percivedAction)[list(committedAgentIds)]])
            committedAgentsPolicyLikelihood = marginalLikelihood[jointAction]
        return committedAgentsPolicyLikelihood

class CalCommittedAgentsContinuousPolicyLikelihood:
    def __init__(self, concernedAgentsIds, policyForCommittedAgent, rationalityBeta):
        self.concernedAgentsIds = concernedAgentsIds
        self.policyForCommittedAgent = policyForCommittedAgent
        self.rationalityBeta = rationalityBeta

    def __call__(self, intention, state, percivedAction):
        goalId, weIds = intention
        committedAgentIds = [Id for Id in list(weIds) if Id in self.concernedAgentsIds]
        if len(committedAgentIds) == 0:
            committedAgentsPolicyLikelihood = 1
        else:
            jointActionDistribution = self.policyForCommittedAgent(state, goalId, weIds)
            jointAction = np.array(percivedAction)[list(committedAgentIds)]
            pdfs = [individualDistribution.pdf(action) for individualDistribution, action in zip(jointActionDistribution, jointAction)]
            committedAgentsPolicyLikelihood = np.power(np.product(pdfs), self.rationalityBeta)
        return committedAgentsPolicyLikelihood

class InferOneStep:
    def __init__(self, jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood, softPrior):
        self.jointHypothesisSpace = jointHypothesisSpace
        self.concernedHypothesisVariable = concernedHypothesisVariable
        self.calJointLikelihood = calJointLikelihood
        self.softPrior = softPrior

    def __call__(self, intentionPrior, state, perceivedAction):
        jointHypothesisDf = pd.DataFrame(index = self.jointHypothesisSpace)
        intentions = jointHypothesisDf.index.get_level_values('intention')
        jointHypothesisDf['likelihood'] = [self.calJointLikelihood(intention, state, perceivedAction) for intention in intentions]
        #jointHypothesisDf['jointLikelihood'] = jointHypothesisDf.apply(lambda row: self.calJointLikelihood(row.name[0], state, row.name[1], nextState))
        marginalLikelihood = jointHypothesisDf.groupby(self.concernedHypothesisVariable).sum()
        oneStepLikelihood = marginalLikelihood['likelihood'].to_dict()
        
        softenPrior = self.softPrior(intentionPrior)
        unnomalizedPosterior = {key: np.exp(np.log(softenPrior[key] + 1e-4) + np.log(oneStepLikelihood[key])) for key in list(intentionPrior.keys())}
        normalizedProbabilities = np.array(list(unnomalizedPosterior.values())) / np.sum(list(unnomalizedPosterior.values()))
        normalizedPosterior = dict(zip(list(unnomalizedPosterior.keys()),normalizedProbabilities))
        #print(normalizedPosterior)
        return normalizedPosterior


class InferOneStepWithActionNoise:
    def __init__(self, jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood, softPrior):
        self.jointHypothesisSpace = jointHypothesisSpace
        self.concernedHypothesisVariable = concernedHypothesisVariable
        self.calJointLikelihood = calJointLikelihood
        self.softPrior = softPrior

    def __call__(self, intentionPrior, state, perceivedAction):
        jointHypothesisDf = pd.DataFrame(index = self.jointHypothesisSpace)
        intentions = jointHypothesisDf.index.get_level_values('intention')
        actions = jointHypothesisDf.index.get_level_values('action')
        jointHypothesisDf['likelihood'] = [self.calJointLikelihood(intention, state, action, perceivedAction) for intention, action in zip(intentions, actions)]
        #jointHypothesisDf['jointLikelihood'] = jointHypothesisDf.apply(lambda row: self.calJointLikelihood(row.name[0], state, row.name[1], nextState))
        marginalLikelihood = jointHypothesisDf.groupby(self.concernedHypothesisVariable).sum()
        oneStepLikelihood = marginalLikelihood['likelihood'].to_dict()
        
        softenPrior = self.softPrior(intentionPrior)
        unnomalizedPosterior = {key: np.exp(np.log(softenPrior[key] + 1e-4) + np.log(oneStepLikelihood[key])) for key in list(intentionPrior.keys())}
        normalizedProbabilities = np.array(list(unnomalizedPosterior.values())) / np.sum(list(unnomalizedPosterior.values()))
        normalizedPosterior = dict(zip(list(unnomalizedPosterior.keys()),normalizedProbabilities))
        #print(normalizedPosterior)
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
