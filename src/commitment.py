import numpy as np
import random
import copy 

class RegulateCommitmentBroken:
    def __init__(self, partnerCommitPrior, updatePartnerCommitDistribution, chooseCommitmentWarning, commitmentInferInterval = 1):
        self.partnerCommitPrior = partnerCommitPrior
        self.updatePartnerCommitDistribution = updatePartnerCommitDistribution
        self.chooseCommitmentWarning = chooseCommitmentWarning
        self.commitmentInferInterval = commitmentInferInterval

    def __call__(self, state, perceivedAction, timeStep):
        if timeStep % self.commitmentInferInterval == 0 and timeStep != 0:
            partnerCommitPosterior = self.updatePartnerCommitDistribution(self.partnerCommitPrior, state, perceivedAction)
        else:
            partnerCommitPosterior = self.partnerCommitPrior.copy()
        commitmentWarn = abs(1 - self.chooseCommitmentWarning(partnerCommitPosterior))
        self.partnerCommitPrior = partnerCommitPosterior.copy()
        return commitmentWarn

class BreakCommitmentBasedOnTime:
    def __init__(self, breakCommitTime):
        self.breakCommitTime = breakCommitTime
    
    def __call__(self, committed, timeStep):
        if timeStep == self.breakCommitTime:
            committed = 0
        return committed

class ActiveReCommit:
    def __init__(self, reCommitProbability):
        self.reCommitProbability = reCommitProbability

    def __call__(self, committed, warned):
        if warned and (not committed):
            newCommitted = min(1, np.random.choice(2, p = [1-self.reCommitProbability, self.reCommitProbability]))
        else:
            newCommitted = committed
        return newCommitted


class PolicyOnChangableIntention:
    def __init__(self, perceptAction, intentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention, planningInterval=1,
                 intentionInferInterval=1, regulateCommitmentBroken=None, activeBreak=None, breakCommitmentPolicy=None, activateReCommit=None):
        self.timeStep = 0
        self.lastState = None
        self.lastAction = None
        self.perceptAction = perceptAction
        self.intentionPrior = intentionPrior
        self.updateIntentionDistribution = updateIntentionDistribution
        self.chooseIntention = chooseIntention
        self.getStateForPolicyGivenIntention = getStateForPolicyGivenIntention
        self.policyGivenIntention = policyGivenIntention
        self.formerIntentionPriors = [intentionPrior]
        self.planningInterval = planningInterval
        self.intentionInferInterval = intentionInferInterval
        self.commitmentWarn = 0
        self.formerCommitmentWarn = [0]
        self.warned = 0
        self.committed = 1
        self.formerCommitted = [1]
        self.regulateCommitmentBroken = regulateCommitmentBroken
        self.activeBreak = activeBreak
        self.breakCommitmentPolicy = breakCommitmentPolicy
        self.activateReCommit = activateReCommit

    def __call__(self, state):
        if self.timeStep != 0:
            perceivedAction = self.perceptAction(self.lastAction)

        if self.timeStep % self.intentionInferInterval == 0 and self.timeStep != 0 and self.committed:
            intentionPosterior = self.updateIntentionDistribution(self.intentionPrior, self.lastState, perceivedAction)
        else:
            intentionPosterior = self.intentionPrior.copy()
        intentionId = self.chooseIntention(intentionPosterior)
        stateRelativeToIntention = self.getStateForPolicyGivenIntention(state, intentionId)

        if self.timeStep % self.planningInterval == 0:
            actionDist = self.policyGivenIntention(stateRelativeToIntention)
        else:
            selfAction = tuple([self.lastAction[id] for id in self.getStateForPolicyGivenIntention.agentSelfId])
            actionDist = {tuple(selfAction): 1}

        self.lastState = state.copy()
        self.formerIntentionPriors.append(intentionPosterior.copy())
        self.intentionPrior = intentionPosterior.copy()

        if not isinstance(self.regulateCommitmentBroken, type(None)) and self.timeStep!= 0:
            self.commitmentWarn = self.regulateCommitmentBroken(self.lastState, perceivedAction, self.timeStep)

        if not isinstance(self.activeBreak, type(None)):
            self.committed = self.activeBreak(self.committed, self.timeStep)
            if not self.committed:
                actionDistOnIndividualActionSpace = self.breakCommitmentPolicy(state)
                actionDist = {(key,): value for key, value in actionDistOnIndividualActionSpace.items()}

        if not isinstance(self.activateReCommit, type(None)):
            self.committed = self.activateReCommit(self.committed, self.warned)
        
        self.formerCommitmentWarn.append(self.commitmentWarn)
        self.formerCommitted.append(self.committed)
        self.timeStep = self.timeStep + 1
        return actionDist


