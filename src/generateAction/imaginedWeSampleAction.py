import numpy as np

class PolicyForUncommittedAgent:
    def __init__(self, allAgentsIdsWholeGroup, uncommittedPolicy, softDistribution, getAgentsStatesForPolicy):
        self.allAgentsIdsWholeGroup = allAgentsIdsWholeGroup
        self.uncommittedPolicy = uncommittedPolicy
        self.softDistribution = softDistribution
        self.getAgentsStatesForPolicy = getAgentsStatesForPolicy
    
    def __call__(self, state, goalId, selfId):
        relativeAgentsStatesForPolicy = self.getAgentsStatesForPolicy(state, goalId, self.allAgentsIdsWholeGroup, selfId)
        actionDistribution = self.uncommittedPolicy(relativeAgentsStatesForPolicy)
        softenActionDistribution = self.softDistribution(actionDistribution)
        return softenActionDistribution

class PolicyForCommittedAgent:
    def __init__(self, policyListBasedOnNumAgentsInWe, softDistribution, getAgentsStatesForPolicy):
        self.policyListBasedOnNumAgentsInWe = policyListBasedOnNumAgentsInWe
        self.softDistribution = softDistribution
        self.getAgentsStatesForPolicy = getAgentsStatesForPolicy

    def __call__(self, state, goalId, weIds):
        numAgentsInWe = len(weIds) 
        policyFunctionIndexInList = (numAgentsInWe - 2)
        policyFunction = self.policyListBasedOnNumAgentsInWe[policyFunctionIndexInList]

        relativeAgentsStatesForPolicy = self.getAgentsStatesForPolicy(state, goalId, weIds)
        actionDistribution = policyFunction(relativeAgentsStatesForPolicy)
        softenActionDistribution = self.softDistribution(actionDistribution)
        #print(actionDistribution[0].mean, actionDistribution[1].mean)
        return softenActionDistribution

class GetActionFromJointActionDistribution:
    def __init__(self, chooseActionMethod, getActionIndex):
        self.chooseActionMethod = chooseActionMethod
        self.getActionIndex = getActionIndex

    def __call__(self, jointActionDistribution, weIds, selfId):
        jointAction = self.chooseActionMethod(jointActionDistribution)
        action = tuple(np.array(jointAction)[self.getActionIndex(weIds, selfId)])
        return action

class HierarchyPolicyForCommittedAgent:
    def __init__(self, numAllGoals, selfId, fineActionSpace, getRoughJointActionDistribution, getJointAction, 
            getAgentsStatesAndRoughActionsForLevel2Policy, level2PolicyListBasedOnNumAgentsInWe, softDistribution):
        self.numAllGoals = numAllGoals
        self.selfId = selfId
        self.fineActionSpace = fineActionSpace
        self.numFineActions = len(self.fineActionSpace)
        self.getRoughJointActionDistribution = getRoughJointActionDistribution
        self.getJointAction = getJointAction
        self.getAgentsStatesAndRoughActionsForLevel2Policy = getAgentsStatesAndRoughActionsForLevel2Policy
        self.level2PolicyListBasedOnNumAgentsInWe = level2PolicyListBasedOnNumAgentsInWe
        self.softDistribution = softDistribution
    
    def __call__(self, state, goalId, weIds):
        roughJointActionDistribution = self.getRoughJointActionDistribution(state, goalId, weIds)
        roughJointAction = self.getJointAction(roughJointActionDistribution, weIds, self.selfId)
        agentsStatesForLevel2Policy = self.getAgentsStatesAndRoughActionsForLevel2Policy(state, goalId, weIds, self.selfId)
        roughActionsForLevel2Policy = self.getAgentsStatesAndRoughActionsForLevel2Policy(roughJointAction, [], 
                np.array(weIds) - self.numAllGoals, self.selfId - self.numAllGoals)
        composedStateForLevel2Policy = np.concatenate([agentsStatesForLevel2Policy, roughActionsForLevel2Policy])
        level2TunningActionDistribution = self.level2PolicyListBasedOnNumAgentsInWe[len(weIds) - 2](composedStateForLevel2Policy)
        level2TunningActionDistribution = {0: 1}
        selfRoughAction = roughJointAction[list(weIds).index(self.selfId)]
        fineActionIndexes = [int((self.fineActionSpace.index(tuple(selfRoughAction)) + tunningAction) % self.numFineActions) 
                for tunningAction in list(level2TunningActionDistribution.keys())] 
        tunnedActions = [self.fineActionSpace[index] for index in fineActionIndexes]
        fineActionDistribution = {key: probability 
                for key, probability in zip(tunnedActions, list(level2TunningActionDistribution.values()))}
        softenDistribution = self.softDistribution(fineActionDistribution)
        #print(composedStateForLevel2Policy, self.selfId)
        return softenDistribution

class SampleIndividualActionGivenIntention:
    def __init__(self, selfId, getActionDistributionForCommittedAgent, getActionDistributionForUncommittedAgent, 
            chooseCommittedAction, chooseUncommittedAction):
        self.selfId = selfId
        self.getActionDistributionForCommittedAgent = getActionDistributionForCommittedAgent
        self.getActionDistributionForUncommittedAgent  = getActionDistributionForUncommittedAgent
        self.chooseCommittedAction = chooseCommittedAction
        self.chooseUncommittedAction = chooseUncommittedAction

    def __call__(self, state, intention):
        goalId, weIds = intention
        if self.selfId not in list(weIds):
            actionDistribution = self.getActionDistributionForUncommittedAgent(state, goalId, self.selfId)
            individualAction = self.chooseUncommittedAction(actionDistribution)
        else:
            actionDistribution = self.getActionDistributionForCommittedAgent(state, goalId, weIds)
            individualAction = self.chooseCommittedAction(actionDistribution, weIds, self.selfId)
        #print(actionDistribution, individualAction, goalId, weIds, self.selfId)
        return individualAction

class SampleActionOnChangableIntention:
    def __init__(self, updateIntention, sampleIndividualActionGivenIntention):
        self.updateIntention= updateIntention
        self.sampleIndividualActionGivenIntention = sampleIndividualActionGivenIntention
    def __call__(self, state):
        intention = self.updateIntention(state)
        individualAction = self.sampleIndividualActionGivenIntention(state, intention)
        return individualAction 

class SampleActionOnFixedIntention:
    def __init__(self, selfId, fixedIntention, policy, chooseActionMethod, blocksId = []):
        self.selfId = selfId
        self.fixedIntention = fixedIntention
        self.goalIdsFromFixedIntention = list(np.array([fixedIntention]).flatten())
        self.blocksId = blocksId
        self.policy = policy
        self.chooseActionMethod = chooseActionMethod
    
    def __call__(self, state):
        relativeAgentsStates = state[np.sort([self.selfId] + self.goalIdsFromFixedIntention + self.blocksId)]
        actionDistribution = self.policy(relativeAgentsStates)
        individualAction = self.chooseActionMethod(actionDistribution)
        return individualAction

class SampleActionMultiagent:
    def __init__(self, individualSampleActions, recordActionForUpdateIntention): 
        self.individualSampleActions = individualSampleActions
        self.recordActionForUpdateIntention = recordActionForUpdateIntention

    def __call__(self, state):
        action = [individualSampleAction(state) for individualSampleAction in self.individualSampleActions]
        self.recordActionForUpdateIntention([action])
        return action
