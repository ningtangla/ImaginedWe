import numpy as np
import bisect
import functools as ft

class GetAgentPosFromState:
    def __init__(self, agentIds, posIndex):
        self.agentIds = agentIds
        self.posIndex = posIndex

    def __call__(self, state):
        state = np.asarray(state)
        agentStates = state[self.agentIds]
        if np.array(agentStates).ndim > 1:
            agentPos = np.asarray([state[self.posIndex] for state in agentStates])
        else:
            agentPos = agentStates[self.posIndex]
        return agentPos

class GetStateForPolicyGivenIntention:
    def __init__(self, agentSelfId):
        self.agentSelfId = agentSelfId

    def __call__(self, state, intentionId):
        IdsRelativeToIntention = list(self.agentSelfId.copy())
        for Id in list(intentionId):
            bisect.insort(IdsRelativeToIntention, Id)
        sortedIds = np.array(IdsRelativeToIntention)
        stateRelativeToIntention = np.array(state)[sortedIds]
        return stateRelativeToIntention
