import numpy as np


class GetStateFromTrajectory:
    def __init__(self, timeStep, stateIndex):
        self.timeStep = timeStep
        self.stateIndex = stateIndex

    def __call__(self, trajectory):
        state = trajectory[self.timeStep][self.stateIndex]

        return state


class GetAgentPosFromTrajectory:
    def __init__(self, getAgentPosFromState, getStateFromTrajectory):
        self.getAgentPosFromState = getAgentPosFromState
        self.getStateFromTrajectory = getStateFromTrajectory

    def __call__(self, trajectory):
        state = self.getStateFromTrajectory(trajectory)
        agentPos = self.getAgentPosFromState(state)

        return agentPos


class GetAgentActionFromTrajectory:
    def __init__(self, timeStep, actionIndex, agentId):
        self.timeStep = timeStep
        self.actionIndex = actionIndex
        self.agentId = agentId

    def __call__(self, trajectory):
        allAgentActions = trajectory[self.timeStep][self.actionIndex]
        agentAction = allAgentActions[self.agentId]

        return agentAction


class ComputeOptimalNextPos:
    def __init__(self, getInitStateFromTrajectory, getOptimalAction, transit, getAgentPosFromState):
        self.getInitStateFromTrajectory = getInitStateFromTrajectory
        self.getOptimalAction = getOptimalAction
        self.transit = transit
        self.getAgentPosFromState = getAgentPosFromState

    def __call__(self, trajectory):
        initState = self.getInitStateFromTrajectory(trajectory)
        optimalAction = self.getOptimalAction(initState)
        nextState = self.transit(initState, optimalAction)
        nextPos = self.getAgentPosFromState(nextState)

        return nextPos


class DistanceBetweenActualAndOptimalNextPosition:
    def __init__(self, computeOptimalNextPos, getPosAtNextStepFromTrajectory):
        self.computeOptimalNextPos = computeOptimalNextPos
        self.getPosAtNextStepFromTrajectory = getPosAtNextStepFromTrajectory

    def __call__(self, trajectory):
        optimalNextPos = self.computeOptimalNextPos(trajectory)
        posAtNextStep = self.getPosAtNextStepFromTrajectory(trajectory)
        L2distance = np.linalg.norm(posAtNextStep - optimalNextPos, ord = 2)

        return L2distance


def calculateCrossEntropy(data, episilon = 1e-12):
    prediction, target = list(data.values())
    ce = -1 * sum([target[index] * np.log(prediction[index]+episilon) for index in range(len(prediction))])
    return ce
