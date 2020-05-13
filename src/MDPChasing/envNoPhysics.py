import numpy as np
import itertools as it

class Reset():
    def __init__(self, xBoundary, yBoundary, numOfAgent):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.numOfAgnet = numOfAgent
    
    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initState = [[np.random.uniform(xMin, xMax),
                      np.random.uniform(yMin, yMax)]
                     for _ in range(self.numOfAgnet)]
        return np.array(initState)


class TransitForNoPhysics():
    def __init__(self, stayInBoundaryByReflectVelocity):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity

    def __call__(self, state, action):
        newState = np.array(state) + np.array(action)
        checkedNewPositionsAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewPositionsAndVelocities))
        return np.array(newState), np.array(newAction)

class TransitWithInterpolateState:
    def __init__(self, numFramesToInterpolate, StayInBoundaryAndOutObstacleByReflectVelocity, isTerminal):
        self.numFramesToInterpolate = numFramesToInterpolate
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        actionForInterpolation = np.array(action) / (self.numFramesToInterpolate + 1)
        for frameIndex in range(self.numFramesToInterpolate + 1):
            noBoundaryNextState = np.array(state) + np.array(actionForInterpolation)
            checkedNewPositionsAndVelocities = [self.stayInBoundaryByReflectVelocity(
                position, velocity) for position, velocity in zip(noBoundaryNextState, actionForInterpolation)]
            nextState, nextActionForInterpolation = list(zip(*checkedNewPositionsAndVelocities))
            if self.isTerminal(nextState):
                break
            state = nextState
            actionForInterpolation = nextActionForInterpolation
        return np.array(nextState)

class IsTerminal():
    def __init__(self, minDistance, getPreyPos, getPredatorPos):
        self.minDistance = minDistance
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos

    def __call__(self, state):
        terminal = False
        preyPositions = self.getPreyPos(state)
        predatorPositions = self.getPredatorPos(state)
        L2Normdistance = np.array([np.linalg.norm(np.array(preyPosition) - np.array(predatorPosition), ord=2) 
            for preyPosition, predatorPosition in it.product(preyPositions, predatorPositions)]).flatten()
        if np.any(L2Normdistance <= self.minDistance):
            terminal = True
        return terminal

class StayInBoundaryByReflectVelocity():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity

class StayInBoundaryAndOutObstacleByReflectVelocity():
    def __init__(self, xBoundary, yBoundary, xObstacles, yObstacles):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
        self.xObstacles = xObstacles
        self.yObstacles = yObstacles
    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
	
        for xObstacle, yObstacle in zip(self.xObstacles, self.yObstacles):
            xObstacleMin, xObstacleMax = xObstacle
            yObstacleMin, yObstacleMax = yObstacle
            if position[0] >= xObstacleMin and position[0] <= xObstacleMax and position[1] >= yObstacleMin and position[1] <= yObstacleMax:
                if position[0]-velocity[0]<=xObstacleMin:
                    adjustedVelX=-velocity[0]
                    adjustedX=2*xObstacleMin-position[0]
                if position[0]-velocity[0]>=xObstacleMax:
                    adjustedVelX=-velocity[0]
                    adjustedX=2*xObstacleMax-position[0]
                if position[1]-velocity[1]<=yObstacleMin:
                    adjustedVelY=-velocity[1]
                    adjustedY=2*yObstacleMin-position[1]
                if position[1]-velocity[1]>=yObstacleMax:
                    adjustedVelY=-velocity[1]
                    adjustedY=2*yObstacleMax-position[1]

        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity

class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        xPos, yPos = position
        if xPos >= self.xMax or xPos <= self.xMin:
            return False
        elif yPos >= self.yMax or yPos <= self.yMin:
            return False
        return True
