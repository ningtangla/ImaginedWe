import unittest
import numpy as np
from ddt import ddt, data, unpack
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Local import
from src.MDPChasing.envNoPhysics import TransitForNoPhysics, TransitGivenOtherPolicy, IsTerminal, StayInBoundaryByReflectVelocity, CheckBoundary
from src.MDPChasing.state import GetAgentPosFromState
from src.MDPChasing.analyticGeometryFunctions import computeVectorNorm
from src.chooseFromDistribution import sampleFromDistribution

@ddt
class TestEnvNoPhysics(unittest.TestCase):
    def setUp(self):
        self.numOfAgent = 2
        self.sheepId = 0
        self.wolfId = 1
        self.posIndex = [0, 1]
        self.xBoundary = [0, 640]
        self.yBoundary = [0, 480]
        self.minDistance = 50
        self.getPreyPos = GetAgentPosFromState(
            self.sheepId, self.posIndex)
        self.getPredatorPos = GetAgentPosFromState(
            self.wolfId, self.posIndex)
        self.stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(
            self.xBoundary, self.yBoundary)
        self.isTerminal = IsTerminal(
            self.minDistance, self.getPredatorPos, self.getPreyPos, )
        self.transition = TransitForNoPhysics(self.stayInBoundaryByReflectVelocity)

    @data((np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])),
          (np.array([[1, 2], [3, 4]]), np.array([[1, 0], [0, 1]]), np.array([[2, 2], [3, 5]])))
    @unpack
    def testTransition(self, state, action, groundTruthReturnedNextState):
        nextState = self.transition(state, action)
        truthValue = nextState == groundTruthReturnedNextState
        self.assertTrue(truthValue.all())
     
    @data((np.array([[0, 0], [0, 0]]), np.array([0, 0]), 0, {(0, 0):1},  np.array([[0, 0], [0, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 0, {(1, 0):1},  np.array([[0, 0], [1, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 1, {(0, 0):1},  np.array([[0, 0], [0, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 1, {(1, 0):1},  np.array([[1, 0], [0, 0]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 0, {(1, 0):1},  np.array([[1, 2], [4, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 0, {(0, 1):1},  np.array([[1, 2], [3, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 1, {(1, 0):1},  np.array([[2, 2], [3, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 1, {(0, 1):1},  np.array([[1, 3], [3, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([2, 3]), 0, {(1, 0):1},  np.array([[3, 5], [4, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([2, 3]), 0, {(0, 1):1},  np.array([[3, 5], [3, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([2, 3]), 1, {(1, 0):1},  np.array([[2, 2], [5, 7]])),
          (np.array([[1, 2], [3, 4]]), np.array([2, 3]), 1, {(0, 1):1},  np.array([[1, 3], [5, 7]])))
    @unpack
    def testTransitionGivenOthersDeterminsticPolicy(self, state, individualAction, selfId, othersActionDist, groundTruthReturnedNextState):
        policy = lambda state: othersActionDist
        otherPolicy = lambda state: [policy(state), policy(state)]
        chooseAction = [sampleFromDistribution, sampleFromDistribution]
        transitGivenOtherPolicy = TransitGivenOtherPolicy(selfId, self.transition, otherPolicy, chooseAction)
        nextState = transitGivenOtherPolicy(state, individualAction)
        truthValue = nextState == groundTruthReturnedNextState
        self.assertTrue(truthValue.all())
    
    @data((np.array([[0, 0], [0, 0]]), np.array([(0, 0),]), 0, {((0, 0),):1},  np.array([[0, 0], [0, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([(0, 0),]), 0, {((1, 0),):1},  np.array([[0, 0], [1, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([(0, 0),]), 1, {((0, 0),):1},  np.array([[0, 0], [0, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([(0, 0),]), 1, {((1, 0),):1},  np.array([[1, 0], [0, 0]])),
          (np.array([[1, 2], [3, 4]]), np.array([(0, 0),]), 0, {((1, 0),):1},  np.array([[1, 2], [4, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([(0, 0),]), 0, {((0, 1),):1},  np.array([[1, 2], [3, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([(0, 0),]), 1, {((1, 0),):1},  np.array([[2, 2], [3, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([(0, 0),]), 1, {((0, 1),):1},  np.array([[1, 3], [3, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([(2, 3),]), 0, {((1, 0),):1},  np.array([[3, 5], [4, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([(2, 3),]), 0, {((0, 1),):1},  np.array([[3, 5], [3, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([(2, 3),]), 1, {((1, 0),):1},  np.array([[2, 2], [5, 7]])),
          (np.array([[1, 2], [3, 4]]), np.array([(2, 3),]), 1, {((0, 1),):1},  np.array([[1, 3], [5, 7]])))
    @unpack
    def testTransitionGivenOthersDeterminsticPolicyCentralControl(self, state, individualAction, selfId, othersActionDist, groundTruthReturnedNextState):
        policy = lambda state: othersActionDist
        otherPolicy = lambda state: [policy(state), policy(state)]
        chooseAction = [sampleFromDistribution, sampleFromDistribution]
        centralControlFlag = True
        transitGivenOtherPolicy = TransitGivenOtherPolicy(selfId, self.transition, otherPolicy, chooseAction, centralControlFlag)
        nextState = transitGivenOtherPolicy(state, individualAction)
        truthValue = nextState == groundTruthReturnedNextState
        self.assertTrue(truthValue.all())
    
    @data((np.array([[0, 0], [0, 0]]), np.array([0, 0]), 0, {(0, 0):1},  np.array([[0, 0], [0, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 0, {(1, 0):1},  np.array([[0, 0], [1, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 1, {(0, 0):1},  np.array([[0, 0], [0, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 1, {(1, 0):1},  np.array([[1, 0], [0, 0]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 0, {(1, 0):1},  np.array([[1, 2], [4, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 0, {(0, 1):1},  np.array([[1, 2], [3, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 1, {(1, 0):1},  np.array([[2, 2], [3, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 1, {(0, 1):1},  np.array([[1, 3], [3, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([2, 3]), 0, {(1, 0):1},  np.array([[3, 5], [4, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([2, 3]), 0, {(0, 1):1},  np.array([[3, 5], [3, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([2, 3]), 1, {(1, 0):1},  np.array([[2, 2], [5, 7]])),
          (np.array([[1, 2], [3, 4]]), np.array([2, 3]), 1, {(0, 1):1},  np.array([[1, 3], [5, 7]])))
    @unpack
    def testTransitionGivenOthersDeterminsticPolicyCentralControl(self, state, individualAction, selfId, othersActionDist, groundTruthReturnedNextState):
        policy = lambda state: othersActionDist
        otherPolicy = lambda state: [policy(state), policy(state)]
        chooseAction = [sampleFromDistribution, sampleFromDistribution]
        transitGivenOtherPolicy = TransitGivenOtherPolicy(selfId, self.transition, otherPolicy, chooseAction)
        nextState = transitGivenOtherPolicy(state, individualAction)
        truthValue = nextState == groundTruthReturnedNextState
        self.assertTrue(truthValue.all())
    
    @data((np.array([[0, 0], [0, 0]]), np.array([0, 0]), 0, {(0, 0):0.5, (1, 1): 0.5},  np.array([[0, 0], [0, 0]]), np.array([[0, 0], [1, 1]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 0, {(1, 0):0.5, (1, 1): 0.5},  np.array([[0, 0], [1, 0]]), np.array([[0, 0], [1, 1]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 1, {(0, 0):0.5, (1, 1): 0.5},  np.array([[0, 0], [0, 0]]), np.array([[1, 1], [0, 0]])),
          (np.array([[0, 0], [0, 0]]), np.array([0, 0]), 1, {(1, 0):0.5, (1, 1): 0.5},  np.array([[1, 0], [0, 0]]), np.array([[1, 1], [0, 0]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 0, {(1, 0):0.5, (1, 1): 0.5},  np.array([[1, 2], [4, 4]]), np.array([[1, 2], [4, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 0, {(0, 1):0.5, (1, 1): 0.5},  np.array([[1, 2], [3, 5]]), np.array([[1, 2], [4, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 1, {(1, 0):0.5, (1, 1): 0.5},  np.array([[2, 2], [3, 4]]), np.array([[2, 3], [3, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([0, 0]), 1, {(0, 1):0.5, (1, 1): 0.5},  np.array([[1, 3], [3, 4]]), np.array([[2, 3], [3, 4]])),
          (np.array([[1, 2], [3, 4]]), np.array([4, 3]), 0, {(1, 0):0.5, (1, 1): 0.5},  np.array([[5, 5], [4, 4]]), np.array([[5, 5], [4, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([4, 3]), 0, {(0, 1):0.5, (1, 1): 0.5},  np.array([[5, 5], [3, 5]]), np.array([[5, 5], [4, 5]])),
          (np.array([[1, 2], [3, 4]]), np.array([4, 3]), 1, {(1, 0):0.5, (1, 1): 0.5},  np.array([[2, 2], [7, 7]]), np.array([[2, 3], [7, 7]])),
          (np.array([[1, 2], [3, 4]]), np.array([4, 3]), 1, {(0, 1):0.5, (1, 1): 0.5},  np.array([[1, 3], [7, 7]]), np.array([[2, 3], [7, 7]])))
    @unpack
    def testTransitionGivenOthersStochasticPolicy(self, state, individualAction, selfId, othersActionDist, groundTruthReturnedNextState1, groundTruthReturnedNextState2):
        policy = lambda state: othersActionDist
        otherPolicy = lambda state: [policy(state), policy(state)]
        chooseAction = [sampleFromDistribution, sampleFromDistribution]
        transitGivenOtherPolicy = TransitGivenOtherPolicy(selfId, self.transition, otherPolicy, chooseAction)
        nextStates = [transitGivenOtherPolicy(state, individualAction) for _ in range(1000)]
        truthValue1 = np.count_nonzero([np.all(nextState == groundTruthReturnedNextState1) for nextState in nextStates])/len(nextStates)
        truthValue2 = np.count_nonzero([np.all(nextState == groundTruthReturnedNextState2) for nextState in nextStates])/len(nextStates)
        self.assertAlmostEqual(truthValue1, 0.5, places=1)
        self.assertAlmostEqual(truthValue2, 0.5, places=1)
        
    
    @unittest.skip
    @data(([[2, 2], [10, 10]], True), ([[10, 23], [100, 100]], False))
    @unpack
    def testIsTerminal(self, state, groundTruthTerminal):
        terminal = self.isTerminal(state)
        self.assertEqual(terminal, groundTruthTerminal)

    @data(([0, 1], [2, 3], [[2, 2], [100,100], [10, 10], [90, 90]], True),
          ([0, 2], [1, 3], [[2, 2], [100,100], [10, 10], [90, 90]], False),
          ([0, 1], [2, 3], [[2, 2], [100,100], [-50, -50], [50, 50]], False),
          ([0, 1], [2, 3], [[2, 2], [100,100], [-5, -5], [50, 50]], True),
          ([0, 1], [2, 3], [[2, 2], [100,100], [50, 50], [-5, -5]], True))
    @unpack
    def testIsTerminalOfMultiPredatorAndPrey(self, predatorIds, preyIds, state, groundTruthTerminal):
        getPredatorPos = GetAgentPosFromState(predatorIds, self.posIndex) 
        getPreyPos = GetAgentPosFromState(preyIds, self.posIndex)
        isTerminal = IsTerminal(self.minDistance, getPredatorPos, getPreyPos)
        terminal = isTerminal(state)
        self.assertEqual(terminal, groundTruthTerminal)
    
    @data(([0, 0], [0, 0], [0, 0]), ([1, -2], [1, -3], [1, 2]), ([1, 3], [2, 2], [1, 3]))
    @unpack
    def testCheckBoundaryAndAdjust(self, state, action, groundTruthNextState):
        checkState, checkAction = self.stayInBoundaryByReflectVelocity(state, action)
        truthValue = checkState == groundTruthNextState
        self.assertTrue(truthValue.all())

    @data(([1, 1], True), ([1, -2], False), ([650, 120], False))
    @unpack
    def testCheckBoundary(self, position, groundTruth):
        self.checkBoundary = CheckBoundary(self.xBoundary, self.yBoundary)
        returnedValue = self.checkBoundary(position)
        truthValue = returnedValue == groundTruth
        self.assertTrue(truthValue)


if __name__ == '__main__':
    unittest.main()
