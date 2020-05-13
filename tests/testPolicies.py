import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from ddt import ddt, data, unpack

from src.MDPChasing.policies import stationaryAgentPolicy, RandomPolicy
from src.MDPChasing.policies import HeatSeekingDiscreteDeterministicPolicy, HeatSeekingContinuesDeterministicPolicy
from src.MDPChasing.policies import PolicyOnChangableIntention, SoftPolicy
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.analyticGeometryFunctions import computeAngleBetweenVectors
from src.chooseFromDistribution import maxFromDistribution, sampleFromDistribution
from src.centralControl import AssignCentralControlToIndividual

@ddt
class TestContinuesActionPolicies(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.xPosIndex = [2, 3]
        self.sheepId = 0
        self.wolfId = 1
        self.getSheepXPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)
        self.posIndex = [0, 1]
        self.getPredatorPos = GetAgentPosFromState(self.wolfId, self.posIndex)
        self.getPreyPos = GetAgentPosFromState(self.sheepId, self.posIndex)

    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), np.asarray((0, 0))),
          (np.asarray([[-8, 6, -8, 6, 0, 0], [4, -3, 4, -3, 0, 0]]), np.asarray((0, 0))),
          (np.asarray([[7, 6, 7, 6, 0, 0], [7, 4, 7, 4, 0, 0]]), np.asarray((0, 0))))
    @unpack
    def testStationaryAgentPolicy(self, state, groundTruthAction):
        action = maxFromDistribution(stationaryAgentPolicy(state))

        truthValue = np.array_equal(action, groundTruthAction)
        self.assertTrue(truthValue)

    @data((np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), 10, np.asarray((10, 0))),
          (np.asarray([[-8, 6, -8, 6, 0, 0], [-4, 3, -4, 3, 0, 0]]), 5, np.asarray((4, -3))),
          (np.asarray([[7, 6, 7, 6, 0, 0], [7, 4, 7, 4, 0, 0]]), 1, np.asarray((0, -1))))
    @unpack
    def testHeatSeekingContinuesDeterministicPolicy(self, state, actionMagnitude, groundTruthWolfAction):
        heatSeekingPolicy = HeatSeekingContinuesDeterministicPolicy(self.getSheepXPos, self.getWolfXPos,
                                                                    actionMagnitude)
        action = maxFromDistribution(heatSeekingPolicy(state))
        truthValue = np.allclose(action, groundTruthWolfAction)
        self.assertTrue(truthValue)

    def tearDown(self):
        pass


@ddt
class TestDiscreteActionPolicies(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.sheepId = 0
        self.wolfId = 1
        self.posIndex = [0, 1]
        self.getPredatorPos = GetAgentPosFromState(self.wolfId, self.posIndex)
        self.getPreyPos = GetAgentPosFromState(self.sheepId, self.posIndex)

    @data((np.array([[20, 20], [1, 1]]), np.array((7, 7))),
          (np.array([[20, 20], [80, 80]]), np.array((-7, -7))),
          (np.array([[20, 20], [20, 30]]), np.array((0, -10))))
    @unpack
    def testHeatSeekingDiscreteDeterministicPolicy(self, state, groundTruthAction):
        heatSeekingPolicy = HeatSeekingDiscreteDeterministicPolicy(self.actionSpace, self.getPredatorPos, self.getPreyPos, computeAngleBetweenVectors)
        action = maxFromDistribution(heatSeekingPolicy(state))
        truthValue = np.allclose(action, groundTruthAction)
        self.assertTrue(truthValue)



@ddt
class TestPolicyOnChangeableIntention(unittest.TestCase):
    def setUp(self):
        self.perceptAction = lambda action: action
        self.updateIntentionDistribution = lambda intentionPrior, state, perceivedAction: intentionPrior
        self.chooseIntention = sampleFromDistribution
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.preyId = [0]
        self.predatorId = [1]
        self.posIndex = [0, 1]
        self.getPredatorPos = GetAgentPosFromState(self.predatorId, self.posIndex)
        self.getPreyPos = GetAgentPosFromState(self.preyId, self.posIndex)
        self.policyGivenIntention = HeatSeekingDiscreteDeterministicPolicy(self.actionSpace, self.getPredatorPos, self.getPreyPos, computeAngleBetweenVectors)

    @data(([2], {(0,):1}, np.array([[20, 0], [0, 20], [0, 0]]), (10, 0)),
          ([2], {(1,):1}, np.array([[20, 0], [0, 20], [0, 0]]), (0, 10)),
          ([0], {(1,):1}, np.array([[20, 0], [0, 20], [0, 0]]), (7, -7)),
          ([0], {(2,):1}, np.array([[20, 0], [0, 20], [0, 0]]), (10, 0)),
          ([2], {(0,):1}, np.array([[20, 20], [0, 20], [0, 0]]), (7, 7)),
          ([2], {(0,):1}, np.array([[20, 20], [0, 20], [20, 30]]), (0, -10)))
    @unpack
    def testHeatSeekingDiscreteDeterministicPolicyOnChangableIntention(self, imaginedWeId, intentionPrior, state, groundTruthCentralControlAction):
        getStateForPolicyGivenIntention = GetStateForPolicyGivenIntention(imaginedWeId)
        policy = PolicyOnChangableIntention(self.perceptAction, intentionPrior, self.updateIntentionDistribution,
                self.chooseIntention, getStateForPolicyGivenIntention, self.policyGivenIntention)
        centralControlActionDist = policy(state)
        action = maxFromDistribution(centralControlActionDist)
        self.assertTrue(np.allclose(action, groundTruthCentralControlAction))

@ddt
class TestSoftPolicies(unittest.TestCase):
    @data((0, {0:1, 1:0}, {0:0.5, 1:0.5}),
          (1, {0:1, 1:0}, {0:1, 1:0}),
          (1000, {0:1, 1:0}, {0:1, 1:0}),
          (0, {0:0.5, 1:0.5}, {0:0.5, 1:0.5}),
          (1, {0:0.5, 1:0.5}, {0:0.5, 1:0.5}),
          (1000, {0:1, 1:0}, {0:1, 1:0}),
          (0, {0:0.6, 1:0.4}, {0:0.5, 1:0.5}),
          (1, {0:0.6, 1:0.4}, {0:0.6, 1:0.4}),
          (1000, {0:0.6, 1:0.4}, {0:1, 1:0}),
          (1000, {0:0.99, 1:0.01}, {0:1, 1:0}))
    @unpack
    def testSoftPolicy(self, softParameter, actionDistBeforeSoft, groundTruthGroundActionDist):
        softPolicy = SoftPolicy(softParameter)
        softenActionDist = softPolicy(actionDistBeforeSoft)
        self.assertDictEqual(softenActionDist, groundTruthGroundActionDist)

if __name__ == "__main__":
    unittest.main()
