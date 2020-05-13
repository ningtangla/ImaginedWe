import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from ddt import ddt, data, unpack
import itertools as it

from src.MDPChasing.analyticGeometryFunctions import computeAngleBetweenVectors
from src.MDPChasing.policies import HeatSeekingDiscreteDeterministicPolicy, HeatSeekingContinuesDeterministicPolicy
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.inference.inference import CalPolicyLikelihood

@ddt
class TestPolicyLikelihood(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.sheepCentralControlActionSapce = list(it.product(self.actionSpace))
        self.preyId = 0
        self.predatorId = 1
        self.posIndex = [0, 1]
        self.getPredatorPos = GetAgentPosFromState(self.predatorId, self.posIndex)
        self.getPreyPos = GetAgentPosFromState(self.preyId, self.posIndex)
        self.policyGivenIntention = HeatSeekingDiscreteDeterministicPolicy(self.sheepCentralControlActionSapce, self.getPredatorPos, self.getPreyPos, computeAngleBetweenVectors)

    @data(([2], [0], np.array([[20, 0], [0, 20], [0, 0]]), ((10, 0),), 1),
          ([2], [0], np.array([[20, 0], [0, 20], [0, 0]]), ((7, 7),), 0),
          ([2], [1], np.array([[20, 0], [0, 20], [0, 0]]), ((0, 10),), 1),
          ([2], [1], np.array([[20, 0], [0, 20], [0, 0]]), ((0, -10),), 0),
          ([0], [1], np.array([[20, 0], [0, 20], [0, 0]]), ((7, -7),), 1),
          ([0], [1], np.array([[20, 0], [0, 20], [0, 0]]), ((-7, -7),), 0),
          ([0], [2], np.array([[20, 0], [0, 20], [0, 0]]), ((10, 0),), 1),
          ([0], [2], np.array([[20, 0], [0, 20], [0, 0]]), ((7, 7),), 0),
          ([2], [0], np.array([[20, 20], [0, 20], [0, 0]]), ((7, 7),), 1),
          ([2], [0], np.array([[20, 20], [0, 20], [0, 0]]), ((0, 10),), 0),
          ([2], [0], np.array([[20, 20], [0, 20], [20, 30]]), ((0, -10),), 1),
          ([2], [0], np.array([[20, 20], [0, 20], [20, 30]]), ((-7, 7),), 0))
    @unpack
    def testHeatSeekingDiscreteDeterministicPolicyLikelihood(self, imaginedWeId, intention, state, action, groundTruthLikelihood):
        getStateForPolicyGivenIntention = GetStateForPolicyGivenIntention(imaginedWeId)
        calPolicyLikelihood = CalPolicyLikelihood(getStateForPolicyGivenIntention, self.policyGivenIntention)
        policyLikelihood = calPolicyLikelihood(intention, state, action)
        self.assertEqual(policyLikelihood, groundTruthLikelihood)

if __name__ == "__main__":
    unittest.main()
