import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
from ddt import ddt, data, unpack
import numpy as np

from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention

@ddt
class TestWrapperFunctions(unittest.TestCase):
    @data((0, [2, 3], np.asarray([[1, 2, 1, 2, 0, 0], [3, 4, 3, 4, 0, 0]]), np.asarray([1, 2])),
          (1, [2, 3], np.asarray([[1, 2, 1, 2, 0, 0], [3, 4, 3, 4, 0, 0]]), np.asarray([3, 4])))
    @unpack
    def testGetAgentPosFromState(self, agentId, posIndex, state, groundTruthAgentPos):
        getAgentPosFromState = GetAgentPosFromState(agentId, posIndex)
        agentPos = getAgentPosFromState(state)

        truthValue = np.array_equal(agentPos, groundTruthAgentPos)
        self.assertTrue(truthValue)

@ddt
class TestComposeState(unittest.TestCase):
    @data(([0], [1], np.asarray([[1, 2], [3, 4], [5, 6]]), np.asarray([[1, 2], [3, 4]])),
          ([0], [2], np.asarray([[1, 2], [3, 4], [5, 6]]), np.asarray([[1, 2], [5, 6]])),
          ([0, 1], [2], np.asarray([[1, 2], [3, 4], [5, 6]]), np.asarray([[1, 2], [3, 4], [5, 6]])),
          ([0, 2], [1], np.asarray([[1, 2], [3, 4], [5, 6]]), np.asarray([[1, 2], [3, 4], [5, 6]])),
          ([0, 1], [3], np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]), np.asarray([[1, 2], [3, 4], [7, 8]])))
    @unpack
    def testGetStateForPolicyGivenIntention(self, agentSelfId, intentionId, state, groundTruthStateRelativeToIntention):
        getStateForPolicyGivenIntention = GetStateForPolicyGivenIntention(agentSelfId)
        stateRelativeToIntention = getStateForPolicyGivenIntention(state, intentionId)

        self.assertTrue(np.all(stateRelativeToIntention == groundTruthStateRelativeToIntention))

if __name__ == "__main__":
    unittest.main()
