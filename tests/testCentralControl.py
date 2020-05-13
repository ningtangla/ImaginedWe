import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
from ddt import ddt, data, unpack
import numpy as np

from src.centralControl import AssignCentralControlToIndividual

@ddt
class TestAssignCentralControlToIndividual(unittest.TestCase):
    @data(([0, 1], 0, ((1, 2), (3, 4)), np.asarray([1, 2])),
          ([0, 1], 1, ((1, 2), (3, 4)), np.asarray([3, 4])),
          ([2, 3], 3, ((1, 2), (3, 4)), np.asarray([3, 4])),
          ([2], 2, ((1, 2),), np.asarray([1, 2])),
          ([2, 3], 2, ((1, 2, 3), (4, 5, 6)), np.asarray([1, 2, 3])))
    @unpack
    def testAssignCentralControlToIndividual(self, imaginedWeId, individualId, centralControlAction, groundTruthIndividualAction):
        assignCentralControlToIndividual = AssignCentralControlToIndividual(imaginedWeId, individualId)
        individualAction = assignCentralControlToIndividual(centralControlAction)
        self.assertTrue(np.all(individualAction == groundTruthIndividualAction))

if __name__ == "__main__":
    unittest.main()
