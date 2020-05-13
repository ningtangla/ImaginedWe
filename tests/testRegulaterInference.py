import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math
from ddt import ddt, data, unpack

from src.inference.regulatorInference import GetPartnerActionLikelihoodOnJointPlanning, GetActionLikelihoodOnIndividualPlanning


@ddt
class TestPercept(unittest.TestCase):
    def setUp(self):
        pass
    
    @data(({(1, 1): 0.6, (1, 0): 0.4}, [(1, 1), (1, 0)], 0, 0.6),
          ({(1, 1): 0.6, (1, 0): 0.4}, [(1, 1), (1, 0)], 1, 0.4),
          ({(1, 1): 0.3, (1, 0): 0.9}, [(1, 1), (1, 0)], 0, 3/12),
          ({(1, 1): 0.3, (1, 0): 0.9}, [(1, 1), (1, 0)], 1, 9/12))
    @unpack
    def testGetIndividualActionLikelihoodOnIndividualPlaning(self, actionDist, action, selfId, groundTruthActionLikelihood):
        getActionLikelihoodOnIndividualPlanning = GetActionLikelihoodOnIndividualPlanning(selfId)
        likelihood = getActionLikelihoodOnIndividualPlanning(actionDist, action) 
        self.assertAlmostEqual(groundTruthActionLikelihood, likelihood)
    
    @data(({((1, 1), (1, 0)): 0.2, ((1, 1), (0, 1)): 0.4, ((0, 0), (1, 0)): 0.1, ((0, 0), (0, 1)): 0.3}, [(0, 0), (0, 1)], 0, 3/4),
          ({((1, 1), (1, 0)): 0.2, ((1, 1), (0, 1)): 0.4, ((0, 0), (1, 0)): 0.1, ((0, 0), (0, 1)): 0.3}, [(0, 0), (0, 1)], 1, 3/7),
          ({((1, 1), (1, 0)): 0.2, ((1, 1), (0, 1)): 0.4, ((0, 0), (1, 0)): 0.1, ((0, 0), (0, 1)): 0.3}, [(1, 1), (0, 1)], 0, 4/6),
          ({((1, 1), (1, 0)): 0.2, ((1, 1), (0, 1)): 0.4, ((0, 0), (1, 0)): 0.1, ((0, 0), (0, 1)): 0.3}, [(1, 1), (0, 1)], 1, 4/7),
          ({((1, 1), (1, 0)): 0.2, ((1, 1), (0, 1)): 0.4, ((0, 0), (1, 0)): 0.1, ((0, 0), (0, 1)): 0.3}, [(0, 0), (1, 0)], 0, 1/4),
          ({((1, 1), (1, 0)): 0.2, ((1, 1), (0, 1)): 0.4, ((0, 0), (1, 0)): 0.1, ((0, 0), (0, 1)): 0.3}, [(0, 0), (1, 0)], 1, 1/3),
          ({((1, 1), (1, 0)): 0.2, ((1, 1), (0, 1)): 0.4, ((0, 0), (1, 0)): 0.1, ((0, 0), (0, 1)): 0.3}, [(1, 1), (1, 0)], 0, 2/6),
          ({((1, 1), (1, 0)): 0.2, ((1, 1), (0, 1)): 0.4, ((0, 0), (1, 0)): 0.1, ((0, 0), (0, 1)): 0.3}, [(1, 1), (1, 0)], 1, 2/3),
          ({((1, 1), (1, 0)): 0.1, ((1, 1), (0, 1)): 0.5, ((0, 0), (1, 0)): 0.3, ((0, 0), (0, 1)): 0.2}, [(0, 0), (0, 1)], 0, 2/5),
          ({((1, 1), (1, 0)): 0.1, ((1, 1), (0, 1)): 0.5, ((0, 0), (1, 0)): 0.3, ((0, 0), (0, 1)): 0.2}, [(0, 0), (0, 1)], 1, 2/7),
          ({((1, 1), (1, 0)): 0.1, ((1, 1), (0, 1)): 0.5, ((0, 0), (1, 0)): 0.3, ((0, 0), (0, 1)): 0.2}, [(1, 1), (0, 1)], 0, 5/6),
          ({((1, 1), (1, 0)): 0.1, ((1, 1), (0, 1)): 0.5, ((0, 0), (1, 0)): 0.3, ((0, 0), (0, 1)): 0.2}, [(1, 1), (0, 1)], 1, 5/7),
          ({((1, 1), (1, 0)): 0.1, ((1, 1), (0, 1)): 0.5, ((0, 0), (1, 0)): 0.3, ((0, 0), (0, 1)): 0.2}, [(0, 0), (1, 0)], 0, 3/5),
          ({((1, 1), (1, 0)): 0.1, ((1, 1), (0, 1)): 0.5, ((0, 0), (1, 0)): 0.3, ((0, 0), (0, 1)): 0.2}, [(0, 0), (1, 0)], 1, 3/4),
          ({((1, 1), (1, 0)): 0.1, ((1, 1), (0, 1)): 0.5, ((0, 0), (1, 0)): 0.3, ((0, 0), (0, 1)): 0.2}, [(1, 1), (1, 0)], 0, 1/6),
          ({((1, 1), (1, 0)): 0.1, ((1, 1), (0, 1)): 0.5, ((0, 0), (1, 0)): 0.3, ((0, 0), (0, 1)): 0.2}, [(1, 1), (1, 0)], 1, 1/4))
    @unpack
    def testGetPartnerActionLikelihoodOnJointPlaning(self, actionDist, action, selfId, groundTruthActionLikelihood):
        numWe = 2
        getPartnerActionLikelihoodOnJointPlanning = GetPartnerActionLikelihoodOnJointPlanning(numWe, selfId)
        likelihood = getPartnerActionLikelihoodOnJointPlanning(actionDist, action) 
        self.assertAlmostEqual(groundTruthActionLikelihood, likelihood)
    

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
