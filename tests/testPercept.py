import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import math
from ddt import ddt, data, unpack

from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction


@ddt
class TestPercept(unittest.TestCase):
    def setUp(self):
        self.numSamples = 10000
        self.spaceOf5Action = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]])
        self.spaceOf2Action = np.array([[1, 0], [-1, 0]])
        self.perceptSelfAction = lambda singleAgentAction: singleAgentAction
        self.perceptOtherAction = lambda singleAgentAction: singleAgentAction
        pass
    
    @data((10, np.array([0, 0])),
          (10, np.array([-4, 3])),
          (5, np.array([-4, 3])),
          (5, np.array([-4, 3, 2])),
          (5, np.array([2])))
    @unpack
    def testSampleNoisyAction(self, noise, action):
        sampleNoisyAction = SampleNoisyAction(noise)
        sampledActions = np.array([sampleNoisyAction(action) for _ in range(self.numSamples)])
        covOfSamples = np.cov(sampledActions.T)
        truthCov = np.array(np.diag([noise**2] * len(action)))
        self.assertTrue(np.allclose(truthCov, covOfSamples, atol = 5))
    
    @data((np.array([1, 0]), np.array([1, 0]), 1),
          (np.array([np.sqrt(3)/2, 1/2]), np.array([1, 0]), 1),
          (np.array([np.sqrt(2)/2, np.sqrt(2)/2]), np.array([1, 0]), 0.5),
          (np.array([np.sqrt(2)/2, np.sqrt(2)/2]), np.array([0, 1]), 0.5))
    @unpack
    def testMapActionToSpaceOf5Action(self, action, mappedAction, groundTruthPropotion):
        mappingActionToAnotherSpace = MappingActionToAnotherSpace(self.spaceOf5Action)
        sampledMappedActions = [mappingActionToAnotherSpace(action) for _ in (range(self.numSamples))]
        proportion = np.mean([int(np.all(np.array(sampledMappedAction) == np.array(mappedAction)))
            for sampledMappedAction in sampledMappedActions])
        self.assertAlmostEqual(proportion, groundTruthPropotion, places = 1)

    @data((np.array([1, 0]), np.array([1, 0]), 1),
          (np.array([np.sqrt(3)/2, 1/2]), np.array([1, 0]), 1),
          (np.array([np.sqrt(2)/2, np.sqrt(2)/2]), np.array([1, 0]), 1),
          (np.array([0, 0]), np.array([1, 0]), 0.5),
          (np.array([0, 0]), np.array([-1, 0]), 0.5),
          (np.array([0, 1]), np.array([1, 0]), 0.5),
          (np.array([0, 1]), np.array([-1, 0]), 0.5))
    @unpack
    def testMapActionToSpaceOf2Action(self, action, mappedAction, groundTruthPropotion):
        mappingActionToAnotherSpace = MappingActionToAnotherSpace(self.spaceOf2Action)
        sampledMappedActions = [mappingActionToAnotherSpace(action) for _ in (range(self.numSamples))]
        proportion = np.mean([int(np.all(np.array(sampledMappedAction) == np.array(mappedAction)))
            for sampledMappedAction in sampledMappedActions])
        self.assertAlmostEqual(proportion, groundTruthPropotion, places = 1)
    
    @data(([0, 1], np.array([[1, 0], [-1, 0]]), np.array([[1, 0], [-1, 0]])),
          ([1, 0], np.array([[1, 0], [-1, 0]]), np.array([[-1, 0], [1, 0]])),
          ([1, 2], np.array([[2, 3], [1, 0], [-1, 0]]), np.array([[1, 0], [-1, 0]])),
          ([2, 0], np.array([[2, 3], [1, 0], [-1, 0]]), np.array([[-1, 0], [2, 3]])),
          ([0, 2], np.array([[2, 3], [1, 0], [-1, 0]]), np.array([[2, 3], [-1, 0]])),
          ([2, 0], np.array([[2, 3, 4], [1, 0, 1], [-1, 0, 3]]), np.array([[-1, 0, 3], [2, 3, 4]])))
    @unpack
    def testPerceptImaginedWeAction(self, imaginedWeId, action, groundTruthPerceivedAction):
        perceptImaginedWeAction = PerceptImaginedWeAction(imaginedWeId, self.perceptSelfAction, self.perceptOtherAction)    
        perceivedAction = perceptImaginedWeAction(action)
        self.assertTrue(np.all(perceivedAction == groundTruthPerceivedAction))

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
