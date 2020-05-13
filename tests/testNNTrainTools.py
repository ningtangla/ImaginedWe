import sys
import os
sys.path.append('..')
sys.path.append('../src/neuralNetwork')
import unittest
from ddt import ddt, unpack, data
import numpy as np
from trainTools import LearningRateModifier

@ddt
class TestNNparameter(unittest.TestCase):
    def setUp(self):
        self.sheepID = 0
        self.preference = False
        self.trajStateIndex = 0
        self.trajActionDistIndex = 2
        self.trajValueIndex = 3

    @data((0.1, 0.9, 100, 200, 0.081))
    @unpack
    def testLearningRateModifier(self, initLearningRate, decayRate, decayStep, globalStep, groundTruth):
        lrModifier = LearningRateModifier(initLearningRate, decayRate, decayStep)
        self.assertAlmostEqual(lrModifier(globalStep), groundTruth, places=5)

if __name__ == "__main__":
    unittest.main()
