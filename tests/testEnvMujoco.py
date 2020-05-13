import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
import numpy as np
from ddt import ddt, data, unpack
from mujoco_py import load_model_from_path, MjSim

# Local import
from src.MDPChasing.envMujoco import ResetUniform, TransitionFunction, IsTerminal
from src.MDPChasing.state import GetAgentPosFromState


@ddt
class TestEnvMujoco(unittest.TestCase):
    def setUp(self):
        self.modelPath = os.path.join(DIRNAME, '..', 'env', 'xmls', 'twoAgents.xml')
        self.model = load_model_from_path(self.modelPath)
        self.simulation = MjSim(self.model)
        self.numJointEachAgent = 2
        self.numAgent = 2
        self.killzoneRadius = 0.5
        self.numSimulationFrames = 20
        self.sheepId = 0
        self.wolfId = 1
        self.xPosIndex = [2, 3]
        self.getSheepPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)
        self.isTerminal = IsTerminal(self.killzoneRadius, self.getSheepPos, self.getWolfPos)
        self.minQPos = (-9.7, -9.7)
        self.maxQPos = (9.7, 9.7)
        self.withinBounds = WithinBounds(self.minQPos, self.maxQPos)

    @data(([0, 0, 0, 0], [0, 0, 0, 0], np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])),
          ([1, 2, 3, 4], [0, 0, 0, 0], np.asarray([[1, 2, 1, 2, 0, 0], [3, 4, 3, 4, 0, 0]])),
          ([1, 2, 3, 4], [5, 6, 7, 8], np.asarray([[1, 2, 1, 2, 5, 6], [3, 4, 3, 4, 7, 8]])))
    @unpack
    def testResetUniform(self, qPosInit, qVelInit, groundTruthReturnedInitialState):
        resetUniform = ResetUniform(self.simulation, qPosInit, qVelInit, self.numAgent)
        returnedInitialState = resetUniform()
        truthValue = returnedInitialState == groundTruthReturnedInitialState
        self.assertTrue(truthValue.all())


    @data((np.asarray([[1, 2, 1, 2, 0, 0], [4, 5, 4, 5, 0, 0]]), [[1, 1], [1, 1]]),
          (np.asarray([[4, 7, 4, 7, 0, 0], [-4, -7, -4, -7, 0, 0]]), [[-1, -1], [-1, -1]]),
          (np.asarray([[-6, 8, -6, 8, 0, 0], [6, -8, 6, -8, 0, 0]]), [[-1, 1], [1, -1]]))
    @unpack
    def testQPositionChangesInDirectionOfActionAfterTransition(self, oldState, allAgentsActions):
        transitionFunction = TransitionFunction(self.simulation, self.isTerminal, self.numSimulationFrames)
        newState = transitionFunction(oldState, allAgentsActions)
        differenceBetweenStates = newState - oldState
        differenceBetweenQPositions = differenceBetweenStates[:, :2].flatten()
        hadamardProductQPosAndAction = np.multiply(differenceBetweenQPositions, np.asarray(allAgentsActions).flatten())
        truthValue = all(i > 0 for i in hadamardProductQPosAndAction)
        self.assertTrue(truthValue)


    @data((np.asarray([[1, 2, 1, 2, 0, 0], [4, 5, 4, 5, 0, 0]]), np.asarray([[1, 1], [1, 1]])),
          (np.asarray([[4, 7, 4, 7, 0, 0], [-4, -7, -4, -7, 0, 0]]), np.asarray([[-1, -1], [-1, -1]])),
          (np.asarray([[-6, 8, -6, 8, 0, 0], [6, -8, 6, -8, 0, 0]]), np.asarray([[-1, 1], [1, -1]])))
    @unpack
    def testXPosEqualsQPosAfterTransition(self, state, allAgentsActions):
        transitionFunction = TransitionFunction(self.simulation, self.isTerminal, self.numSimulationFrames)
        newState = transitionFunction(state, allAgentsActions)
        newXPos = newState[:, 2:4]
        newQPos = newState[:, :2]
        truthValue = newQPos == newXPos
        self.assertTrue(truthValue.all())


    @data((0.2, np.asarray([[0, 0, 0, 0, 0, 0], [0.21, 0, 0.21, 0, 0, 0]]), False),
          (1, np.asarray([[-0.5, -0.5, -0.5, -0.5, 0, 0], [0, 0, 0, 0, 0, 0]]), True),
          (0.5, np.asarray([[10, -10, 10, -10, 0, 0], [-10, 10, -10, 10, 0, 0]]), False),
          (0.5, np.asarray([[10, -10, 10, -10, 0, 0], [10.4, -10, 10.4, -10, 0, 0]]), True))
    @unpack
    def testIsTerminal(self, minXDis, state, groundTruthTerminal):
        isTerminal = IsTerminal(minXDis, self.getSheepPos, self.getWolfPos)
        terminal = isTerminal(state)
        self.assertEqual(terminal, groundTruthTerminal)


if __name__ == "__main__":
    unittest.main()
