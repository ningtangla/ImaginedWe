import sys
import os
sys.path.append('..')

import unittest
from ddt import ddt, unpack, data
import numpy as np

from src.MDPChasing.state import GetAgentPosFromState
from src.MDPChasing.envMujoco import IsTerminal
from src.MDPChasing.reward import HeuristicDistanceToTarget, RewardFunctionCompete, RewardFunctionWithWall


@ddt
class TestRewardFunctions(unittest.TestCase):
    def setUp(self):
        self.actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        self.sheepId = 0
        self.wolfId = 1
        self.xPosIndex = [2, 3]
        self.getSheepXPos = GetAgentPosFromState(self.sheepId, self.xPosIndex)
        self.getWolfXPos = GetAgentPosFromState(self.wolfId, self.xPosIndex)
        self.killzoneRadius = 0.5
        self.isTerminal = IsTerminal(self.killzoneRadius, self.getSheepXPos, self.getWolfXPos)

    @data((np.asarray([[-8, 0, -8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), -1.6), (np.asarray([[8, 0, 8, 0, 0, 0], [8, 0, 8, 0, 0, 0]]), 0), (np.asarray([[10, -10, 10, -10, 0, 0], [-10, 10, -10, 10, 0, 0]]), -2 * np.sqrt(2)))
    @unpack
    def testRolloutHeuristicBasedOnClosenessToTarget(self, state, groundTruthReward):
        weight = 0.1

        rolloutHeuristic = HeuristicDistanceToTarget(weight, self.getWolfXPos, self.getSheepXPos)
        reward = rolloutHeuristic(state)
        self.assertEqual(reward, groundTruthReward)


    @data((-0.05, 1, np.asarray([[0, 0, 0, 0, 0, 0, ], [1, 0, 1, 0, 0, 0]]), None, -0.05),
          (-0.05, 1, np.asarray([[0, 0, 0, 0, 0, 0, ], [0.3, 0, 0.3, 0, 0, 0]]), None, 1-0.05))
    @unpack
    def testRewardFunctionCompete(self, aliveBonus, deathPenalty, state, action, groundTruthReward):
        rewardFunction = RewardFunctionCompete(aliveBonus, deathPenalty, self.isTerminal)
        reward = rewardFunction(state, action)

        self.assertEqual(reward, groundTruthReward)

    @data((-0.05, 1, 1, 10, 1, np.asarray([[0, 0, 0, 0, 0, 0, ], [1, 0, 1, 0, 0, 0]]), None, -0.05),
          (-0.05, 1, 1, 10, 1, np.asarray([[0, 0, 0, 0, 0, 0, ], [0.3, 0, 0.3, 0, 0, 0]]), None, 1-0.05),
          (-0.05, 1, 9.2, 10, 1, np.asarray([[0, 0, 0, 0, 0, 0, ], [1, 0, 1, 0, 0, 0]]), None, -0.05-(0.2**2)/(9.2**2)*0.05),
          (-0.05, 1, 9.2, 10, 0.74, np.asarray([[0, 0, 0, 0, 0, 0, ], [1, 0, 1, 0, 0, 0]]), None, -0.05-(0.2**2)/(9.2**2)*0.74*0.05),
          (-0.05, 1, 9.2, 10, 1, np.asarray([[0, 0, 0, 0, 0, 0, ], [0.3, 0, 0.3, 0, 0, 0]]), None, 1-0.05))
    @unpack
    def testRewardFunctionWithWall(self, aliveBonus, deathPenalty, safeBound, wallDistanceToCenter, wallPunishRatio, state, action, groundTruthReward):
        rewardFunction = RewardFunctionWithWall(aliveBonus, deathPenalty, safeBound, wallDistanceToCenter, wallPunishRatio, self.isTerminal, self.getWolfXPos)
        reward = rewardFunction(state, action)

        self.assertAlmostEqual(reward, groundTruthReward)

if __name__ == "__main__":
    unittest.main()
