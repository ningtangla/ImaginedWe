import sys
sys.path.append('..')
import unittest
from ddt import ddt, data, unpack
from src.simple1DEnv import TransitionFunction, RewardFunction, Terminal


@ddt
class TestTransition(unittest.TestCase):
    def setUp(self):
        self.bound_low = 0
        self.bound_high = 7
        self.transition = TransitionFunction(self.bound_low, self.bound_high)

    @data((0, 1, 1), (3, -1, 2), (0, -1, 0), (7, 1, 7))
    @unpack
    def testTransitionDDT(self, state, action, next_state):
        # transition = TransitionFunction(self.bound_low, self.bound_high)
        cal_next_state = self.transition(state, action)
        self.assertEqual(cal_next_state, next_state)


@ddt
class TestTerminal(unittest.TestCase):
    def setUp(self):
        self.target_state = 7
        self.isTerminal = Terminal(self.target_state)

    @data((7, True), (3, False))
    @unpack
    def testTerminalDDT(self, state, is_terminal):
        cal_is_terminal = self.isTerminal(state)
        self.assertEqual(cal_is_terminal, is_terminal)


@ddt
class TestReward(unittest.TestCase):
    def setUp(self):
        self.target_state = 7
        self.isTerminal = Terminal(self.target_state)
        self.step_penalty = -1
        self.catch_reward = 1

        self.reward = RewardFunction(
            self.step_penalty, self.catch_reward, self.isTerminal)

    @data((3, 1, -1), (7, 1, 1))
    @unpack
    def testRewardDDT(self, state, action, reward):
        cal_reward = self.reward(state, action)
        self.assertEqual(cal_reward, reward)


if __name__ == '__main__':
    unittest.main()
