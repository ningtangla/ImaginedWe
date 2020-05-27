import numpy as np

class RewardFunctionByTerminal():
    def __init__(self, timeReward, terminalReward, isTerminal):
        self.timeReward = timeReward
        self.terminalReward = terminalReward
        self.isTerminal = isTerminal
    def __call__(self, state, action, nextState):
        if self.isTerminal(nextState):
            reward = self.terminalReward
        else:
            reward = self.timeReward
        return reward
