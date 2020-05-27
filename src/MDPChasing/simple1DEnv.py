import numpy as np

class TransitionFunction:
    def __init__(self, bound_low, bound_high):
        self.bound_high = bound_high
        self.bound_low = bound_low
    def __call__(self, state, action):
        new_state = np.array(state) + np.array(action)
        if new_state < self.bound_low or new_state > self.bound_high:
            new_state = state
        
        return new_state
    
class Terminal:
    def __init__(self, target_state):
        self.target_state = target_state

    def __call__(self, state):
        return state == self.target_state

class RewardFunction:
    def __init__(self, step_penalty, catch_reward, isTerminal):
        self.step_penalty = step_penalty
        self.catch_reward = catch_reward
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        if self.isTerminal(state):
            return self.catch_reward
        else:
            return self.step_penalty


