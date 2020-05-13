class EstimateValueFromNode:
    def __init__(self, terminalReward, isTerminal, getStateFromNode, getApproximateValue):
        self.terminalReward = terminalReward
        self.isTerminal = isTerminal
        self.getStateFromNode = getStateFromNode
        self.getApproximateValue = getApproximateValue

    def __call__(self, node):
        state = self.getStateFromNode(node)
        if self.isTerminal(state):
            return self.terminalReward
        else:
            return self.getApproximateValue(state)
