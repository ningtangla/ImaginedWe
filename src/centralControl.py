import numpy as np

class AssignCentralControlToIndividual:
    def __init__(self, imaginedWeId, individualId):
        self.imaginedWeId = imaginedWeId
        self.individualId = individualId
        self.individualIndexInWe = list(self.imaginedWeId).index(self.individualId)

    def __call__(self, centralControlAction):
        individualAction = centralControlAction[self.individualIndexInWe]
        return individualAction

class sampleActionImaginedWeMultiagent:
    def __init__(self, individualPolices, softPolicyMethods, chooseCentralControlMethods, assignCentralControlMethods, recordActionForIndividualPolicies): 
        self.individualPolicies = individualPolicies
        self.softPolicyMethods = softPolicyMethods
        self.chooseCentralControlMethods = chooseCentralControlMethods
        self.assignCentralControlMethods = assignCentralControlMethods
        self.recordActionForIndividualPolicies = recordActionForIndividualPolicies

    def __call__(self, state):
        centralControlActionDists = [individualPolicy(state) for individualPolicy in self.individualPolicies]
        centralControlAction = [chooseCentralControl(centralControlActionDist) for centralControlActionDist in centralControlActionDists]
        action = [assignToIndividual(centralControlAction) for assignToIndividual in self.assignCentralControlMethods]
        self.recordActionForIndividualPolicies(action, self.individualPolicies)
        return action
