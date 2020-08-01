import numpy as np
import random
import scipy.stats as ss

class SoftDistribution:
    def __init__(self, softParameter):
        self.softParameter = softParameter

    def __call__(self, distribution):
        hypotheses = list(distribution.keys())
        softenUnnormalizedProbabilities = np.array([np.power(probability, self.softParameter)for probability in list(distribution.values())])
        softenNormalizedProbabilities = list(softenUnnormalizedProbabilities / np.sum(softenUnnormalizedProbabilities))
        softenDistribution = dict(zip(hypotheses, softenNormalizedProbabilities))
        return softenDistribution

def maxFromDistribution(distribution):
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis 


def sampleFromDistribution(distribution):
    hypotheses = list(distribution.keys())
    probs = list(distribution.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedHypothesis = hypotheses[selectedIndex]
    return selectedHypothesis 

class BuildGaussianFixCov:
    def __init__(self, cov):
        self.cov = cov
    def __call__(self, mean):
        return ss.multivariate_normal(mean, self.cov)

def sampleFromContinuousSpace(distribution):
    return distribution.rvs() 
