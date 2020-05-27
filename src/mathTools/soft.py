import numpy as np
import random


class SoftMax:
    def __init__(self, softParameter):
        self.softParameter = softParameter

    def __call__(self, originalMapping):
        hypotheses = list(originalMapping.keys())
        softenUnnormalizedValues = np.array([np.power(np.e, value * self.softParameter) for value in list(originalMapping.values())])
        softenNormalizedValues = list(softenUnnormalizedValues / np.sum(softenUnnormalizedValues))
        softenMapping = dict(zip(hypotheses, softenNormalizedValues))
        return softenMapping


