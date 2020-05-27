import numpy as np
import pandas as pd
import copy

class RecordValuesForObjects:
    def __init__(self, attributes, objects):
        self.attributes = attributes
        self.objects = objects

    def __call__(self, values):
        [[setattr(objectCase, attribute, value) for attribute, value in zip(self.attributes, copy.deepcopy(values))]
         for objectCase in self.objects]
        return None


class ResetObjects:
    def __init__(self, attributeValues, objects):
        self.attributeValues = attributeValues
        self.objects = objects

    def __call__(self):
        [[setattr(objectCase, attribute, value) for attribute, value in zip(list(attributeValue.keys()), copy.deepcopy(list(attributeValue.values())))]
         for objectCase, attributeValue in zip(self.objects, self.attributeValues)]



class GetObjectsValuesOfAttributes:
    def __init__(self, returnAttributes, objects):
        self.returnAttributes = returnAttributes
        self.objects = objects

    def __call__(self):
        returnAttributeValues = list(zip(*[list(zip(*[getattr(objectCase, attribute).copy() 
            for objectCase in self.objects]))
            for attribute in self.returnAttributes]))

        return returnAttributeValues



