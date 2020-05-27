import numpy as np
import pandas as pd

class ComputeStatistics:
    def __init__(self, getTrajectories, measurementFunction):
        self.getTrajectories = getTrajectories
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        measurementFunction = self.measurementFunction(oneConditionDf)
        allMeasurements = np.array([measurementFunction(trajectory) for trajectory in allTrajectories])
        #data = pd.DataFrame(allMeasurements)
        #data.to_csv(str(oneConditionDf))
        measurementMean = np.mean(allMeasurements, axis = 0)
        measurementSe = np.std(allMeasurements, axis = 0)/np.sqrt(len(allTrajectories) - 1)

        return pd.Series({'mean': measurementMean, 'se': measurementSe})


