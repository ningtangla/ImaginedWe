import numpy as np


class SaveToBuffer:
    def __init__(self, windowSize):
        self.windowSize = windowSize

    def __call__(self, buffer, trajectories):
        bufferWithNewTraj = buffer[:] + trajectories
        if len(bufferWithNewTraj) >= self.windowSize:
            updatedBuffer = bufferWithNewTraj[-self.windowSize:]
        else:
            updatedBuffer = bufferWithNewTraj
        print("BUFFER SIZE: ", len(updatedBuffer))
        return updatedBuffer


class SampleBatchFromBuffer:
    def __init__(self, batchSize, getSamplingProbabilities):
        self.batchSize = batchSize
        self.getSamplingProbabilities = getSamplingProbabilities

    def __call__(self, buffer):
        samplingProbabilities = self.getSamplingProbabilities(buffer)
        sampledTrajectoriesIndex = np.random.choice(len(buffer), self.batchSize, samplingProbabilities)
        sampledTrajectories = [buffer[index] for index in sampledTrajectoriesIndex]
        sampleTimeStep = lambda trajectory: trajectory[np.random.randint(len(trajectory))]
        sampledTimeSteps = [sampleTimeStep(trajectory) for trajectory in sampledTrajectories]

        return sampledTimeSteps
