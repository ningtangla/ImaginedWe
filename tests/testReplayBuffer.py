import os
import sys
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
from ddt import ddt, data, unpack
import numpy as np

from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer


@ddt
class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.sizeOfOneTimeStep = 3
        self.getUniformSamplingProbabilities = lambda buffer: [(1/len(buffer)) for _ in buffer]
        self.compareTuples = lambda tuple1, tuple2: all(np.array_equal(element1, element2) for element1, element2
                                                   in zip(tuple1, tuple2))
        self.compareTrajectories = lambda traj1, traj2: all(self.compareTuples(tuple1, tuple2) for tuple1, tuple2 in
                                                            zip(traj1, traj2))


    @data(([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 0)],
            [(np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 1)]],
           2, [[(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))], 12)],
               [(np.asarray([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]), [np.asarray((10, 0)), np.asarray((7, 7))], 43)]],
           2),
          ([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 0)],
            [(np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 1)]],
           3, [[(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))], 12)]],
           3),
          ([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 0)],
            [(np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 1)]],
           10, [[(np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]), [np.asarray((0, 10)), np.asarray((7, 7))], 12)],
                [(np.asarray([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]), [np.asarray((10, 0)), np.asarray((7, 7))], 43)]],
           4))
    @unpack
    def testSaveToBuffer(self, buffer, windowSize, trajectories, groundTruthNewBufferSize):
        originalBufferSize = len(buffer)
        saveToBuffer = SaveToBuffer(windowSize)
        updatedBuffer = saveToBuffer(buffer, trajectories)
        newBufferSize = len(updatedBuffer)

        self.assertEqual(groundTruthNewBufferSize, newBufferSize)
        self.assertEqual(originalBufferSize, len(buffer))

        compareFirstTrajectoryAfterUpdate = self.compareTrajectories(updatedBuffer[0], buffer[0])
        if len(buffer) < windowSize:
            self.assertTrue(compareFirstTrajectoryAfterUpdate)
        else:
            self.assertFalse(compareFirstTrajectoryAfterUpdate)

        isLastTrajectoryInBufferCorrect = self.compareTrajectories(trajectories[-1], updatedBuffer[-1])
        self.assertTrue(isLastTrajectoryInBufferCorrect)


    @data(([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 0)],
            [(np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 1)],
            [(np.asarray([[1, 2, 1, 2, 0, 0], [1, 2, 1, 2, 0, 0]]), [np.asarray((0, 10)), np.asarray((0, 10))], 3)],
            [(np.asarray([[7, 8, 7, 8, 0, 0], [2, 4, 2, 4, 0, 0]]), [np.asarray((7, 7)), np.asarray((0, 0))], 4)]], 2),
          ([[(np.asarray([[-4, 0, -4, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 0)],
            [(np.asarray([[-3, 0, -3, 0, 0, 0], [4, 0, 4, 0, 0, 0]]), [np.asarray((10, 0)), np.asarray((0, 0))], 1)]], 2))
    @unpack
    def testSampleBatch(self, buffer, batchSize):
        originalBufferSize = len(buffer)
        sampleBatch = SampleBatchFromBuffer(batchSize, self.getUniformSamplingProbabilities)
        sampledBatch = sampleBatch(buffer)

        self.assertEqual(len(sampledBatch), batchSize)
        self.assertEqual(len(buffer), originalBufferSize)
        self.assertEqual(len(sampledBatch[0]), self.sizeOfOneTimeStep)


if __name__ == '__main__':
    unittest.main()