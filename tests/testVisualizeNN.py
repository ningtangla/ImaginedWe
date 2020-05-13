import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
import src.neuralNetwork.visualizeNN as visualizeNN


@ddt
class TestVisualizeNN(unittest.TestCase):
    @data(({'a': 2, 'b': 3, 'c': 1}, [('a', 1), ('a', 2), ('b', 1), ('b', 2), ('b', 3), ('c', 1)]),
          ({'shared': 1, 'action': 2, 'value': 0}, [('shared', 1), ('action', 1), ('action', 2)]),
          ({'value': 1, 'action': 2, 'shared': 1}, [('value', 1), ('action', 1), ('action', 2), ('shared', 1)]),
          ({'shared': 4, 'action': 3, 'value': 1}, [('shared', 1), ('shared', 2), ('shared', 3), ('shared', 4),
                                                    ('action', 1), ('action', 2), ('action', 3), ('value', 1)]))
    @unpack
    def testIndexLayers(self, sectionNameToDepth, groundTruthIndices):
        indices = visualizeNN.indexLayers(sectionNameToDepth)
        self.assertEqual(indices, groundTruthIndices)

    @data((['var1/sec1/fc1/', 'var1/sec1/fc2/', 'var2/sec1/fc1/', 'var2/sec1/fc2/'], 'var1', 'sec1', 1, 'var1/sec1/fc1/'),
          (['var1/sec1/fc1/', 'var1/sec1/fc2/', 'var2/sec1/fc1/', 'var2/sec1/fc2/'], 'var2', 'sec1', 1, 'var2/sec1/fc1/'),
          (['var1/sec1/fc1/', 'var1/sec1/fc2/', 'var2/sec1/fc1/', 'var2/sec1/fc2/'], 'var1', 'sec1', 2, 'var1/sec1/fc2/'),
          (['var1/sec1/fc1/', 'var1/sec1/fc2/', 'var1/sec2/fc1/', 'var1/sec2/fc2/'], 'var1', 'sec2', 1, 'var1/sec2/fc1/'),
          (['var1/sec1/fc1/kernel', 'var1/sec1/fc2/bias', 'var2/sec1/fc1/kernel', 'var2/sec1/fc2/add'], 'var1', 'sec1', 1, 'var1/sec1/fc1/kernel'))
    @unpack
    def testFindKey(self, allKeys, varName, sectionName, layerNum, groundTruthKey):
        findKey = visualizeNN.FindKey(allKeys)
        key = findKey(varName, sectionName, layerNum)
        self.assertEqual(key, groundTruthKey)

    @data(([1e-2, 1e-1, 1, 1e1, 1e2], 4, 0, [1, 1, 1, 2], [1e-2, 1e-1, 1, 1e1, 1e2]),
          ([0, 1e-2, 1e-1, 1, 1e1, 1e2 - 1e-3], 5, 1e-3, [1, 1, 1, 1, 2],
           [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
    @unpack
    def testLogHist(self, data, bins, base, groundTruthCounts, groundTruthBins):
        counts, bins = visualizeNN.logHist(data, bins, base)
        for count, groundTruthCount in zip(counts, groundTruthCounts):
            self.assertEqual(count, groundTruthCount)
        for num, groundTruthNum in zip(bins, groundTruthBins):
            self.assertAlmostEqual(num, groundTruthNum)


if __name__ == "__main__":
    unittest.main()
