import sys
import os
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..'))

import unittest
from ddt import ddt, data, unpack
import numpy as np
import pandas as pd

from src.trajectoriesSaveLoad import ConvertTrajectoryToStateDf, GetAgentCoordinateFromTrajectoryAndStateDf, \
        conditionDfFromParametersDict, GetSavePath, LoadTrajectories, readParametersFromDf, loadFromPickle, GenerateAllSampleIndexSavePaths


@ddt
class TestTrajectoriesSaveLoad(unittest.TestCase):
    def setUp(self):
        self.dataDirectory = 'testData'
        self.stateIndex = 0
        self.getRangeNumAgentsFromTrajectory = lambda trajectory: list(range(np.shape(trajectory[0][self.stateIndex])[0]))
        self.getRangeTrajectoryLength = lambda trajectory: list(range(len(trajectory)))
        self.getAllLevelValuesRange = {'timeStep': self.getRangeTrajectoryLength, 'agentId': self.getRangeNumAgentsFromTrajectory}
        self.getAgentPosXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(self.stateIndex, 2)
        self.getAgentPosYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(self.stateIndex, 3)
        self.getAgentVelXCoord = GetAgentCoordinateFromTrajectoryAndStateDf(self.stateIndex, 4)
        self.getAgentVelYCoord = GetAgentCoordinateFromTrajectoryAndStateDf(self.stateIndex, 5)
        self.extractColumnValues = {'xPos': self.getAgentPosXCoord, 'yPos': self.getAgentPosYCoord,
                                    'xVel': self.getAgentVelXCoord, 'yVel': self.getAgentVelYCoord}
        self.convertTrajectoryToStateDf = ConvertTrajectoryToStateDf(self.getAllLevelValuesRange,
                                                                     conditionDfFromParametersDict,
                                                                     self.extractColumnValues)
    # @data(([(np.asarray([[1, 2, 1, 2, 3, 4], [5, 6, 5, 6, 7, 8]]), [np.asarray((0, 10)), np.asarray((7, 7))]),
    #         (np.asarray([[-1, -2, -1, -2, -3, -4], [-5, -6, -5, -6, -7, -8]]), [np.asarray((0, 10)), np.asarray((7, 7))])],
    #        pd.DataFrame([(1, 2, 3, 4), (5, 6, 7, 8), (-1, -2, -3, -4),(-5, -6, -7, -8)],
    #                     index = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=['timeStep', 'agentId']),
    #                     columns = ('xPos', 'yPos', 'xVel', 'yVel'))))
    # @unpack
    # def testConvertTrajectoryToStateDf(self, trajectory, groundTruthDf):
    #     df = self.convertTrajectoryToStateDf(trajectory)
    #     print(df)
    #     truthValue = groundTruthDf.equals(df)
    #     self.assertTrue(truthValue)


    # @data(('..', '.txt', {'qPosInit': (1, 2, 3, 4), 'numSimulations': 12}, {'numTrials': 23, 'trainSteps': 2},
    #        '../numSimulations=12_numTrials=23_qPosInit=(1,2,3,4)_trainSteps=2.txt'),
    #       ('', '.pickle', {'qPosInit': [1, 2, 3, 4], 'numSimulations': 12}, {'numTrials': 23, 'trainSteps': 2},
    #        'numSimulations=12_numTrials=23_qPosInit=[1,2,3,4]_trainSteps=2.pickle'))
    # @unpack
    # def testGetSavePathWithFixedParameters(self, dataDirectory, extension, fixedParameters, parameters, groundTruthPath):
    #     getSavePath = GetSavePath(dataDirectory, extension, fixedParameters)
    #     path = getSavePath(parameters)
    #     self.assertEqual(path, groundTruthPath)
    #
    #
    # @data(('..', '.txt', {'numTrials': 23, 'trainSteps': 2}, '../numTrials=23_trainSteps=2.txt'),
    #       ('', '.pickle', {'numTrials': 23, 'trainSteps': 2}, 'numTrials=23_trainSteps=2.pickle'))
    # @unpack
    # def testGetSavePathWithoutFixedParameters(self, dataDirectory, extension, parameters, groundTruthPath):
    #     getSavePath = GetSavePath(dataDirectory, extension)
    #     path = getSavePath(parameters)
    #     self.assertEqual(path, groundTruthPath)
    #
    #
    # @data(({'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}, {'maxRunningSteps': 15, 'numSimulations': 200,
    #                                                        'sheepPolicyName': 'MCTS'}))
    # @unpack
    # def testLoadTrajectoriesNumTrials(self, parameters, fixedParameters):
    #     getSavePath = GetSavePath('testData', '.pickle', fixedParameters)
    #
    #     loadTrajectories = LoadTrajectories(getSavePath, loadFromPickle)
    #     loadedTrajectories = loadTrajectories(parameters)
    #     numTrials = len(loadedTrajectories)
    #
    #     groundTruthNumTrials = parameters['numTrials']
    #     self.assertEqual(numTrials, groundTruthNumTrials)
    #
    #
    # @data(({'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}, {'maxRunningSteps': 15, 'numSimulations': 200,
    #                                                        'sheepPolicyName': 'MCTS'}))
    # @unpack
    # def testLoadTrajectoriesQPosInit(self, parameters, fixedParameters):
    #     getSavePath = GetSavePath('testData', '.pickle', fixedParameters)
    #
    #     loadTrajectories = LoadTrajectories(getSavePath, loadFromPickle)
    #     loadedTrajectories = loadTrajectories(parameters)
    #     initTimeStep = 0
    #     stateIndex = 0
    #     qPosIndex = 0
    #     numQPosEachAgent = 2
    #     allInitStates = [trajectory[initTimeStep][stateIndex] for trajectory in loadedTrajectories]
    #     allQPosInit = [initState[:, qPosIndex:qPosIndex+numQPosEachAgent].flatten() for initState in allInitStates]
    #
    #     groundTruthQPosInit = parameters['qPosInit']
    #
    #     allTruthValues = np.asarray([np.all(qPosInit == groundTruthQPosInit) for qPosInit in allQPosInit])
    #     self.assertTrue(np.all(allTruthValues))
    #
    #
    # @data((pd.DataFrame(index=pd.MultiIndex.from_tuples([(100, (-4, 0, 4, 0))], names=['numTrials', 'qPosInit'])),
    #        {'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}))
    # @unpack
    # def testReadParametersFromDf(self, df, groundTruthParameters):
    #     parameters = readParametersFromDf(df)
    #     self.assertEqual(parameters, groundTruthParameters)


    @data((GetSavePath('testData', '.pickle', {'maxRunningSteps': 15, 'numTrials': 100, 'qPosInit': (-4,0,4,0), 'sheepPolicyName': 'MCTS'}), 3, {'numSimulations': 200},
           {0: os.path.join('testData', 'maxRunningSteps=15_numSimulations=200_numTrials=100_qPosInit=(-4,0,4,0)_sampleIndex=100_sheepPolicyName=MCTS.pickle'),
            1: os.path.join('testData', 'maxRunningSteps=15_numSimulations=200_numTrials=100_qPosInit=(-4,0,4,0)_sampleIndex=101_sheepPolicyName=MCTS.pickle'),
            2: os.path.join('testData', 'maxRunningSteps=15_numSimulations=200_numTrials=100_qPosInit=(-4,0,4,0)_sampleIndex=102_sheepPolicyName=MCTS.pickle')}))
    @unpack
    def testGenerateAllSampleIndexSavePaths(self, getSavePath, numSamples, pathParameters, groundTruthAllPaths):
        generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getSavePath)
        allPaths = generateAllSampleIndexSavePaths(numSamples, pathParameters)
        print("ALL PATHS", allPaths)
        self.assertEqual(allPaths, groundTruthAllPaths)


    @data(('..', '.txt', {'qPosInit': (1, 2, 3, 4), 'numSimulations': 12}, {'numTrials': 23, 'trainSteps': 2},
           '../numSimulations=12_numTrials=23_qPosInit=(1,2,3,4)_trainSteps=2.txt'),
          ('', '.pickle', {'qPosInit': (1, 2, 3, 4), 'numSimulations': 12}, {'numTrials': 23, 'trainSteps': 2},
           'numSimulations=12_numTrials=23_qPosInit=(1,2,3,4)_trainSteps=2.pickle'))
    @unpack
    def testGetSavePathWithFixedParameters(self, dataDirectory, extension, fixedParameters, parameters, groundTruthPath):
        getSavePath = GetSavePath(dataDirectory, extension, fixedParameters)
        path = getSavePath(parameters)
        self.assertEqual(path, groundTruthPath)


    @data(('..', '.txt', {'numTrials': 23, 'trainSteps': 2}, '../numTrials=23_trainSteps=2.txt'),
          ('', '.pickle', {'numTrials': 23, 'trainSteps': 2}, 'numTrials=23_trainSteps=2.pickle'))
    @unpack
    def testGetSavePathWithoutFixedParameters(self, dataDirectory, extension, parameters, groundTruthPath):
        getSavePath = GetSavePath(dataDirectory, extension)
        path = getSavePath(parameters)
        self.assertEqual(path, groundTruthPath)


    @data((pd.DataFrame(index=pd.MultiIndex.from_tuples([(100, (-4, 0, 4, 0))], names=['numTrials', 'qPosInit'])),
           {'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}))
    @unpack
    def testReadParametersFromDf(self, df, groundTruthParameters):
        parameters = readParametersFromDf(df)
        self.assertEqual(parameters, groundTruthParameters)


    @data(({'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}, {'maxRunningSteps': 15, 'numSimulations': 200,
        'sheepPolicyName': 'MCTS'}, ['sampleIndex']))
    @unpack
    def testLoadMultipleTrajectoriesFromMultipleFiles(self, parameters, fixedParameters, fuzzySearchParameterNames):
        getSavePath = GetSavePath('testData', '.pickle', fixedParameters)
        loadTrajectories = LoadTrajectories(getSavePath, loadFromPickle, fuzzySearchParameterNames)
        loadedTrajectories = loadTrajectories(parameters)
        numTrials = len(loadedTrajectories)

        groundTruthNumTrials = parameters['numTrials']
        self.assertEqual(numTrials, groundTruthNumTrials)
    
    @data(({'numTrials': 50, 'qPosInit': (0, 0, 0, 0)}, {'maxRunningSteps': 2, 'numSimulations': 800}, ['sampleIndex']),
          ({'numTrials': 100, 'qPosInit': (0, 0, 0, 0)}, {'maxRunningSteps': 2, 'numSimulations': 800}, ['sampleIndex']))
    @unpack
    def testLoadMultipleTrajectoriesFromOneFile(self, parameters, fixedParameters, fuzzySearchParameterNames):
        getSavePath = GetSavePath('testData', '.pickle', fixedParameters)
        loadTrajectories = LoadTrajectories(getSavePath, loadFromPickle, fuzzySearchParameterNames)
        loadedTrajectories = loadTrajectories(parameters)
        numTrials = len(loadedTrajectories)

        groundTruthNumTrials = parameters['numTrials']
        self.assertEqual(numTrials, groundTruthNumTrials)
    
    @data(({'numTrials': 100, 'qPosInit': (-4, 0, 4, 0)}, {'maxRunningSteps': 15, 'numSimulations': 200,
                                                           'sheepPolicyName': 'MCTS'}, ['sampleIndex']))
    @unpack
    def testLoadTrajectoriesQPosInit(self, parameters, fixedParameters, fuzzySearchParameterNames):
        getSavePath = GetSavePath('testData', '.pickle', fixedParameters)
        loadTrajectories = LoadTrajectories(getSavePath, loadFromPickle, fuzzySearchParameterNames)
        loadedTrajectories = loadTrajectories(parameters)
        initTimeStep = 0
        stateIndex = 0
        qPosIndex = 0
        numQPosEachAgent = 2
        allInitStates = [trajectory[initTimeStep][stateIndex] for trajectory in loadedTrajectories]
        allQPosInit = [initState[:, qPosIndex:qPosIndex+numQPosEachAgent].flatten() for initState in allInitStates]

        groundTruthQPosInit = parameters['qPosInit']

        allTruthValues = np.asarray([np.all(qPosInit == groundTruthQPosInit) for qPosInit in allQPosInit])
        self.assertTrue(np.all(allTruthValues))



 #   @data((GetSavePath('..', '.pickle', {'numTrials': 25, 'maxRunningSteps': 3}), 3, {'numSimulations': 20},
 #          {0: os.path.join('..', 'maxRunningSteps=3_numSimulations=20_numTrials=25_sampleIndex=0.pickle'),
 #           1: os.path.join('..', 'maxRunningSteps=3_numSimulations=20_numTrials=25_sampleIndex=1.pickle'),
 #           2: os.path.join('..', 'maxRunningSteps=3_numSimulations=20_numTrials=25_sampleIndex=2.pickle')}))
 #   @unpack
 #   def testGenerateAllSampleIndexSavePaths(self, getSavePath, numSamples, pathParameters, groundTruthAllPaths):
 #       generateAllSampleIndexSavePaths = GenerateAllSampleIndexSavePaths(getSavePath)
 #       allPaths = generateAllSampleIndexSavePaths(numSamples, pathParameters)
 #       self.assertEqual(allPaths, groundTruthAllPaths)


if __name__ == '__main__':
    unittest.main()
