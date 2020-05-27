from subprocess import Popen, PIPE
import json
import math
import numpy as np
import itertools as it


class GenerateTrajectoriesParallel:
    def __init__(self, codeFileName, numSample, numCmdList):
        self.codeFileName = codeFileName
        self.numSample = numSample
        self.numCmdList = numCmdList

    def __call__(self, parameters):
        startSampleIndexes = np.arange(0, self.numSample, math.ceil(self.numSample / self.numCmdList))
        endSampleIndexes = np.concatenate([startSampleIndexes[1:], [self.numSample]])
        startEndIndexesPair = zip(startSampleIndexes, endSampleIndexes)
        parametersString = dict([(key, str(value)) for key, value in parameters.items()])
        parametersStringJS = json.dumps(parametersString)
        cmdList = [['python3', self.codeFileName, parametersStringJS, str(startSampleIndex), str(endSampleIndex)] for startSampleIndex, endSampleIndex in startEndIndexesPair]
        print(cmdList)
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            # proc.wait()
            proc.communicate()
        return cmdList


class ExcuteCodeOnConditionsParallel:
    def __init__(self, codeFileName, numSample, numCmdList):
        self.codeFileName = codeFileName
        self.numSample = numSample
        self.numCmdList = numCmdList

    def __call__(self, conditions):
        assert self.numCmdList >= len(conditions), "condition number > cmd number, use more cores or less conditions"
        numCmdListPerCondition = math.floor(self.numCmdList / len(conditions))
        startSampleIndexes = np.arange(0, self.numSample, math.ceil(self.numSample / numCmdListPerCondition))
        endSampleIndexes = np.concatenate([startSampleIndexes[1:], [self.numSample]])
        startEndIndexesPair = zip(startSampleIndexes, endSampleIndexes)
        conditionStartEndIndexesPair = list(it.product(conditions, startEndIndexesPair))
        cmdList = [['python3', self.codeFileName, json.dumps(condition), str(startEndSampleIndex[0]), str(startEndSampleIndex[1])]
                   for condition, startEndSampleIndex in conditionStartEndIndexesPair]
        print(cmdList)
        processList = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmdList]
        for proc in processList:
            proc.communicate()
            # proc.wait()
        return cmdList
