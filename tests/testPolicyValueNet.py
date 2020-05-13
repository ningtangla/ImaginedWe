import sys
sys.path.append("..")
import unittest
from ddt import ddt, data, unpack
import src.neuralNetwork.policyValueNet as net
import numpy as np 

@ddt
class TestPolicyValueNet(unittest.TestCase):
    def setUp(self):
        self.numStateSpace = 4
        self.numActionSpace = 8
        self.summaryPath = None
        self.generateModel = net.GenerateModel(self.numStateSpace, self.numActionSpace)
    
    @data(([], [], [], [[4, 8], [4, 1]]),
          ([6], [], [], [[4, 6], [6, 8], [6, 1]]),
          ([], [10], [], [[4, 10], [10, 8], [4, 1]]),
          ([], [], [10], [[4, 8], [4, 10], [10, 1]]),
          ([6, 28, 32], [], [], [[4, 6], [6, 28], [28, 32], [32, 8], [32, 1]]),
          ([16, 32], [16], [8], [[4, 16], [16, 32], [32, 16], [16, 8], [32, 8], [8, 1]]),
          ([16, 32], [32, 16, 8], [8], [[4, 16], [16, 32], [32, 32], [32, 16], [16, 8], [8, 8], [32, 8], [8, 1]]),
          ([64, 128, 128], [64, 64, 64, 32, 32], [32, 8],
           [[4, 64], [64, 128], [128, 128], [128, 64], [64, 64], [64, 64], [64, 32], [32, 32], [32, 8], [128, 32], [32, 8], [8, 1]]))
    @unpack
    def testGeneratedParamShapes(self, sharedWidths, actionWidths, valueWidths, groundTruthShapes):
        model = self.generateModel(sharedWidths, actionWidths, valueWidths, summaryPath=self.summaryPath)
        g = model.graph
        weights_ = g.get_collection("weights")
        generatedWeightShapes = [w_.shape.as_list() for w_ in weights_]
        self.assertEqual(generatedWeightShapes, groundTruthShapes)

        biasShapes = [[shape[1]] for shape in groundTruthShapes]
        biases_ = g.get_collection("biases")
        generatedBiasShapes = [b_.shape.as_list() for b_ in biases_]
        self.assertEqual(generatedBiasShapes, biasShapes)

        activationShapes = [[None, shape[1]] for shape in groundTruthShapes]
        activations_ = g.get_collection("activations")
        generatedActivationShapes = [b_.shape.as_list() for b_ in activations_]
        self.assertEqual(generatedActivationShapes, activationShapes)

    @data((np.array([[4,8,7,3]]), np.array([[1,0,0,0,0,0,0,0]]), np.array([[0.9]])), 
          (np.array([[5,6,2,9]]), np.array([[0,0,1,0,0,0,0,0]]), np.array([[0.5]])))
    @unpack
    def testActionCrossEntropyLossAndValueLoss(self, stateBatch, actionBatch, valueBatch):
        model = self.generateModel([128], [128], [128], summaryPath=self.summaryPath)
        graph = model.graph
        state_ = graph.get_collection_ref("inputs")[0]
        groundTruthAction_, groundTruthValue_ = graph.get_collection_ref("groundTruths")
        loss_ = graph.get_collection_ref("loss")[0]
        actionLoss_ = graph.get_collection_ref("actionLoss")[0]
        valueLoss_ = graph.get_collection_ref("valueLoss")[0]
        fetches = {"actionLoss": actionLoss_, "valueLoss": valueLoss_}
        evalDict = model.run(fetches, feed_dict={state_: stateBatch, groundTruthAction_: actionBatch, groundTruthValue_: valueBatch})
       
        actionDist_ = graph.get_collection_ref("actionDistributions")[0]
        actionProbsBatch = model.run(actionDist_, feed_dict={state_: stateBatch}) 
        groundTruthCrossEntropy = np.mean([np.sum(np.multiply(-np.log(actionProb), actionOneHot)) for actionProb, actionOneHot in zip(actionProbsBatch, actionBatch)])
        NNCrossEntropy = evalDict['actionLoss']
        print(groundTruthCrossEntropy, NNCrossEntropy) 
        self.assertAlmostEqual(groundTruthCrossEntropy, NNCrossEntropy)

        valuePrediction_ = graph.get_collection_ref("values")[0]
        valuePredictionBatch = model.run(valuePrediction_, feed_dict={state_: stateBatch})
        groundTruthL2Norm = np.mean([np.power(valuePrediction - value, 2) for valuePrediction, value in zip(valuePredictionBatch, valueBatch)])
        NNL2Norm = evalDict['valueLoss']
        print(groundTruthL2Norm, NNL2Norm) 
        self.assertAlmostEqual(groundTruthL2Norm, NNL2Norm, places = 5)

if __name__ == "__main__":
    unittest.main()
