import numpy as np


def transiteCartesianToPolar(vector):
    return np.arctan2(vector[1], vector[0])


def transitePolarToCartesian(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(np.array(vector1), np.array(vector2).T)
    innerProduct = vectoriseInnerProduct
    norm1 = computeVectorNorm(vector1)
    norm2 = computeVectorNorm(vector2)
    if norm1 > 0 and norm2 > 0:
        unclipRatio = innerProduct / (norm1 * norm2)
        ratio = np.clip(unclipRatio, -1.0, 1.0)  # float precision probblem as enmin report
        angle = np.arccos(ratio)
    else:
        angle = np.nan
    return angle


def computeVectorNorm(vector):
    L2Norm = np.linalg.norm(vector, ord=2)
    return L2Norm
