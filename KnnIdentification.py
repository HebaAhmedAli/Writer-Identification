import FeatureExtraction as featureExtraction
from matplotlib import pyplot as plt
import numpy as np
import constants

# TODO: Delete after testing.
def show(image):
    # Figure size in inches
    plt.figure(figsize=(10, 10))
    # Display an image on the axes, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest',cmap="gray")
        

# Get the normalized eculidean dist.
def getNormalizedDist(featureVector,featureVectorToCompare):
    totalDist=0
    for i in range(len(featureVector)):
        a = np.array(featureVector[i])
        b = np.array(featureVectorToCompare[i])
        dist = np.linalg.norm(a-b)
        totalDist+=abs(dist)
        
    return totalDist/len(featureVector)

def identifyWriter(methods,featureVectors,tariningDataWriters,writerImage,writerImageGray,classifiedCO3=[],classifiedSift=[]):
    featureVector=featureExtraction.extractAndConcatinateFeauturesDuringIdentification(methods,writerImage,writerImageGray,classifiedCO3,classifiedSift)
    distsArr=[]
    for i in range(len(featureVectors)):
        if len(featureVectors[i])==0:
            continue
        newDist=getNormalizedDist(featureVector,featureVectors[i])
        distsArr.append((newDist,tariningDataWriters[i]))
    distsArr=sorted(distsArr,key=lambda x: x[0])
    nearestWriter=vote(distsArr)
    return nearestWriter

def vote(distsArr):
    votingArr=[0 for i in range(constants.writersIdSize)]
    votingArrDists=[0 for i in range(constants.writersIdSize)]
    for i in range(min(constants.k,len(distsArr))):
        votingArr[distsArr[i][1]]+=1
        votingArrDists[distsArr[i][1]]+=distsArr[i][0]
    # In case equal votes.
    nearestIndex=-1
    minDist=10000000
    maxVotingNo=max(votingArr)
    indices = [i for i, x in enumerate(votingArr) if x == maxVotingNo]
    for i in range(len(indices)):
        if votingArrDists[indices[i]] < minDist:
            minDist=votingArrDists[indices[i]]
            nearestIndex=indices[i]
    return nearestIndex