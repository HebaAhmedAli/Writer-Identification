import Preprocessing as pp
import numpy as np
from matplotlib import pyplot as plt
import math

inf=1.2
# The function input is array of writers images
def getFeatureVector(writerWithCoresspondingImages):
    # TO DO: Write our method for extracting the feature vector.
    featureVector=[]
    cnt,polycnt = pp.splitImageIntoSmallImagesAndGetContors(writerWithCoresspondingImages,
            writerWithCoresspondingImages.shape[1], writerWithCoresspondingImages.shape[0])
    slopes=[]
    for i in range(len(polycnt)):
        for j in range(1,len(polycnt[i])):
            slope = getSlope(polycnt[i][j-1],polycnt[i][j])
            slopes.append(slope)
    #print(slopes)
    plt.hist(slopes,500, [0,1.2])
    plt.show()
    return featureVector

def getFeatureVectors(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    featureVectors=[[] for i in range(len(trainingDataImages))] 
    for i in range(len(trainingDataImages)):
        featureVectors[i]=getFeatureVector(trainingDataImages[i])
    return [],featureVectors

# Get line slope using two points

def getSlope(pt1,pt2):
    slope = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
    length = math.sqrt(math.pow(pt1[1]-pt2[1],2)+math.pow(pt1[0]-pt2[0],2))
    if pt2[0] == pt1[0]:
        return inf, length
    return slope, length