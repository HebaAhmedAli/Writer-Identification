import constants
def getFeatureVector(image):
    # TO DO: Write our method for extracting the feature vector for this image.
    featureVector=[]
    featureVector.append((1,2))
    featureVector.append((2,3))
    featureVector.append((5))
    featureVector.append((0))
    return featureVector

def getFeatureVectors(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    featureVectors=[[] for i in range(len(trainingDataImages))] 
    for i in range(len(trainingDataImages)):
        featureVectors[i]=getFeatureVector(trainingDataImages[i])
    return [],featureVectors


