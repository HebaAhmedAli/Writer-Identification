def getFeatureVector(writerWithCoresspondingImages):
    # TO DO: Write our method for extracting the feature vector.
    featureVector=[]
    featureVector.append((1,2))
    featureVector.append((2,3))
    featureVector.append((5))
    featureVector.append((0))
    return featureVector

def getFeatureVectors(writersWithCorrespondingImages):
    # Initialize the vectors of each writer with empty vector.
    # Largest Id in the dataset is 671 for 657 writer.
    featureVectors=[[] for i in range(671)] 
    for i in range(len(writersWithCorrespondingImages)):
        featureVectors[i]=getFeatureVector(writersWithCorrespondingImages[i])
    return featureVectors


