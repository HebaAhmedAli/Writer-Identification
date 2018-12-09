import Preprocessing as pp
import cv2

# The function input is array of writers images
def getFeatureVector(writerWithCoresspondingImages):
    # TO DO: Write our method for extracting the feature vector.
    featureVector=[]
    print("bo2 bo2  "+str(len(writerWithCoresspondingImages)))
    cnt = pp.splitImageIntoSmallImagesAndGetContors(writerWithCoresspondingImages,
            writerWithCoresspondingImages.shape[1], writerWithCoresspondingImages.shape[0])

    return featureVector

def getFeatureVectors(trainingDataImages):
    # Initialize the vectors of each image with empty vector.
    featureVectors=[[] for i in range(len(trainingDataImages))] 
    for i in range(len(trainingDataImages)):
        featureVectors[i]=getFeatureVector(trainingDataImages[i])
    return [],featureVectors


