import FeatureExtraction as featureExtraction
import Preprocessing as preprocessing
from matplotlib import pyplot as plt
import cv2
import numpy

# TODO: Delete after testing.
def show(image):
    # Figure size in inches
    plt.figure(figsize=(10, 10))
    # Display an image on the axes, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')


# Get the normalized eculidean dist.
def getNormalizedDist(featureVector,featureVectorToCompare):
    totalDist=0
    for i in range(len(featureVector)):
        a = numpy.array(featureVector[i])
        b = numpy.array(featureVectorToCompare[i])
        dist = numpy.linalg.norm(a-b)
        totalDist+=abs(dist)
        
    return totalDist/len(featureVector)

# Read the given images of the writer we want to identify.
def readWriterImages(imagesNum):
    writerImages=[]
    x=1
    writerImages=[]
    for i in range(imagesNum):
       image= cv2.imread('writerImagesToIdentify/'+str(x)+'.png')
       processedImage=preprocessing.processImage(image)
       x=x+1
       writerImages.append(processedImage);
       #show(processedImage)
    return writerImages
    
def identifyWriter(method,featureVectors,k,imagesNum):
    # TODO: Modify to be knn instead of nearest one neighbour
    writerImages=readWriterImages(imagesNum)
    featureVector=featureExtraction.extractFeaturesDuringIdentification(method,writerImages)
    nearestWriter=-1
    minDist=10000000
    for i in range(len(featureVectors)):
        newDist=getNormalizedDist(featureVector,featureVectors[i])
        if newDist < minDist:
            minDist=newDist
            nearestWriter=i
    print("The nearest writer id = "+str(nearestWriter))



